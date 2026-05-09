"""Backtest the points projection model.

For each game in a held-out window, project points using only
data BEFORE that game, then compare to actual points scored.
Reports calibration on probability predictions across hypothetical lines.
"""
from __future__ import annotations
import sqlite3
from datetime import datetime
import numpy as np
import pandas as pd

from src.nba_pipeline.config import DB_PATH
from src.nba_pipeline.points_model import project_points


def main():
    conn = sqlite3.connect(DB_PATH)

    test_start = "2025-03-15"
    test_end = "2025-04-13"

    # Players with 20+ pre-test games and avg 15+ minutes
    players = pd.read_sql("""
        SELECT pb.player_id, p.full_name, COUNT(*) AS pre_games
        FROM player_box pb JOIN players p ON pb.player_id = p.player_id
        JOIN games g ON pb.game_id = g.game_id
        WHERE g.game_date < ? AND pb.minutes > 15
        GROUP BY pb.player_id
        HAVING pre_games >= 20
    """, conn, params=(test_start,))

    print(f"Testing on {len(players)} players")

    actuals = pd.read_sql("""
        SELECT pb.player_id, g.game_date, pb.minutes, pb.points
        FROM player_box pb JOIN games g ON pb.game_id = g.game_id
        WHERE g.game_date >= ? AND g.game_date <= ?
          AND pb.minutes > 5
    """, conn, params=(test_start, test_end))
    actuals = actuals[actuals["player_id"].isin(players["player_id"])]
    print(f"Test rows: {len(actuals)}")

    print("Running projections (~1-2 minutes)...")
    rows = []
    for i, r in enumerate(actuals.itertuples(index=False)):
        if i % 200 == 0:
            print(f"  {i}/{len(actuals)}...")
        proj = project_points(r.player_id, as_of_date=r.game_date)
        rows.append({
            "player_id": r.player_id,
            "game_date": r.game_date,
            "actual_pts": r.points,
            "actual_min": r.minutes,
            "pred_pts": proj.expected,
            "pred_std": proj.std,
            "error": proj.expected - r.points,
            "abs_error": abs(proj.expected - r.points),
        })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print("POINTS MODEL — POINT ESTIMATE METRICS")
    print("=" * 60)
    print(f"  Sample:           {len(df):,} player-games")
    print(f"  Bias (mean err):  {df['error'].mean():+.2f} pts")
    print(f"  MAE:              {df['abs_error'].mean():.2f} pts")
    print(f"  RMSE:             {np.sqrt((df['error']**2).mean()):.2f} pts")
    print(f"  Median abs error: {df['abs_error'].median():.2f}")

    # Naive baseline: predict each player's own season average
    print("\n  Naive baseline (predict player's pre-test season avg):")
    seasonal_avg = pd.read_sql("""
        SELECT pb.player_id, AVG(pb.points*1.0) AS avg_pts
        FROM player_box pb JOIN games g ON pb.game_id = g.game_id
        WHERE g.game_date < ? AND pb.minutes > 5
        GROUP BY pb.player_id
    """, conn, params=(test_start,))
    df_naive = df.merge(seasonal_avg, on="player_id")
    naive_mae = (df_naive["avg_pts"] - df_naive["actual_pts"]).abs().mean()
    print(f"    MAE: {naive_mae:.2f} pts")
    print(f"    Improvement over naive: {naive_mae - df['abs_error'].mean():+.2f} pts")

    # === CALIBRATION ANALYSIS ===
    # For each prediction, treat the predicted point as a proxy line and check
    # whether actual went over. If the model's std is calibrated, ~50% should
    # be over and ~50% under.
    print("\n" + "=" * 60)
    print("CALIBRATION CHECK")
    print("=" * 60)
    df["actually_over_pred"] = (df["actual_pts"] > df["pred_pts"]).astype(int)
    print(f"  P(actual > predicted) = {df['actually_over_pred'].mean():.3f}")
    print(f"  Expected if well-calibrated: 0.500")

    # For hypothetical lines at pred ± k*std, what's the actual hit rate?
    from scipy.stats import norm
    print("\n  Calibration at various confidence levels:")
    print("  (pretend our P(over) prediction is the line; check if actuals match)")
    print(f"  {'Predicted P(over)':<22}{'Actual hit rate':<18}{'N':<8}{'Calibrated?':<12}")
    bins = [(0.30, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 0.80)]
    for low, high in bins:
        # For this we need to compute P(over) at a specific line
        # Use the line = pred * 0.95 (5% under-pred) as a fake test line
        df["line_test"] = df["pred_pts"] * 0.95
        df["p_over"] = norm.cdf((df["pred_pts"] - df["line_test"]) / df["pred_std"])
        df["went_over"] = (df["actual_pts"] > df["line_test"]).astype(int)
        bucket = df[(df["p_over"] >= low) & (df["p_over"] < high)]
        if len(bucket) > 5:
            actual_rate = bucket["went_over"].mean()
            n = len(bucket)
            mid = (low + high) / 2
            calibrated = "✓" if abs(actual_rate - mid) < 0.05 else "off"
            print(f"  {low:.2f}-{high:.2f}              {actual_rate:.3f}             {n:<8}{calibrated}")
        else:
            print(f"  {low:.2f}-{high:.2f}              (insufficient sample)")

    # The most useful calibration: actual std vs predicted std
    print(f"\n  Mean predicted std:  {df['pred_std'].mean():.2f} pts")
    actual_resid_std = df['error'].std()
    print(f"  Actual residual std: {actual_resid_std:.2f} pts")
    if abs(df['pred_std'].mean() - actual_resid_std) < 1.0:
        print(f"  ✓ Std calibration is reasonable")
    elif df['pred_std'].mean() > actual_resid_std:
        print(f"  ⚠ Model is OVER-confident in uncertainty (preds std too high)")
    else:
        print(f"  ⚠ Model is UNDER-confident in uncertainty (preds std too low)")

    conn.close()


if __name__ == "__main__":
    main()
