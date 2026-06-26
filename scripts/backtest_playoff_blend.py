"""Backtest the playoff blend against the 487 graded picks in picks_log.

Compares the OLD model_pred (already stored) against the NEW projection
(with playoff blend) for each graded pick, then re-grades based on the
new projection's implied probability vs the book's odds.

Reports:
  - Old vs New MAE on point predictions
  - Old vs New win rate, P/L, ROI if you'd bet only picks where new model
    still shows positive edge
  - Side-by-side over/under breakdown

Run after applying the integration steps in INTEGRATION_NOTES.md.
"""
from __future__ import annotations
import sqlite3

import pandas as pd

from src.nba_pipeline.config import DB_PATH
from src.nba_pipeline.stat_model import project_stat


STAT_COL_MAP = {
    "points": "points", "rebounds": "rebounds", "assists": "assists",
    "threes": "fg3m", "steals": "steals", "blocks": "blocks",
}


def implied_prob(american_odds: int) -> float:
    if american_odds > 0:
        return 100.0 / (american_odds + 100.0)
    return abs(american_odds) / (abs(american_odds) + 100.0)


def main():
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql("""
        SELECT id, pick_date, player_name, player_id, market, side, line,
               book_odds, book_implied, model_prob AS old_model_prob,
               model_pred AS old_model_pred, model_std AS old_model_std,
               edge AS old_edge, opp_team, actual, won, pl
        FROM picks_log
        WHERE won IS NOT NULL
        ORDER BY pick_date, id
    """, conn)

    print(f"Loaded {len(df)} graded picks across {df['pick_date'].nunique()} dates")
    print(f"Date range: {df['pick_date'].min()} to {df['pick_date'].max()}\n")

    # Re-project each pick using the new model
    print("Re-projecting with playoff blend... (this takes ~1-2 minutes)")
    new_preds = []
    new_probs = []
    new_edges = []
    for i, r in enumerate(df.itertuples(index=False)):
        if i % 50 == 0:
            print(f"  {i}/{len(df)}")
        try:
            # Resolve opponent team_id
            opp_team_id = None
            if r.opp_team:
                row = conn.execute(
                    "SELECT team_id FROM teams WHERE abbreviation = ?",
                    (r.opp_team,)
                ).fetchone()
                if row:
                    opp_team_id = int(row[0])
            
            proj = project_stat(
                r.player_id, r.market,
                as_of_date=r.pick_date,
                opponent_team_id=opp_team_id,
            )
            
            if r.side == "over":
                p = proj.p_over(r.line)
            else:
                p = proj.p_under(r.line)
            
            new_preds.append(proj.expected)
            new_probs.append(p)
            new_edges.append(p - r.book_implied)
        except Exception as e:
            print(f"  WARN re-proj failed for {r.player_name} {r.market}: {e}")
            new_preds.append(r.old_model_pred)
            new_probs.append(r.old_model_prob)
            new_edges.append(r.old_edge)

    df["new_pred"] = new_preds
    df["new_prob"] = new_probs
    df["new_edge"] = new_edges
    df["pred_diff"] = df["new_pred"] - df["old_model_pred"]

    # --- Point-estimate accuracy ---
    print("\n" + "=" * 70)
    print("POINT-ESTIMATE ACCURACY")
    print("=" * 70)
    print(f"{'':<15}{'OLD':>12}{'NEW':>12}{'IMPROVEMENT':>15}")
    
    old_mae = (df["old_model_pred"] - df["actual"]).abs().mean()
    new_mae = (df["new_pred"] - df["actual"]).abs().mean()
    print(f"{'MAE':<15}{old_mae:>12.3f}{new_mae:>12.3f}{old_mae - new_mae:>+15.3f}")
    
    old_bias = (df["old_model_pred"] - df["actual"]).mean()
    new_bias = (df["new_pred"] - df["actual"]).mean()
    print(f"{'Bias':<15}{old_bias:>+12.3f}{new_bias:>+12.3f}{abs(old_bias) - abs(new_bias):>+15.3f}")
    
    print(f"\nMean prediction change: {df['pred_diff'].mean():+.3f}")
    print(f"Predictions moved down: {(df['pred_diff'] < 0).sum()} ({(df['pred_diff'] < 0).mean()*100:.0f}%)")
    print(f"Predictions moved up:   {(df['pred_diff'] > 0).sum()} ({(df['pred_diff'] > 0).mean()*100:.0f}%)")

    # --- By market and side: where did the new model help? ---
    print("\n" + "=" * 70)
    print("BIAS BY MARKET × SIDE  (positive bias = model predicts too high)")
    print("=" * 70)
    print(f"{'Market':<10}{'Side':<8}{'N':<6}{'OLD bias':>12}{'NEW bias':>12}{'Δ':>10}")
    for (mkt, side), g in df.groupby(["market", "side"]):
        ob = (g["old_model_pred"] - g["actual"]).mean()
        nb = (g["new_pred"] - g["actual"]).mean()
        print(f"{mkt:<10}{side:<8}{len(g):<6}{ob:>+12.2f}{nb:>+12.2f}{abs(ob) - abs(nb):>+10.2f}")

    # --- Hypothetical P/L if we'd used new model edges ---
    # Filter to picks where the NEW model still says positive edge at our 5% threshold
    print("\n" + "=" * 70)
    print("HYPOTHETICAL P/L: picks that NEW model would still recommend (edge >= 5%)")
    print("=" * 70)
    new_recommended = df[df["new_edge"] >= 0.05].copy()
    print(f"  Picks NEW model still recommends: {len(new_recommended)} / {len(df)}")
    print(f"  Of those, win rate: {new_recommended['won'].mean()*100:.1f}%")
    print(f"  Total P/L:          {new_recommended['pl'].sum():+.2f} units")
    if len(new_recommended) > 0:
        print(f"  ROI:                {new_recommended['pl'].sum()/len(new_recommended)*100:+.1f}%")
    
    print("\n  By market on NEW-recommended:")
    for mkt, g in new_recommended.groupby("market"):
        print(f"    {mkt:<10}{len(g):<4} picks  WR={g['won'].mean()*100:.0f}%  ROI={g['pl'].sum()/len(g)*100:+.0f}%")

    print("\n  By side on NEW-recommended:")
    for side, g in new_recommended.groupby("side"):
        print(f"    {side:<8}{len(g):<4} picks  WR={g['won'].mean()*100:.0f}%  ROI={g['pl'].sum()/len(g)*100:+.0f}%")

    # --- Picks the new model dropped (used to recommend, no longer does) ---
    dropped = df[(df["old_edge"] >= 0.05) & (df["new_edge"] < 0.05)]
    if len(dropped) > 0:
        print(f"\n--- Picks NEW model DROPPED (old edge >= 5%, new edge < 5%): {len(dropped)} ---")
        print(f"  Old win rate on dropped: {dropped['won'].mean()*100:.1f}%")
        print(f"  Old P/L on dropped:      {dropped['pl'].sum():+.2f} units")
        print(f"  (We avoided losing money here if these were bad bets)")
    
    # --- Picks the new model adds ---
    added = df[(df["old_edge"] < 0.05) & (df["new_edge"] >= 0.05)]
    if len(added) > 0:
        print(f"\n--- Picks NEW model ADDED (old edge < 5%, new edge >= 5%): {len(added)} ---")
        print(f"  Win rate on added: {added['won'].mean()*100:.1f}%")
        print(f"  P/L on added:      {added['pl'].sum():+.2f} units")

    conn.close()


if __name__ == "__main__":
    main()
