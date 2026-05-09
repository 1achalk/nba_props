"""Model comparison harness."""
from __future__ import annotations

import argparse
import sqlite3
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.nba_pipeline.config import DB_PATH


def model_normal_simple(player_id, as_of_date):
    from src.nba_pipeline.points_model import project_points
    p = project_points(player_id, as_of_date=as_of_date)
    pred = p.expected
    std = max(p.std, 0.5)
    def p_over(line):
        return float(1 - norm.cdf((line - pred) / std))
    return {"pred": pred, "std": std, "p_over": p_over}


def model_stat_market(market):
    from src.nba_pipeline.stat_model import project_stat
    def adapter(player_id, as_of_date):
        p = project_stat(player_id, market, as_of_date=as_of_date)
        return {"pred": p.expected, "std": p.std, "p_over": p.p_over}
    return adapter


def model_stat_market_with_opp(market):
    """Same as model_stat_market but applies opponent-context adjustment."""
    from src.nba_pipeline.stat_model import project_stat
    from src.nba_pipeline.opponent_context import get_opponent_for_player_game
    import sqlite3
    from src.nba_pipeline.config import DB_PATH

    def adapter(player_id, as_of_date):
        # Find the game on this date and the opponent
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute("""
                SELECT pb.game_id FROM player_box pb JOIN games g ON pb.game_id = g.game_id
                WHERE pb.player_id = ? AND g.game_date = ? LIMIT 1
            """, (player_id, as_of_date)).fetchone()
        opp_id = None
        if row is not None:
            game_id = row[0]
            opp_id = get_opponent_for_player_game(player_id, game_id)
        p = project_stat(player_id, market, as_of_date=as_of_date, opponent_team_id=opp_id)
        return {"pred": p.expected, "std": p.std, "p_over": p.p_over}
    return adapter


def backtest_model(name, model_fn, market, test_start, test_end):
    conn = sqlite3.connect(DB_PATH)
    stat_col_map = {
        "points": "points", "rebounds": "rebounds", "assists": "assists",
        "threes": "fg3m", "steals": "steals", "blocks": "blocks",
    }
    stat_col = stat_col_map.get(market, market)

    prior = pd.read_sql("""
        SELECT player_id FROM player_box pb JOIN games g ON pb.game_id = g.game_id
        WHERE g.game_date < ? AND pb.minutes > 15
        GROUP BY player_id HAVING COUNT(*) >= 20
    """, conn, params=(test_start,))

    actuals = pd.read_sql(f"""
        SELECT pb.player_id, g.game_date, pb.minutes, pb.{stat_col} AS actual
        FROM player_box pb JOIN games g ON pb.game_id = g.game_id
        WHERE g.game_date >= ? AND g.game_date <= ? AND pb.minutes > 5
    """, conn, params=(test_start, test_end))
    actuals = actuals[actuals["player_id"].isin(prior["player_id"])]
    conn.close()

    print(f"  [{name}] projecting {len(actuals)} games...")
    rows = []
    for r in actuals.itertuples(index=False):
        result = model_fn(r.player_id, r.game_date)
        rows.append({
            "actual": r.actual,
            "pred": result["pred"],
            "p_over_callable": result["p_over"],
        })
    return pd.DataFrame(rows)


def evaluate(df, name):
    df = df.copy()
    df["error"] = df["pred"] - df["actual"]
    df["abs_error"] = df["error"].abs()

    mae = df["abs_error"].mean()
    bias = df["error"].mean()
    rmse = np.sqrt((df["error"] ** 2).mean())
    median_ae = df["abs_error"].median()

    cal_records = []
    for _, row in df.iterrows():
        for offset in np.linspace(-12, 12, 25):
            line = max(0.5, row["pred"] + offset)
            p = row["p_over_callable"](line)
            went_over = 1 if row["actual"] > line else 0
            cal_records.append({"p_over": p, "went_over": went_over})
    cal = pd.DataFrame(cal_records)

    buckets = [(0.0,0.1),(0.1,0.2),(0.2,0.3),(0.3,0.4),(0.4,0.5),
               (0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,1.0)]
    bucket_results = []
    for low, high in buckets:
        b = cal[(cal["p_over"] >= low) & (cal["p_over"] < high)]
        if len(b) > 30:
            bucket_results.append({
                "bucket": f"{low:.1f}-{high:.1f}",
                "n": len(b),
                "predicted": b["p_over"].mean(),
                "actual": b["went_over"].mean(),
                "diff": b["went_over"].mean() - b["p_over"].mean(),
            })

    return {
        "name": name, "n": len(df), "mae": mae, "bias": bias,
        "rmse": rmse, "median_ae": median_ae, "buckets": bucket_results,
    }


def print_comparison(results):
    print("\n" + "=" * 70)
    print("POINT ESTIMATE METRICS")
    print("=" * 70)
    print(f"{'Model':<25}{'N':<8}{'MAE':<10}{'Bias':<10}{'RMSE':<10}{'Med AE':<10}")
    for r in results:
        print(f"{r['name']:<25}{r['n']:<8}{r['mae']:<10.3f}{r['bias']:<+10.3f}{r['rmse']:<10.3f}{r['median_ae']:<10.3f}")

    print("\n" + "=" * 70)
    print("CALIBRATION TABLES")
    print("=" * 70)
    for r in results:
        print(f"\n  {r['name']}")
        print(f"  {'Bucket':<10}{'N':<10}{'Predicted':<13}{'Actual':<13}{'Diff':<10}")
        for b in r["buckets"]:
            flag = "OK" if abs(b["diff"]) < 0.03 else ("warn" if abs(b["diff"]) < 0.08 else "BAD")
            print(f"  {b['bucket']:<10}{b['n']:<10}{b['predicted']:<13.3f}{b['actual']:<13.3f}{b['diff']:<+10.3f}{flag}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", default="points",
                        choices=["points","rebounds","assists","threes","steals","blocks"])
    parser.add_argument("--start", default="2025-03-15")
    parser.add_argument("--end", default="2025-04-13")
    args = parser.parse_args()

    print(f"Comparing models on {args.market} from {args.start} to {args.end}")

    models_to_test = []
    models_to_test.append((f"baseline_{args.market}", model_stat_market(args.market)))
    models_to_test.append((f"opp_aware_{args.market}", model_stat_market_with_opp(args.market)))
    if args.market == "points":
        models_to_test.append(("points_model_old", model_normal_simple))

    results = []
    for name, fn in models_to_test:
        df = backtest_model(name, fn, args.market, args.start, args.end)
        results.append(evaluate(df, name))

    print_comparison(results)


if __name__ == "__main__":
    main()
