"""Diagnose whether minutes projections are systematically too high.

For each graded pick in picks_log, compare:
  - What minutes did the model project (we have to re-run since we didn't log it)
  - What minutes did the player actually play

If projected > actual systematically, that's the source of the over-prediction
bias. Per-minute rates could be correct, but inflated minutes still produce
inflated totals.

This is a pure diagnostic — no model changes made.
"""
from __future__ import annotations
import sqlite3

import pandas as pd

from src.nba_pipeline.config import DB_PATH
from src.nba_pipeline.minutes_model import project_minutes


def main():
    conn = sqlite3.connect(DB_PATH)

    # Get unique player-game combos from graded picks
    df = pd.read_sql("""
        SELECT DISTINCT pl.player_id, pl.player_name, pl.pick_date, pl.opp_team
        FROM picks_log pl
        WHERE pl.won IS NOT NULL
        ORDER BY pl.pick_date, pl.player_id
    """, conn)

    print(f"Diagnosing minutes for {len(df)} unique player-games...")
    print("(this takes ~1-2 minutes)")

    rows = []
    for i, r in enumerate(df.itertuples(index=False)):
        if i % 20 == 0:
            print(f"  {i}/{len(df)}")
        
        # Look up actual minutes for this player on this date
        actual_row = conn.execute("""
            SELECT pb.minutes FROM player_box pb
            JOIN games g ON pb.game_id = g.game_id
            WHERE pb.player_id = ? AND g.game_date = ?
            LIMIT 1
        """, (r.player_id, r.pick_date)).fetchone()
        
        if not actual_row:
            continue
        
        actual_min = float(actual_row[0])
        
        try:
            mp = project_minutes(r.player_id, as_of_date=r.pick_date)
            projected_min = mp.expected
            p_play = mp.p_play
        except Exception:
            continue
        
        rows.append({
            "player_name": r.player_name,
            "pick_date": r.pick_date,
            "projected_min": projected_min,
            "actual_min": actual_min,
            "p_play": p_play,
            "diff": projected_min - actual_min,
        })

    res = pd.DataFrame(rows)
    
    if len(res) == 0:
        print("\nNo data to analyze.")
        return

    print("\n" + "=" * 70)
    print("MINUTES PROJECTION DIAGNOSTIC")
    print("=" * 70)
    print(f"\nSample size: {len(res)} player-games")
    print(f"\nOverall:")
    print(f"  Mean projected min: {res['projected_min'].mean():.2f}")
    print(f"  Mean actual min:    {res['actual_min'].mean():.2f}")
    print(f"  Mean diff (proj - actual): {res['diff'].mean():+.2f}")
    print(f"  MAE on minutes:     {res['diff'].abs().mean():.2f}")
    print(f"  Median diff:        {res['diff'].median():+.2f}")
    
    print(f"\n  Over-projecting (proj > actual): "
          f"{(res['diff'] > 1).sum()} ({(res['diff'] > 1).mean()*100:.0f}%)")
    print(f"  Under-projecting (actual > proj): "
          f"{(res['diff'] < -1).sum()} ({(res['diff'] < -1).mean()*100:.0f}%)")
    print(f"  Close (within ±1 min): "
          f"{((res['diff'] >= -1) & (res['diff'] <= 1)).sum()} ({((res['diff'] >= -1) & (res['diff'] <= 1)).mean()*100:.0f}%)")
    
    # Where is the bias coming from? By bucket of projected minutes
    print(f"\nBy projected-minutes bucket:")
    res["bucket"] = pd.cut(res["projected_min"], bins=[0, 15, 25, 32, 40, 48],
                            labels=["0-15", "15-25", "25-32", "32-40", "40+"])
    for bucket, g in res.groupby("bucket", observed=True):
        if len(g) > 0:
            print(f"  {bucket:<10}{len(g):>4} games  "
                  f"proj={g['projected_min'].mean():>5.1f}  "
                  f"actual={g['actual_min'].mean():>5.1f}  "
                  f"diff={g['diff'].mean():>+5.2f}")
    
    # Largest over-projections
    print(f"\nLargest over-projections (proj - actual):")
    biggest_over = res.nlargest(10, "diff")
    for _, r in biggest_over.iterrows():
        print(f"  {r['pick_date']} {r['player_name'][:25]:<25} "
              f"proj={r['projected_min']:.1f}  actual={r['actual_min']:.1f}  "
              f"diff={r['diff']:+.1f}")
    
    # Largest under-projections
    print(f"\nLargest under-projections (actual - proj):")
    biggest_under = res.nsmallest(10, "diff")
    for _, r in biggest_under.iterrows():
        print(f"  {r['pick_date']} {r['player_name'][:25]:<25} "
              f"proj={r['projected_min']:.1f}  actual={r['actual_min']:.1f}  "
              f"diff={r['diff']:+.1f}")
    
    # By date — does the bias vary by slate?
    print(f"\nBy date:")
    for date, g in res.groupby("pick_date"):
        print(f"  {date}: {len(g):>4} games  "
              f"avg proj={g['projected_min'].mean():>5.2f}  "
              f"avg actual={g['actual_min'].mean():>5.2f}  "
              f"bias={g['diff'].mean():>+5.2f}")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    overall_bias = res['diff'].mean()
    if overall_bias > 1.5:
        pct_pts_inflation = overall_bias / res['actual_min'].mean()
        print(f"\n⚠️  Minutes are over-projected by {overall_bias:+.1f} min on average.")
        print(f"   At average actual minutes of {res['actual_min'].mean():.1f}, that's "
              f"{pct_pts_inflation*100:.0f}% inflation.")
        print(f"\n   If we fixed minutes, expected points reduction:")
        print(f"     - Player averaging 1.0 PPM: ~{overall_bias:.1f} pts less projected")
        print(f"     - Player averaging 0.5 PPM: ~{overall_bias*0.5:.1f} pts less projected")
        print(f"\n   This would meaningfully reduce the over-bias on points lines.")
    elif overall_bias > 0.5:
        print(f"\n  Minutes over-projected by {overall_bias:+.1f} min — modest bias.")
        print(f"  Fixing minutes alone won't fully resolve over-bias on points.")
    elif overall_bias < -0.5:
        print(f"\n  Minutes are UNDER-projected by {abs(overall_bias):.1f} min on average.")
        print(f"  So the over-prediction bias is NOT from minutes — it's from per-minute rates.")
    else:
        print(f"\n  Minutes projection is roughly calibrated (bias {overall_bias:+.2f}).")
        print(f"  The over-prediction bias must come from per-minute rates, not minutes.")
        print(f"  This points to a league-wide playoff scoring effect (Option A).")
    
    conn.close()


if __name__ == "__main__":
    main()
