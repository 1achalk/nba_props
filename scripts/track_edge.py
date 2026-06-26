"""Rolling realized-edge tracker.

The single source of truth for "is the model actually working?" during the
freeze-and-collect period. Excludes the known bug-corrupted picks (wrong-side
recommendations from the pre-fix p_under inversion), computes realized edge
per slate, and shows the cumulative trend.

Run anytime after grading:
    python -m scripts.track_edge

The number that matters: cumulative realized edge across clean slates.
If it stabilizes positive over 15+ slates, the model has a real edge.
If it decays toward zero, it doesn't. Either way you'll see it clearly.
"""
from __future__ import annotations
import sqlite3

import numpy as np
import pandas as pd

from src.nba_pipeline.config import DB_PATH

# The bug signature: recommended side contradicts the projection by a wide
# margin. These came from the pre-2026-05-17 p_under inversion bug.
# Any pick matching this is excluded from clean evaluation.
BUG_FILTER = """
    NOT ((side='under' AND model_pred > line * 1.3)
      OR (side='over'  AND model_pred < line * 0.7))
"""

# Date the conditional-probability fix went live. Picks on/after this date
# are "clean by construction"; before, we filter heuristically.
FIX_DATE = "2026-05-17"


def main():
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql(f"""
        SELECT pick_date, market, side, line, won, pl, edge,
               book_implied, model_pred, model_prob
        FROM picks_log
        WHERE won IS NOT NULL AND {BUG_FILTER}
        ORDER BY pick_date
    """, conn)

    if len(df) == 0:
        print("No clean graded picks yet.")
        conn.close()
        return

    print("=" * 72)
    print("ROLLING REALIZED-EDGE TRACKER  (clean picks only)")
    print("=" * 72)

    total_n = len(df)
    total_wr = df["won"].mean() * 100
    total_roi = df["pl"].sum() / len(df) * 100
    total_re = (df["won"].mean() - df["book_implied"].mean()) * 100
    total_pe = df["edge"].mean() * 100

    print(f"\nCumulative ({total_n} clean graded picks):")
    print(f"  Win rate:       {total_wr:.1f}%")
    print(f"  ROI:            {total_roi:+.1f}%")
    print(f"  Realized edge:  {total_re:+.1f}%  (model predicted {total_pe:+.1f}%)")
    print(f"  Edge capture:   {total_re/total_pe*100:.0f}% of predicted" if total_pe > 0 else "")

    # Per-slate breakdown with running cumulative realized edge
    print(f"\n{'Date':<12}{'N':>5}{'WR':>8}{'ROI':>9}{'RealEdge':>10}{'CumRealEdge':>13}{'CumROI':>9}")
    print("-" * 72)
    
    cum_rows = []
    for date, g in df.groupby("pick_date"):
        slate_wr = g["won"].mean() * 100
        slate_roi = g["pl"].sum() / len(g) * 100
        slate_re = (g["won"].mean() - g["book_implied"].mean()) * 100
        
        cum_rows.append(g)
        cum = pd.concat(cum_rows)
        cum_re = (cum["won"].mean() - cum["book_implied"].mean()) * 100
        cum_roi = cum["pl"].sum() / len(cum) * 100
        
        flag = ""
        if slate_re > 2:
            flag = " +"
        elif slate_re < -2:
            flag = " -"
        
        print(f"{date:<12}{len(g):>5}{slate_wr:>7.1f}%{slate_roi:>+8.1f}%"
              f"{slate_re:>+9.1f}%{cum_re:>+12.1f}%{cum_roi:>+8.1f}%{flag}")

    # Sparkline of cumulative realized edge
    cum_re_series = []
    cum_rows = []
    for date, g in df.groupby("pick_date"):
        cum_rows.append(g)
        cum = pd.concat(cum_rows)
        cum_re_series.append((cum["won"].mean() - cum["book_implied"].mean()) * 100)
    
    print(f"\nCumulative realized-edge trend:")
    _sparkline(cum_re_series)

    # Stability assessment
    print(f"\n{'=' * 72}")
    print("ASSESSMENT")
    print("=" * 72)
    n_slates = df["pick_date"].nunique()
    
    if n_slates < 8:
        print(f"\n  {n_slates} clean slates so far. Need ~15 for a confident read.")
        print(f"  Current cumulative realized edge: {total_re:+.1f}%")
        print(f"  Keep collecting. Don't change the model.")
    else:
        # Look at last 8 slates' realized edge stability
        recent_dates = sorted(df["pick_date"].unique())[-8:]
        recent = df[df["pick_date"].isin(recent_dates)]
        recent_re = (recent["won"].mean() - recent["book_implied"].mean()) * 100
        # Per-slate realized edges for variance
        slate_res = []
        for date, g in recent.groupby("pick_date"):
            slate_res.append((g["won"].mean() - g["book_implied"].mean()) * 100)
        re_std = np.std(slate_res)
        
        print(f"\n  {n_slates} clean slates collected.")
        print(f"  Cumulative realized edge: {total_re:+.1f}%")
        print(f"  Last 8 slates realized edge: {recent_re:+.1f}% (std {re_std:.1f}%)")
        
        if total_re > 1.5 and recent_re > 0:
            print(f"\n  → Edge appears REAL and stable. Consider moving to")
            print(f"    fractional-Kelly bet sizing. Still paper-trade-validate first.")
        elif total_re > 0 and recent_re > -1:
            print(f"\n  → Marginal edge. Keep collecting — not decisive yet.")
        else:
            print(f"\n  → Edge has decayed toward/below zero. The model does not")
            print(f"    have a durable edge. Do not bet real money.")

    # By market — which markets carry the edge (clean)
    print(f"\n  By market (clean, all slates):")
    for m, g in df.groupby("market"):
        re = (g["won"].mean() - g["book_implied"].mean()) * 100
        print(f"    {m:<10}{len(g):>4} picks  WR={g['won'].mean()*100:.0f}%  "
              f"ROI={g['pl'].sum()/len(g)*100:+.0f}%  realized_edge={re:+.1f}%")

    conn.close()


def _sparkline(values):
    """Print a simple text sparkline."""
    if not values or len(values) < 2:
        print("  (need more slates)")
        return
    blocks = "▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    rng = hi - lo if hi > lo else 1
    line = "".join(blocks[min(int((v - lo) / rng * (len(blocks) - 1)), len(blocks) - 1)]
                    for v in values)
    print(f"  {line}")
    print(f"  range: {lo:+.1f}% to {hi:+.1f}%   latest: {values[-1]:+.1f}%")


if __name__ == "__main__":
    main()
