"""Playoff baseline blending for stat projections.

Computes a separate playoff rate per player and blends it into the projection
based on (a) playoff sample size (confidence) and (b) statistical evidence
that playoff performance differs from regular season (signal).

Plug into stat_model.py by calling get_playoff_blend() and applying the
returned (rate, weight) pair to the existing blend.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Playoff-window date heuristic. Adjust if season_type field is reliable.
# NBA playoffs typically run mid-April through mid-June.
PLAYOFF_WINDOWS = [
    ("2024-04-15", "2024-06-30"),
    ("2025-04-15", "2025-06-30"),
    ("2026-04-15", "2026-06-30"),
]

MAX_PLAYOFF_WEIGHT = 0.40        # never exceed 40% playoff in the blend
FULL_CONFIDENCE_GAMES = 15       # n_playoff_games for full confidence
MIN_PLAYOFF_GAMES = 1            # require at least this many to blend at all


def _playoff_date_filter() -> str:
    """SQL WHERE clause matching any playoff-window date."""
    clauses = [f"(g.game_date >= '{a}' AND g.game_date <= '{b}')"
               for a, b in PLAYOFF_WINDOWS]
    return "(" + " OR ".join(clauses) + ")"


@dataclass
class PlayoffBlend:
    """Output of get_playoff_blend.
    
    rate: per-minute playoff rate (e.g. points-per-minute)
    weight: blend weight 0.0-MAX_PLAYOFF_WEIGHT to apply to this rate
    n_games: number of playoff games used
    confidence: data-quantity confidence 0-1
    signal_strength: statistical evidence the playoff rate differs from regular 0-1
    debug: misc fields for inspection
    """
    rate: float
    weight: float
    n_games: int
    confidence: float
    signal_strength: float
    debug: dict


def get_playoff_blend(
    player_id: int,
    stat_col: str,            # "points" / "rebounds" / "assists" / "fg3m" / "steals" / "blocks"
    as_of_date: str,
    conn: sqlite3.Connection,
    regular_rate: float,      # the existing blended regular-season rate, for comparison
) -> PlayoffBlend:
    """Compute playoff rate + appropriate blend weight for this player."""
    pdf = pd.read_sql(f"""
        SELECT pb.minutes, pb.{stat_col} AS stat
        FROM player_box pb
        JOIN games g ON pb.game_id = g.game_id
        WHERE pb.player_id = ?
          AND g.game_date < ?
          AND pb.minutes > 0
          AND {_playoff_date_filter()}
    """, conn, params=(player_id, as_of_date))

    n_playoff = len(pdf)
    if n_playoff < MIN_PLAYOFF_GAMES:
        return PlayoffBlend(
            rate=regular_rate, weight=0.0, n_games=0,
            confidence=0.0, signal_strength=0.0,
            debug={"reason": "no_playoff_games"},
        )

    pdf["rate"] = pdf["stat"] / pdf["minutes"]
    playoff_rate = float(pdf["rate"].mean())
    playoff_std = float(pdf["rate"].std(ddof=1)) if n_playoff > 1 else playoff_rate * 0.5

    # Confidence scales linearly with sample, capped at 1.0
    confidence = min(n_playoff / FULL_CONFIDENCE_GAMES, 1.0)

    # Signal strength: how far is playoff_rate from regular_rate, in units of SE?
    # SE = std / sqrt(n). t-stat = |playoff - regular| / SE.
    # Map t-stat to a 0-1 signal: t<1 → 0, t>=2.5 → 1, linear in between
    if n_playoff > 1 and playoff_std > 0:
        se = playoff_std / np.sqrt(n_playoff)
        t_stat = abs(playoff_rate - regular_rate) / se if se > 0 else 0.0
        signal_strength = max(0.0, min(1.0, (t_stat - 1.0) / 1.5))
    else:
        # With only 1 playoff game, we have no idea if it's a real signal
        signal_strength = 0.0

    # Blend weight: scaled by both factors, capped
    weight = confidence * signal_strength * MAX_PLAYOFF_WEIGHT

    return PlayoffBlend(
        rate=playoff_rate,
        weight=weight,
        n_games=n_playoff,
        confidence=confidence,
        signal_strength=signal_strength,
        debug={
            "playoff_std": round(playoff_std, 4),
            "regular_rate": round(regular_rate, 4),
            "lift": round(playoff_rate - regular_rate, 4),
            "t_stat": round(t_stat, 2) if n_playoff > 1 and playoff_std > 0 else None,
        },
    )
