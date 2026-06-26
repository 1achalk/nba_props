"""Playoff minutes adjustment.

NBA playoff rotations are tighter than regular season — coaches lean on top
6-8 players, end-of-bench guys get cut. Regular-season minutes projections
are systematically wrong for playoff games as a result:
  - Rotation players: over-projected (get 5-8 min, model says 20)
  - Starters: under-projected (play 38-44, model says 30-34)

This module computes a player-specific multiplier based on their historical
regular-season-vs-playoff minutes ratio. Apply to baseline minutes projection
when projecting a playoff game.

Signal scaling — high confidence at 10+ playoff games, none at 0.
Magnitude clamped to [0.5, 1.3] to prevent extreme adjustments from small samples.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import pandas as pd

# Same playoff windows as playoff_blend.py — keep consistent
PLAYOFF_WINDOWS = [
    ("2024-04-15", "2024-06-30"),
    ("2025-04-15", "2025-06-30"),
    ("2026-04-15", "2026-06-30"),
]

# Bounds on the multiplier — prevents extreme adjustments from small samples
MIN_MULTIPLIER = 0.50
MAX_MULTIPLIER = 1.30
FULL_CONFIDENCE_PLAYOFF_GAMES = 10
MIN_PLAYOFF_GAMES = 1


def _playoff_filter() -> str:
    clauses = [f"(g.game_date >= '{a}' AND g.game_date <= '{b}')"
               for a, b in PLAYOFF_WINDOWS]
    return "(" + " OR ".join(clauses) + ")"


def _regular_season_filter() -> str:
    """Inverse: games NOT in any playoff window."""
    clauses = [f"NOT (g.game_date >= '{a}' AND g.game_date <= '{b}')"
               for a, b in PLAYOFF_WINDOWS]
    return "(" + " AND ".join(clauses) + ")"


@dataclass
class PlayoffMinutesAdj:
    """Output of compute_playoff_minutes_adjustment.
    
    multiplier: multiply baseline minutes by this (1.0 = no adjustment)
    confidence: 0-1 based on playoff sample size
    n_playoff: playoff games used
    n_regular: regular season games used
    debug: misc fields
    """
    multiplier: float
    confidence: float
    n_playoff: int
    n_regular: int
    debug: dict


def is_playoff_date(as_of_date: str) -> bool:
    """Is this date in any playoff window?"""
    for start, end in PLAYOFF_WINDOWS:
        if start <= as_of_date <= end:
            return True
    return False


def compute_playoff_minutes_adjustment(
    player_id: int,
    as_of_date: str,
    conn: sqlite3.Connection,
) -> PlayoffMinutesAdj:
    """Compute how much to adjust regular-season minutes for playoff context.
    
    Returns a multiplier centered at 1.0. Values < 1.0 mean the player gets
    fewer minutes in playoffs than regular season (rotation guys); > 1.0
    means more (starters).
    """
    # Only apply if we're actually projecting a playoff game
    if not is_playoff_date(as_of_date):
        return PlayoffMinutesAdj(
            multiplier=1.0, confidence=0.0,
            n_playoff=0, n_regular=0,
            debug={"reason": "not_playoff_date"},
        )

    # Pull playoff games
    p_df = pd.read_sql(f"""
        SELECT pb.minutes FROM player_box pb
        JOIN games g ON pb.game_id = g.game_id
        WHERE pb.player_id = ?
          AND g.game_date < ?
          AND pb.minutes > 0
          AND {_playoff_filter()}
    """, conn, params=(player_id, as_of_date))

    # Pull regular-season games (limit to last 2 seasons for relevance)
    r_df = pd.read_sql(f"""
        SELECT pb.minutes FROM player_box pb
        JOIN games g ON pb.game_id = g.game_id
        WHERE pb.player_id = ?
          AND g.game_date < ?
          AND g.game_date >= ?
          AND pb.minutes > 0
          AND {_regular_season_filter()}
    """, conn, params=(player_id, as_of_date, "2024-10-01"))

    n_p = len(p_df)
    n_r = len(r_df)

    if n_p < MIN_PLAYOFF_GAMES or n_r < 10:
        # Not enough playoff data, OR not enough regular-season comparison
        return PlayoffMinutesAdj(
            multiplier=1.0, confidence=0.0,
            n_playoff=n_p, n_regular=n_r,
            debug={"reason": "insufficient_sample"},
        )

    playoff_mean = float(p_df["minutes"].mean())
    regular_mean = float(r_df["minutes"].mean())

    if regular_mean < 1.0:  # Avoid division by zero
        return PlayoffMinutesAdj(
            multiplier=1.0, confidence=0.0,
            n_playoff=n_p, n_regular=n_r,
            debug={"reason": "zero_regular_mean"},
        )

    raw_ratio = playoff_mean / regular_mean
    confidence = min(n_p / FULL_CONFIDENCE_PLAYOFF_GAMES, 1.0)

    # Shrink toward 1.0 based on confidence. Low confidence = barely move from 1.0.
    # multiplier = 1.0 + confidence * (raw_ratio - 1.0)
    adjusted = 1.0 + confidence * (raw_ratio - 1.0)
    # Clamp to safe bounds
    multiplier = max(MIN_MULTIPLIER, min(MAX_MULTIPLIER, adjusted))

    return PlayoffMinutesAdj(
        multiplier=multiplier,
        confidence=confidence,
        n_playoff=n_p,
        n_regular=n_r,
        debug={
            "playoff_mean_min": round(playoff_mean, 2),
            "regular_mean_min": round(regular_mean, 2),
            "raw_ratio": round(raw_ratio, 3),
            "shrunk_ratio": round(adjusted, 3),
        },
    )
