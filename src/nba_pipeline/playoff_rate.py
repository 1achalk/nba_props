"""Playoff per-minute rate adjustment.

NBA playoff basketball has structurally lower scoring than regular season:
defenses tighten, possessions slow, refs swallow whistles in crunch time.
Players score fewer points per minute. The model uses regular-season-trained
per-minute rates which over-predict in playoff games.

This module computes a player-specific points-per-minute adjustment based on
their historical regular-season-vs-playoff PPM ratio. Applies only to the
points market by default.

Same scaling philosophy as playoff_minutes.py:
  - Confidence scales with playoff sample size
  - Multiplier is shrunk toward 1.0 by confidence
  - Clamped to safe bounds to prevent extreme adjustments

Why only points?
  - Backtest showed points overs at 34% WR (the bleeding market)
  - Other markets had small biases — adjusting them broadly risks net harm
  - Threes/rebounds/assists rate-changes in playoffs are noisier and smaller
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import pandas as pd

from .playoff_minutes import (
    is_playoff_date,
    _playoff_filter,
    _regular_season_filter,
)

# Markets supported. Add to this set later after backtesting on those markets.
SUPPORTED_MARKETS = {"points"}

# Map our market name → DB column
MARKET_TO_COL = {
    "points": "points",
    "rebounds": "rebounds",
    "assists": "assists",
    "threes": "fg3m",
}

# Bounds — playoff scoring effect is real but modest. ±15% covers it.
MIN_MULTIPLIER = 0.85
MAX_MULTIPLIER = 1.15
FULL_CONFIDENCE_PLAYOFF_GAMES = 10
MIN_PLAYOFF_GAMES = 3        # require more games than minutes adj — rates noisier
MIN_PLAYOFF_MINUTES = 60     # need real minute volume, not garbage time spikes


@dataclass
class PlayoffRateAdj:
    """Output of compute_playoff_rate_adjustment.

    multiplier: multiply baseline rate by this (1.0 = no adjustment)
    confidence: 0-1 based on playoff sample
    n_playoff: playoff games used
    debug: misc fields
    """
    multiplier: float
    confidence: float
    n_playoff: int
    debug: dict


def compute_playoff_rate_adjustment(
    player_id: int,
    market: str,
    as_of_date: str,
    conn: sqlite3.Connection,
) -> PlayoffRateAdj:
    """Compute multiplier for a player's per-minute stat rate in playoffs.

    Only applies when:
      - market is in SUPPORTED_MARKETS
      - as_of_date is in a playoff window
      - player has enough playoff sample
    
    Otherwise returns multiplier=1.0 (no-op).
    """
    if market not in SUPPORTED_MARKETS:
        return PlayoffRateAdj(1.0, 0.0, 0, {"reason": "market_not_supported"})

    if not is_playoff_date(as_of_date):
        return PlayoffRateAdj(1.0, 0.0, 0, {"reason": "not_playoff_date"})

    db_col = MARKET_TO_COL.get(market)
    if not db_col:
        return PlayoffRateAdj(1.0, 0.0, 0, {"reason": "unknown_market"})

    # Pull playoff games with this stat
    p_df = pd.read_sql(f"""
        SELECT pb.minutes, pb.{db_col} as stat
        FROM player_box pb
        JOIN games g ON pb.game_id = g.game_id
        WHERE pb.player_id = ?
          AND g.game_date < ?
          AND pb.minutes > 0
          AND {_playoff_filter()}
    """, conn, params=(player_id, as_of_date))

    # Pull regular-season games (last 2 seasons for relevance)
    r_df = pd.read_sql(f"""
        SELECT pb.minutes, pb.{db_col} as stat
        FROM player_box pb
        JOIN games g ON pb.game_id = g.game_id
        WHERE pb.player_id = ?
          AND g.game_date < ?
          AND g.game_date >= ?
          AND pb.minutes > 0
          AND {_regular_season_filter()}
    """, conn, params=(player_id, as_of_date, "2024-10-01"))

    n_p = len(p_df)
    n_r = len(r_df)
    playoff_minutes = p_df["minutes"].sum() if n_p > 0 else 0

    if n_p < MIN_PLAYOFF_GAMES or playoff_minutes < MIN_PLAYOFF_MINUTES or n_r < 10:
        return PlayoffRateAdj(1.0, 0.0, n_p,
                              {"reason": "insufficient_sample",
                               "n_p": n_p, "n_r": n_r,
                               "playoff_min": float(playoff_minutes)})

    # Compute weighted per-minute rates (total stat / total minutes)
    # Weighting by minutes is more robust than averaging per-game rates,
    # since it down-weights low-minute games that have noisy ratios.
    playoff_ppm = float(p_df["stat"].sum() / p_df["minutes"].sum())
    regular_ppm = float(r_df["stat"].sum() / r_df["minutes"].sum())

    if regular_ppm < 0.01:  # Avoid noise for low-PPM stats
        return PlayoffRateAdj(1.0, 0.0, n_p,
                              {"reason": "regular_ppm_too_low",
                               "regular_ppm": regular_ppm})

    raw_ratio = playoff_ppm / regular_ppm
    confidence = min(n_p / FULL_CONFIDENCE_PLAYOFF_GAMES, 1.0)

    # Shrink toward 1.0 by confidence (same as minutes adj)
    adjusted = 1.0 + confidence * (raw_ratio - 1.0)
    multiplier = max(MIN_MULTIPLIER, min(MAX_MULTIPLIER, adjusted))

    return PlayoffRateAdj(
        multiplier=multiplier,
        confidence=confidence,
        n_playoff=n_p,
        debug={
            "playoff_ppm": round(playoff_ppm, 4),
            "regular_ppm": round(regular_ppm, 4),
            "raw_ratio": round(raw_ratio, 3),
            "shrunk_ratio": round(adjusted, 3),
            "n_r": n_r,
            "playoff_minutes": float(playoff_minutes),
        },
    )
