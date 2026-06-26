"""Points projection model (v2 - Opponent Adjusted).

Predicts a player's points distribution for a game by combining:
  1. Injury-adjusted, positionally distributed Minutes
  2. Baseline Points-Per-Minute (PPM)
  3. PBP Opponent Defensive Context (Def RTG / Pace / Shot Profile)
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import DB_PATH
from .minutes_model import project_minutes, season_for_date, season_start_date, _shrinkage
from .opponent_context import get_opponent_for_player_game

# Hyperparameters
RECENT_WINDOW_GAMES = 10
MIN_GAMES_FOR_FULL_TRUST = 25
DEFAULT_PPM_PRIOR = 0.50

@dataclass
class PointsProjection:
    expected: float
    std: float
    p_play: float
    n_games: int
    minutes_proj: dict
    ppm_debug: dict
    opp_context: dict

def project_points(player_id: int, as_of_date: str = None, game_id: str = None, db_path: Path = DB_PATH) -> PointsProjection:
    """Project a player's points distribution for one game (v1 baseline model).

    Multiplies conditional projected minutes by a sample-size-weighted
    points-per-minute estimate and an opponent-defense multiplier, with a simple
    proportional standard deviation. Retained as the v1 baseline that the generic
    stat_model.project_stat (normal/negative-binomial distributions) is
    benchmarked against in compare_models.py and backtest_points.py.
    """
    if as_of_date is None:
        as_of_date = datetime.now().strftime("%Y-%m-%d")
        
    # 1) Get the highly accurate, injury-adjusted minutes
    minutes_proj = project_minutes(player_id, as_of_date, db_path)
    
    if minutes_proj.expected <= 0.1 or minutes_proj.p_play == 0:
        return PointsProjection(0.0, 0.0, 0.0, 0, {}, {}, {})

    # 2) Calculate Baseline Points-Per-Minute (PPM)
    conn = sqlite3.connect(db_path)
    dt = datetime.fromisoformat(as_of_date.replace("Z", "+00:00")).replace(tzinfo=None)
    season = season_for_date(dt)
    s_start = season_start_date(season)

    df_season = pd.read_sql("""
        SELECT pb.minutes, pb.points FROM player_box pb JOIN games g ON pb.game_id = g.game_id
        WHERE pb.player_id = ? AND g.game_date >= ? AND g.game_date < ? AND pb.minutes > 0
        ORDER BY g.game_date DESC
    """, conn, params=(player_id, s_start, as_of_date))

    df_career = pd.read_sql("""
        SELECT pb.minutes, pb.points FROM player_box pb JOIN games g ON pb.game_id = g.game_id
        WHERE pb.player_id = ? AND g.game_date < ? AND pb.minutes > 0
        ORDER BY g.game_date DESC LIMIT 82
    """, conn, params=(player_id, s_start))

    df_season["ppm"] = df_season["points"] / df_season["minutes"]
    df_career["ppm"] = df_career["points"] / df_career["minutes"]

    n_season = len(df_season)
    n_recent = min(n_season, RECENT_WINDOW_GAMES)
    
    season_ppm_mean = df_season["ppm"].mean() if n_season > 0 else DEFAULT_PPM_PRIOR
    recent_ppm_mean = df_season["ppm"].head(n_recent).mean() if n_recent > 0 else season_ppm_mean
    career_ppm_mean = df_career["ppm"].mean() if len(df_career) > 0 else DEFAULT_PPM_PRIOR

    w_season = _shrinkage(n_season, MIN_GAMES_FOR_FULL_TRUST) * 0.5
    w_recent = _shrinkage(n_recent, RECENT_WINDOW_GAMES) * 0.3
    w_career = _shrinkage(len(df_career), 82) * 0.15
    w_prior = max(0, 1.0 - (w_season + w_recent + w_career))

    expected_ppm = (w_season * season_ppm_mean + w_recent * recent_ppm_mean + 
                    w_career * career_ppm_mean + w_prior * DEFAULT_PPM_PRIOR)

    # 3) Apply Opponent Context Adjustment
    opp_adjustment = 1.0
    opp_debug = {}
    
    if game_id:
        try:
            opp_ctx = get_opponent_for_player_game(player_id, game_id, db_path)
            if opp_ctx:
                # pts_adj is already computed in your opponent_context.py 
                # based on Def RTG and Pace
                opp_adjustment = opp_ctx.pts_adj
                opp_debug = {
                    "team_id": opp_ctx.team_id,
                    "def_rtg": opp_ctx.def_rtg,
                    "pace": opp_ctx.pace,
                    "pts_adj": opp_ctx.pts_adj
                }
        except Exception:
            pass # Fallback to 1.0 if opponent data is missing

    conn.close()

    # 4) Final Math: Conditional Expectation
    # Sportsbook bets void if the player doesn't play. We must project their stats 
    # based ONLY on the minutes they play given they are active.
    conditional_minutes = minutes_proj.expected / minutes_proj.p_play if minutes_proj.p_play > 0 else 0
    
    expected_points = conditional_minutes * expected_ppm * opp_adjustment
    
    # Simple variance scaling for now (can upgrade to negative binomial later)
    points_std = expected_points * 0.35 if expected_points > 0 else 0.0

    return PointsProjection(
        expected=round(expected_points, 2),
        std=round(points_std, 2),
        p_play=minutes_proj.p_play,
        n_games=n_season + len(df_career),
        minutes_proj={"expected": minutes_proj.expected, "std": minutes_proj.std},
        ppm_debug={"expected_ppm": expected_ppm, "opp_adjustment": opp_adjustment},
        opp_context=opp_debug
    )
