"""Opponent context using PBP Stats team defensive profiles.

Pulls from pbp_team_defense table. Market-specific adjustments based on
the most relevant PBP metric for that prop type.

Falls back to neutral (1.0) adjustment if PBP data unavailable.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .config import DB_PATH

# League baselines computed from PBP data (2024-25 averages)
LEAGUE_AVG_PACE = 98.6
LEAGUE_AVG_DEF_RTG = 114.8        # opp pts per 100 poss
LEAGUE_AVG_3PA_PG = 37.6           # opp 3PA per game
LEAGUE_AVG_3P_PCT = 0.351          # opp 3P%
LEAGUE_AVG_OPP_REB_PG = 44.0       # opp total rebounds per game (rough)
LEAGUE_AVG_OPP_AST_PG = 26.5

# Adjustment caps - prevent extreme values
ADJ_CAP_LOW = 0.92
ADJ_CAP_HIGH = 1.08


@dataclass
class OpponentContext:
    team_id: int
    season: str
    n_games: int
    pace: float
    def_rtg: float
    opp_3pa_pg: float
    opp_3p_pct: float
    opp_drb_pct: float
    opp_off_reb_pct: float
    opp_ast_pg: float
    
    pts_adj: float       # for points market
    threes_adj: float    # for threes market (combines 3PA and 3P%)
    reb_adj: float       # for rebounds market
    ast_adj: float       # for assists market

    def adjustment_for(self, market: str) -> float:
        # Backtested on 3779 games. Opp context helps points and threes
        # (real bias improvement). Doesn't help rebounds/assists where
        # individual player factors dominate over team-level matchups.
        if market == "points":
            return self.pts_adj
        elif market == "threes":
            return self.threes_adj
        return 1.0


def _season_for_date(date_str: str) -> str:
    """Convert game date to NBA season string e.g. '2025-26'."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    if dt.month >= 10:
        return f"{dt.year}-{str(dt.year + 1)[-2:]}"
    else:
        return f"{dt.year - 1}-{str(dt.year)[-2:]}"


def _cap(x):
    return float(np.clip(x, ADJ_CAP_LOW, ADJ_CAP_HIGH))


def get_opponent_context(
    team_id: int,
    as_of_date: str,
    db_path: Path = DB_PATH,
) -> Optional[OpponentContext]:
    """Get opponent's defensive profile from PBP data."""
    season = _season_for_date(as_of_date)

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("""
            SELECT n_games, pace, def_poss, opp_points,
                   opp_arc3_fga, opp_corner3_fga, opp_arc3_pct, opp_corner3_pct,
                   opp_def_reb_pct, opp_off_reb_pct,
                   opp_assists
            FROM pbp_team_defense
            WHERE team_id = ? AND season = ? AND season_type = 'Regular Season'
              AND as_of_date IS NULL
            LIMIT 1
        """, (team_id, season)).fetchone()

    # Try previous season fallback if current season missing
    if row is None and season != "2023-24":
        prev_season = f"{int(season[:4]) - 1}-{str(int(season[:4]))[-2:]}"
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("""
                SELECT n_games, pace, def_poss, opp_points,
                       opp_arc3_fga, opp_corner3_fga, opp_arc3_pct, opp_corner3_pct,
                       opp_def_reb_pct, opp_off_reb_pct,
                       opp_assists
                FROM pbp_team_defense
                WHERE team_id = ? AND season = ? AND season_type = 'Regular Season'
                  AND as_of_date IS NULL
                LIMIT 1
            """, (team_id, prev_season)).fetchone()

    if row is None:
        return None

    (n_games, pace, def_poss, opp_pts,
     arc3_fga, corner3_fga, arc3_pct, corner3_pct,
     def_reb_pct, off_reb_pct, assists) = row

    # Compute derived metrics
    def_rtg = (opp_pts * 100.0 / def_poss) if def_poss else LEAGUE_AVG_DEF_RTG
    
    # Combined three-point allowance: total 3PA per game × 3P% allowed
    # = expected 3PM per game allowed
    total_3pa = (arc3_fga or 0) + (corner3_fga or 0)
    total_3pm = (arc3_fga or 0) * (arc3_pct or 0) + (corner3_fga or 0) * (corner3_pct or 0)
    opp_3pm_pg = total_3pm / n_games if n_games else 13.5
    league_avg_3pm = LEAGUE_AVG_3PA_PG * LEAGUE_AVG_3P_PCT  # ~13.2

    opp_ast_pg = assists / n_games if n_games and assists else LEAGUE_AVG_OPP_AST_PG

    # Compute adjustments (>1 = opponent allows MORE than league average)
    pts_adj_raw = def_rtg / LEAGUE_AVG_DEF_RTG
    threes_adj_raw = opp_3pm_pg / league_avg_3pm if league_avg_3pm > 0 else 1.0
    
    # For rebounds: high opp_off_reb_pct = opponent gets a lot of OREBs against
    # this defense, meaning this defense is bad at securing DREBs. That's good
    # for rebounders on the offensive team. We invert and use opp_def_reb_pct.
    # Lower opp DRB% = more OREB chances for offensive rebounders.
    # For simplicity (and since rebounds are mostly defensive), use opp_def_reb_pct
    # directly: high value means opponent secures their own DREBs well, fewer
    # rebounds available for this player to grab on offense.
    # Honestly: the best adjustment here is opp eFG% — bad shooters give up more
    # rebound chances. We don't have clean opp eFG% so we use total reb proxy.
    if off_reb_pct and def_reb_pct:
        # opp_off_reb_pct = % of available OREBs the opponent gets
        # opp_def_reb_pct = % of available DREBs the opponent gets
        # Higher of either = opponent is rebounding-strong, fewer rebs for our player
        opp_reb_strength = (off_reb_pct + def_reb_pct) / 2
        league_avg_reb_strength = (1 - 0.716) / 2 + 0.716 / 2  # ~0.5
        reb_adj_raw = league_avg_reb_strength / opp_reb_strength
    else:
        reb_adj_raw = 1.0
    
    ast_adj_raw = opp_ast_pg / LEAGUE_AVG_OPP_AST_PG

    return OpponentContext(
        team_id=int(team_id),
        season=season,
        n_games=int(n_games),
        pace=float(pace) if pace else LEAGUE_AVG_PACE,
        def_rtg=float(def_rtg),
        opp_3pa_pg=float(total_3pa / n_games) if n_games else LEAGUE_AVG_3PA_PG,
        opp_3p_pct=float(arc3_pct) if arc3_pct else LEAGUE_AVG_3P_PCT,
        opp_drb_pct=float(def_reb_pct) if def_reb_pct else 0.716,
        opp_off_reb_pct=float(off_reb_pct) if off_reb_pct else 0.284,
        opp_ast_pg=float(opp_ast_pg),
        pts_adj=_cap(pts_adj_raw),
        threes_adj=_cap(threes_adj_raw),
        reb_adj=_cap(reb_adj_raw),
        ast_adj=_cap(ast_adj_raw),
    )


def get_opponent_for_player_game(
    player_id: int,
    game_id: str,
    db_path: Path = DB_PATH,
) -> Optional[int]:
    """Find opponent team_id for a player in a given game."""
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("""
            SELECT pb.team_id, g.home_team_id, g.away_team_id
            FROM player_box pb JOIN games g ON pb.game_id = g.game_id
            WHERE pb.player_id = ? AND pb.game_id = ?
        """, (player_id, game_id)).fetchone()
        if row is None:
            return None
        player_team_id, home_id, away_id = row
        return int(away_id if player_team_id == home_id else home_id)
