"""Generic stat projection with market-aware distributions and conditional expectation."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy.stats import nbinom, norm, poisson

from .config import DB_PATH
from .minutes_model import (
    MIN_PLAY_THRESHOLD,
    project_minutes,
    season_for_date,
    season_start_date,
)
from .opponent_context import get_opponent_context, get_opponent_for_player_game

MARKETS = {
    "points":   ("points",     0.50, "normal"),
    "rebounds": ("rebounds",   0.16, "normal"),
    "assists":  ("assists",    0.10, "normal"),
    "threes":   ("fg3m",       0.06, "nbinom"),
    "steals":   ("steals",     0.03, "nbinom"),
    "blocks":   ("blocks",     0.025, "nbinom"),
}

Market = Literal["points", "rebounds", "assists", "threes", "steals", "blocks"]
RECENT_WINDOW_GAMES = 10
MIN_GAMES_FOR_FULL_TRUST = 25

@dataclass
class StatProjection:
    market: str
    expected: float
    std: float
    p_play: float
    n_games: int
    distribution: str
    dist_params: dict
    minutes_proj: dict
    rate_debug: dict

    def p_over(self, line: float) -> float:
        """P(stat > line AND player plays).

        Props void on DNP — the bet neither wins nor loses.
        So the true win probability for an 'over' bet is:
            P(plays) * P(stat > line | plays)

        self.expected is already a *conditional* projection (given the player
        plays), so the distribution parameters reflect the conditional world.
        We just need to scale the resulting probability by p_play.
        """
        if self.p_play <= 0:
            return 0.0
        if self.distribution == "normal":
            std = max(self.std, 0.1)
            cond_p = float(1.0 - norm.cdf((line - self.expected) / std))
        elif self.distribution == "nbinom":
            n = self.dist_params.get("n", 1.0)
            p = self.dist_params.get("p", 0.5)
            k = np.floor(line)
            cond_p = float(1.0 - nbinom.cdf(k, n, p))
        elif self.distribution == "poisson":
            k = np.floor(line)
            cond_p = float(1.0 - poisson.cdf(k, self.expected))
        else:
            cond_p = 0.5
        return self.p_play * cond_p

    def p_under(self, line: float) -> float:
        """P(stat <= line AND player plays).

        Symmetric to p_over. Note p_over + p_under = p_play (not 1.0),
        because the void outcome (DNP) is the remaining probability.
        """
        if self.p_play <= 0:
            return 0.0
        if self.distribution == "normal":
            std = max(self.std, 0.1)
            cond_p = float(norm.cdf((line - self.expected) / std))
        elif self.distribution == "nbinom":
            n = self.dist_params.get("n", 1.0)
            p = self.dist_params.get("p", 0.5)
            k = np.floor(line)
            cond_p = float(nbinom.cdf(k, n, p))
        elif self.distribution == "poisson":
            k = np.floor(line)
            cond_p = float(poisson.cdf(k, self.expected))
        else:
            cond_p = 0.5
        return self.p_play * cond_p

def _normal_dist_params(mean: float, std: float) -> dict:
    return {"mean": mean, "std": std}

def _nbinom_dist_params(mean: float, var: float) -> dict:
    if var <= mean:
        var = mean * 1.05
    p = mean / var
    n = (mean ** 2) / (var - mean)
    return {"n": n, "p": p, "implied_mean": mean, "implied_var": var}


def _get_player_team_from_schedule(player_id: int, as_of_date: str, conn: sqlite3.Connection) -> Optional[int]:
    """Look up a player's team for today's games using the schedule table.
    
    Falls back to most recent player_box entry if schedule lookup fails.
    This is important for projecting today's games before player_box is populated.
    """
    # First try: find the player on a team scheduled to play today
    row = conn.execute("""
        SELECT t.team_id FROM teams t
        JOIN schedule s ON (s.home_team_id = t.team_id OR s.away_team_id = t.team_id)
        WHERE s.game_date = ?
          AND t.team_id = (
              SELECT pb.team_id FROM player_box pb
              JOIN games g ON pb.game_id = g.game_id
              WHERE pb.player_id = ?
              ORDER BY g.game_date DESC LIMIT 1
          )
        LIMIT 1
    """, (as_of_date, player_id)).fetchone()
    
    if row:
        return int(row[0])
    
    # Fallback: most recent team from player_box
    row = conn.execute("""
        SELECT pb.team_id FROM player_box pb
        JOIN games g ON pb.game_id = g.game_id
        WHERE pb.player_id = ?
        ORDER BY g.game_date DESC LIMIT 1
    """, (player_id,)).fetchone()
    
    return int(row[0]) if row else None


def _get_opponent_team_from_schedule(player_team_id: int, as_of_date: str, conn: sqlite3.Connection) -> Optional[int]:
    """Find the opponent team for a given team on a given date via the schedule."""
    row = conn.execute("""
        SELECT 
            CASE WHEN home_team_id = ? THEN away_team_id ELSE home_team_id END AS opp_id
        FROM schedule
        WHERE game_date = ? AND (home_team_id = ? OR away_team_id = ?)
        LIMIT 1
    """, (player_team_id, as_of_date, player_team_id, player_team_id)).fetchone()
    
    return int(row[0]) if row else None


def project_stat(
    player_id: int,
    market: Market,
    as_of_date: str = None,
    game_id: str = None,
    opponent_team_id: int = None,  # explicit override, e.g. from daily_picks
    db_path: Path = DB_PATH
) -> StatProjection:
    if as_of_date is None:
        as_of_date = datetime.now().strftime("%Y-%m-%d")
        
    minutes_proj = project_minutes(player_id, as_of_date, db_path)
    if minutes_proj.expected <= 0.1 or minutes_proj.p_play == 0:
        return StatProjection(market, 0.0, 0.0, 0.0, 0, "normal", {}, {}, {})

    db_col, prior_rate, dist_type = MARKETS[market]

    conn = sqlite3.connect(db_path)
    dt = datetime.fromisoformat(as_of_date.replace("Z", "+00:00")).replace(tzinfo=None)
    season = season_for_date(dt)
    s_start = season_start_date(season)

    df_season = pd.read_sql(f"""
        SELECT pb.minutes, pb.{db_col} as stat 
        FROM player_box pb JOIN games g ON pb.game_id = g.game_id
        WHERE pb.player_id = ? AND g.game_date >= ? AND g.game_date < ? AND pb.minutes > 0
        ORDER BY g.game_date DESC
    """, conn, params=(player_id, s_start, as_of_date))

    df_career = pd.read_sql(f"""
        SELECT pb.minutes, pb.{db_col} as stat 
        FROM player_box pb JOIN games g ON pb.game_id = g.game_id
        WHERE pb.player_id = ? AND g.game_date < ? AND pb.minutes > 0
        ORDER BY g.game_date DESC LIMIT 82
    """, conn, params=(player_id, s_start))

    df_season["rate"] = df_season["stat"] / df_season["minutes"]
    df_career["rate"] = df_career["stat"] / df_career["minutes"]

    n_season = len(df_season)
    n_recent = min(n_season, RECENT_WINDOW_GAMES)
    n_career_used = len(df_career)
    
    season_rate = df_season["rate"].mean() if n_season > 0 else prior_rate
    recent_rate = df_season["rate"].head(n_recent).mean() if n_recent > 0 else season_rate
    career_rate = df_career["rate"].mean() if n_career_used > 0 else prior_rate

    w_season = min(n_season / MIN_GAMES_FOR_FULL_TRUST, 1.0) * 0.5
    w_recent = min(n_recent / RECENT_WINDOW_GAMES, 1.0) * 0.3
    w_career = min(n_career_used / 82, 1.0) * 0.15
    w_prior = max(0, 1.0 - (w_season + w_recent + w_career))
    
    expected_rate = (w_season * season_rate + w_recent * recent_rate + 
                     w_career * career_rate + w_prior * prior_rate)

    # --- Opponent Adjustment ---
    # Priority: explicit opponent_team_id > game_id lookup > schedule-based lookup
    opp_adj = 1.0
    resolved_opp_id = opponent_team_id

    if resolved_opp_id is None and game_id:
        try:
            resolved_opp_id = get_opponent_for_player_game(player_id, game_id, db_path)
        except Exception:
            pass

    if resolved_opp_id is None:
        # Auto-resolve from today's schedule
        try:
            player_team_id = _get_player_team_from_schedule(player_id, as_of_date, conn)
            if player_team_id:
                resolved_opp_id = _get_opponent_team_from_schedule(player_team_id, as_of_date, conn)
        except Exception:
            pass

    if resolved_opp_id is not None:
        try:
            opp_ctx = get_opponent_context(resolved_opp_id, as_of_date, db_path)
            if opp_ctx:
                if market == "points":
                    opp_adj = opp_ctx.pts_adj
                elif market == "threes":
                    opp_adj = opp_ctx.threes_adj
                elif market == "rebounds":
                    opp_adj = opp_ctx.reb_adj
                elif market == "assists":
                    opp_adj = opp_ctx.ast_adj
        except Exception:
            pass

    expected_rate *= opp_adj
    conn.close()

    # === CONDITIONAL EXPECTATION ===
    # Sportsbook props void if the player DNPs. Project stats assuming they play.
    conditional_minutes = minutes_proj.expected / minutes_proj.p_play if minutes_proj.p_play > 0 else 0
    
    expected = expected_rate * conditional_minutes

    all_history = pd.concat([df_season.head(30), df_career.head(20)])
    observed_mean = all_history["stat"].mean() if len(all_history) > 0 else expected
    observed_std = all_history["stat"].std(ddof=1) if len(all_history) > 1 else expected * 0.5
    observed_var = observed_std ** 2

    if dist_type == "normal":
        std_estimate = max(observed_std, 0.5) * 1.05
        params = _normal_dist_params(expected, std_estimate)
        std = params["std"]
    elif dist_type == "nbinom":
        dispersion = observed_var / observed_mean if observed_mean > 0 else 1.5
        target_var = expected * max(dispersion, 1.05)
        params = _nbinom_dist_params(expected, target_var)
        std = float(np.sqrt(params["implied_var"]))
    else:
        params = {"mean": expected, "std": expected * 0.5}
        std = params["std"]

    return StatProjection(
        market=market, expected=round(expected, 2), std=round(std, 2),
        p_play=minutes_proj.p_play, n_games=n_season + n_career_used,
        distribution=dist_type,
        dist_params={k: round(v, 4) if isinstance(v, float) else v for k, v in params.items()},
        minutes_proj={"expected": round(conditional_minutes, 2), "std": round(minutes_proj.std, 2)},
        rate_debug={
            "season_rate": round(season_rate, 4), "recent_rate": round(recent_rate, 4),
            "career_rate": round(career_rate, 4), "opp_adj": round(opp_adj, 3),
            "opp_team_id": resolved_opp_id,
        }
    )
