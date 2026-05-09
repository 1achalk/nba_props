"""Minutes projection model with Cascading Positional Spillover."""
from __future__ import annotations
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from .config import DB_PATH

RECENT_WINDOW_GAMES = 10
MIN_GAMES_FOR_FULL_TRUST = 25
MIN_PLAY_THRESHOLD = 5
DEFAULT_PRIOR_MEAN = 18.0
DEFAULT_PRIOR_STD = 8.0

@dataclass
class MinutesProjection:
    expected: float
    std: float
    p_play: float
    n_season: int
    n_recent: int
    n_career: int
    debug: dict

def season_for_date(as_of: datetime) -> str:
    if as_of.month >= 10: return f"{as_of.year}-{str(as_of.year + 1)[2:]}"
    return f"{as_of.year - 1}-{str(as_of.year)[2:]}"

def season_start_date(season: str) -> str:
    return f"{season.split('-')[0]}-10-01"

def _shrinkage(n: int, target: int) -> float:
    return min(n / target, 1.0)

def _get_player_baseline(player_id: int, as_of_date: str, conn: sqlite3.Connection) -> dict:
    dt = datetime.fromisoformat(as_of_date.replace("Z", "+00:00")).replace(tzinfo=None)
    season = season_for_date(dt)
    s_start = season_start_date(season)

    df_season = pd.read_sql("""
        SELECT pb.minutes FROM player_box pb JOIN games g ON pb.game_id = g.game_id
        WHERE pb.player_id = ? AND g.game_date >= ? AND g.game_date < ?
        ORDER BY g.game_date DESC
    """, conn, params=(player_id, s_start, as_of_date))

    df_career = pd.read_sql("""
        SELECT pb.minutes FROM player_box pb JOIN games g ON pb.game_id = g.game_id
        WHERE pb.player_id = ? AND g.game_date < ?
        ORDER BY g.game_date DESC LIMIT 82
    """, conn, params=(player_id, s_start))

    n_season = len(df_season)
    n_recent = min(n_season, RECENT_WINDOW_GAMES)
    n_total = n_season + len(df_career)

    if n_total == 0:
        return {"mean": DEFAULT_PRIOR_MEAN, "std": DEFAULT_PRIOR_STD, "n_season": 0, "n_recent": 0, "n_career": 0, "debug": {}}

    season_mean = df_season["minutes"].mean() if n_season > 0 else 0
    season_std = df_season["minutes"].std(ddof=1) if n_season > 1 else DEFAULT_PRIOR_STD
    recent_mean = df_season["minutes"].head(n_recent).mean() if n_recent > 0 else season_mean
    recent_std = df_season["minutes"].head(n_recent).std(ddof=1) if n_recent > 1 else season_std
    career_mean = df_career["minutes"].mean() if len(df_career) > 0 else DEFAULT_PRIOR_MEAN
    career_std = df_career["minutes"].std(ddof=1) if len(df_career) > 1 else DEFAULT_PRIOR_STD

    w_season = _shrinkage(n_season, MIN_GAMES_FOR_FULL_TRUST) * 0.5
    w_recent = _shrinkage(n_recent, RECENT_WINDOW_GAMES) * 0.3
    w_career = _shrinkage(len(df_career), 82) * 0.15
    w_prior = max(0, 1.0 - (w_season + w_recent + w_career))

    expected = (w_season * season_mean + w_recent * recent_mean + w_career * career_mean + w_prior * DEFAULT_PRIOR_MEAN)
    blended_std = np.sqrt(w_season * season_std**2 + w_recent * recent_std**2 + w_career * career_std**2 + w_prior * DEFAULT_PRIOR_STD**2)

    return {"mean": expected, "std": blended_std, "n_season": n_season, "n_recent": n_recent, "n_career": len(df_career), "debug": {}}

def _map_position(pos_str: str) -> str:
    if not pos_str: return "F"
    pos = pos_str.upper()
    return "C" if "C" in pos else "G" if "G" in pos else "F"

def _injury_p_play(status: str) -> float:
    """Convert normalized injury status to probability of playing.
    
    For healthy players (not on injury report), p_play = 1.0.
    This is intentional — conditional expectation for props assumes the player plays.
    We only discount for players explicitly listed on the injury report.
    """
    return {
        "OUT":          0.0,
        "DOUBTFUL":     0.1,
        "QUESTIONABLE": 0.5,
        "PROBABLE":     0.85,
        "HEALTHY":      1.0,   # On report but expected to play
    }.get(status, 1.0)          # Not on report at all → assume playing

def get_team_minutes_projection(team_id: int, as_of_date: str, db_path: Path = DB_PATH) -> dict[int, MinutesProjection]:
    conn = sqlite3.connect(db_path)
    dt = datetime.fromisoformat(as_of_date.replace("Z", "+00:00")).replace(tzinfo=None)
    s_start = season_start_date(season_for_date(dt))
    
    active_players = pd.read_sql("""
        SELECT DISTINCT pb.player_id, p.position FROM player_box pb 
        JOIN games g ON pb.game_id = g.game_id LEFT JOIN players p ON pb.player_id = p.player_id
        WHERE pb.team_id = ? AND g.game_date >= ? AND g.game_date < ?
    """, conn, params=(team_id, s_start, as_of_date))

    injuries = pd.read_sql("SELECT player_id, status_normalized FROM player_injuries", conn).set_index("player_id")["status_normalized"].to_dict()

    team_projections = {}
    target_bucket_mins = {"G": 0.0, "F": 0.0, "C": 0.0}
    player_buckets = {}

    # 1. Establish Baselines & Targets
    for _, row in active_players.iterrows():
        pid = int(row["player_id"])
        bucket = _map_position(row["position"])
        player_buckets[pid] = bucket
        
        baseline = _get_player_baseline(pid, as_of_date, conn)
        target_bucket_mins[bucket] += baseline["mean"]
        
        # p_play is driven purely by injury status.
        # If a player is not on the injury report, they get p_play=1.0.
        # This is correct for conditional expectation in prop betting:
        # props void on DNPs, so we project stats assuming the player plays.
        status = injuries.get(pid, "NOT_LISTED")
        p_play = _injury_p_play(status)

        if status in ("OUT", "DOUBTFUL"):
            baseline["mean"] = 0.0

        baseline["p_play"] = p_play
        team_projections[pid] = baseline
        team_projections[pid]["expected"] = baseline["mean"]

    # Normalize targets to 240
    total_target = sum(target_bucket_mins.values())
    if total_target > 0:
        target_bucket_mins = {b: (v / total_target) * 240.0 for b, v in target_bucket_mins.items()}

    # Calculate missing minutes from injured/out players
    active_bucket_mins = {"G": 0.0, "F": 0.0, "C": 0.0}
    for pid, p in team_projections.items():
        active_bucket_mins[player_buckets[pid]] += p["expected"]
    
    missing_mins = {b: max(0, target_bucket_mins[b] - active_bucket_mins[b]) for b in ["G", "F", "C"]}
    active_pids = sorted(
        [pid for pid, p in team_projections.items() if p["p_play"] > 0],
        key=lambda x: team_projections[x]["debug"].get("mean", team_projections[x]["mean"]),
        reverse=True
    )

    # 2. Cascading Spillover Distribution
    for b in ["G", "F", "C", "SPILLOVER"]:
        pool = sum(missing_mins.values()) if b == "SPILLOVER" else missing_mins.get(b, 0)
        target_pids = active_pids if b == "SPILLOVER" else [p for p in active_pids if player_buckets[p] == b]
        
        if pool <= 0: continue

        for _ in range(3):
            if pool <= 0.5: break
            
            eligible = []
            for pid in target_pids:
                p = team_projections[pid]
                base = p["debug"].get("mean", p["mean"])
                cap = 40.0 if base >= 25 else (34.0 if base >= 15 else 22.0)
                cap = min(cap, base + 14.0)
                
                if p["expected"] < cap:
                    eligible.append((pid, cap - p["expected"], max(base, 1.0)))
            
            if not eligible: break
                
            total_weight = sum(e[2] for e in eligible)
            for pid, room, weight in eligible:
                allocation = min((weight / total_weight) * pool, room)
                team_projections[pid]["expected"] += allocation
                pool -= allocation
                
        if b != "SPILLOVER":
            missing_mins[b] = pool

    final_projections = {
        pid: MinutesProjection(
            expected=round(p["expected"], 2),
            std=round(p["std"], 2),
            p_play=round(p["p_play"], 3),
            n_season=p["n_season"],
            n_recent=p["n_recent"],
            n_career=p["n_career"],
            debug={**p.get("debug", {}), "bucket": player_buckets[pid], "raw_mean": round(p["mean"], 2)}
        ) for pid, p in team_projections.items()
    }
    conn.close()
    return final_projections

def project_minutes(player_id: int, as_of_date: str = None, db_path: Path = DB_PATH) -> MinutesProjection:
    if as_of_date is None: as_of_date = datetime.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect(db_path)
    team_query = conn.execute("""
        SELECT pb.team_id FROM player_box pb JOIN games g ON pb.game_id = g.game_id
        WHERE pb.player_id = ? ORDER BY g.game_date DESC LIMIT 1
    """, (player_id,)).fetchone()
    conn.close()
    if not team_query: return MinutesProjection(18.0, 8.0, 0.5, 0, 0, 0, {})
    return get_team_minutes_projection(team_query[0], as_of_date, db_path).get(
        player_id, MinutesProjection(0.0, 0.0, 0.0, 0, 0, 0, {})
    )
