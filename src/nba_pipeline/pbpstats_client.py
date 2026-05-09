"""PBP Stats API client.

Free unauthenticated API. Rate-limit ourselves to be polite (~1 req/sec).
Endpoint: https://api.pbpstats.com/get-game-stats?Type={Player|Lineup|LineupOpponent}&GameId={gid}
"""
from __future__ import annotations

import time
from typing import Optional

import requests

API_BASE = "https://api.pbpstats.com"
DEFAULT_TIMEOUT = 60
RATE_LIMIT_SEC = 1.2  # min seconds between requests

_last_request_time = 0.0


def _rate_limit():
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < RATE_LIMIT_SEC:
        time.sleep(RATE_LIMIT_SEC - elapsed)
    _last_request_time = time.time()


def fetch_game_stats(game_id: str, stat_type: str = "Player",
                     timeout: int = DEFAULT_TIMEOUT) -> Optional[dict]:
    """Fetch stats for a single game. Returns None on error."""
    _rate_limit()
    # Pad game ID to 10 chars (PBP requires this)
    if len(game_id) < 10:
        game_id = game_id.zfill(10)
    url = f"{API_BASE}/get-game-stats?Type={stat_type}&GameId={game_id}"
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
        return None
    except (requests.RequestException, ValueError):
        return None


def aggregate_lineup_opponent_to_team(data: dict) -> dict:
    """Sum lineup-opponent stats (defensive stats) to team level for one game.
    
    LineupOpponent endpoint returns 5-man lineups; what opponents did against
    each lineup. We sum across all lineups for the team to get total opponent
    production against that team.
    """
    result = {"home": {}, "away": {}}
    if "stats" not in data:
        return result

    for side in ("Home", "Away"):
        side_key = side.lower()
        try:
            lineups = data["stats"][side]["FullGame"]
        except (KeyError, TypeError):
            continue
        
        # Sum across all lineups for this team
        agg = {
            "def_poss": 0,
            "opp_at_rim_fga": 0, "opp_at_rim_fgm": 0,
            "opp_short_mid_fga": 0, "opp_short_mid_fgm": 0,
            "opp_long_mid_fga": 0, "opp_long_mid_fgm": 0,
            "opp_arc3_fga": 0, "opp_arc3_fgm": 0,
            "opp_assists": 0,
            "opp_pts_assisted_2s": 0, "opp_pts_assisted_3s": 0,
            "opp_def_rebounds": 0, "opp_off_rebounds": 0,
            "opp_points": 0,
        }
        for lineup in lineups:
            if lineup.get("Name") == "Team":
                continue
            agg["def_poss"] += lineup.get("DefPoss", 0)
            agg["opp_at_rim_fga"] += lineup.get("AtRimFGA", 0)
            agg["opp_at_rim_fgm"] += lineup.get("AtRimFGM", 0)
            agg["opp_short_mid_fga"] += lineup.get("ShortMidRangeFGA", 0)
            agg["opp_short_mid_fgm"] += lineup.get("ShortMidRangeFGM", 0)
            agg["opp_long_mid_fga"] += lineup.get("LongMidRangeFGA", 0)
            agg["opp_long_mid_fgm"] += lineup.get("LongMidRangeFGM", 0)
            agg["opp_arc3_fga"] += lineup.get("Arc3FGA", 0)
            agg["opp_arc3_fgm"] += lineup.get("Arc3FGM", 0)
            agg["opp_assists"] += lineup.get("Assists", 0)
            agg["opp_pts_assisted_2s"] += lineup.get("PtsAssisted2s", 0)
            agg["opp_pts_assisted_3s"] += lineup.get("PtsAssisted3s", 0)
            agg["opp_def_rebounds"] += lineup.get("DefRebounds", 0)
            agg["opp_off_rebounds"] += lineup.get("OffRebounds", 0)
        
        # Compute derived metrics
        if agg["def_poss"] > 0:
            # Total opponent points (rough estimate from shot makes + FT)
            # We don't have FT in this aggregation, so use pts_assisted + non-assisted
            opp_2pm = agg["opp_at_rim_fgm"] + agg["opp_short_mid_fgm"] + agg["opp_long_mid_fgm"]
            opp_3pm = agg["opp_arc3_fgm"]  # Note: may miss corner threes, will check
            agg["opp_points_estimate"] = opp_2pm * 2 + opp_3pm * 3
        
        result[side_key] = agg
    return result
