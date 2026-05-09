"""Fetch team defensive profiles from PBP Stats API.

One API call per team per season = ~30 calls per season. Fast and clean.
Stores in pbp_team_defense table.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import datetime, timezone

import requests

from src.nba_pipeline.config import DB_PATH, setup_logging

logger = setup_logging("fetch_pbp_defense")

API_BASE = "https://api.pbpstats.com"
RATE_LIMIT_SEC = 1.5


def fetch_team_opponent(team_id: int, season: str, season_type: str = "Regular Season",
                         to_date: str = None, timeout: int = 120) -> dict:
    params = {
        "Season": season,
        "SeasonType": season_type,
        "Type": "Opponent",
        "TeamId": str(team_id),
    }
    if to_date:
        params["ToDate"] = to_date
    
    url = f"{API_BASE}/get-totals/nba"
    resp = requests.get(url, params=params, timeout=timeout)
    if resp.status_code != 200:
        return None
    data = resp.json()
    rows = data.get("multi_row_table_data", [])
    return rows[0] if rows else None


def parse_opponent_row(row: dict) -> dict:
    """Extract the fields we care about from PBP Stats response."""
    games = row.get("GamesPlayed", 0)
    return {
        "n_games": games,
        "def_poss": row.get("DefPoss"),
        "pace": row.get("Pace"),
        "opp_points": row.get("Points"),  # opponent points scored against this team
        "opp_efg": row.get("EfgPct"),
        "opp_ts_pct": row.get("TsPct"),
        "opp_at_rim_fga": row.get("AtRimFGA"),
        "opp_at_rim_pct": row.get("AtRimAccuracy"),
        "opp_short_mid_fga": row.get("ShortMidRangeFGA"),
        "opp_short_mid_pct": row.get("ShortMidRangeAccuracy"),
        "opp_long_mid_fga": row.get("LongMidRangeFGA"),
        "opp_long_mid_pct": row.get("LongMidRangeAccuracy"),
        "opp_arc3_fga": row.get("Arc3FGA"),
        "opp_arc3_pct": row.get("Arc3Accuracy"),
        "opp_corner3_fga": row.get("Corner3FGA"),
        "opp_corner3_pct": row.get("Corner3Accuracy"),
        "opp_def_reb_pct": row.get("DefFGReboundPct"),
        "opp_off_reb_pct": row.get("OffFGReboundPct"),
        "opp_assists": row.get("Assists"),
        "opp_turnovers": row.get("Turnovers"),
        "opp_blocks": row.get("Blocks"),
        "opp_steals": row.get("Steals"),
        "opp_fouls": row.get("Fouls"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", default="2023-24,2024-25,2025-26")
    parser.add_argument("--season-type", default="Regular Season",
                        choices=["Regular Season", "Playoffs", "All"])
    parser.add_argument("--to-date", default=None,
                        help="Snapshot as-of date (YYYY-MM-DD)")
    args = parser.parse_args()

    seasons = [s.strip() for s in args.seasons.split(",")]
    fetched_at = datetime.now(timezone.utc).isoformat()
    
    conn = sqlite3.connect(DB_PATH)
    teams = conn.execute(
        "SELECT team_id, abbreviation FROM teams ORDER BY abbreviation"
    ).fetchall()
    
    total_calls = len(teams) * len(seasons)
    logger.info(f"Fetching {total_calls} team-seasons "
                f"(~{total_calls * RATE_LIMIT_SEC:.0f}s with rate limit)")
    
    inserted = 0
    failed = 0
    for season in seasons:
        for team_id, abbr in teams:
            row = fetch_team_opponent(team_id, season, args.season_type, args.to_date)
            time.sleep(RATE_LIMIT_SEC)
            
            if row is None:
                logger.warning(f"  {abbr} {season}: no data")
                failed += 1
                continue
            
            parsed = parse_opponent_row(row)
            parsed.update({
                "team_id": team_id,
                "season": season,
                "season_type": args.season_type,
                "as_of_date": args.to_date,
                "fetched_at": fetched_at,
            })
            
            conn.execute("""
                INSERT OR REPLACE INTO pbp_team_defense
                (team_id, season, season_type, as_of_date, n_games, def_poss, pace,
                 opp_points, opp_efg, opp_ts_pct,
                 opp_at_rim_fga, opp_at_rim_pct, opp_short_mid_fga, opp_short_mid_pct,
                 opp_long_mid_fga, opp_long_mid_pct, opp_arc3_fga, opp_arc3_pct,
                 opp_corner3_fga, opp_corner3_pct,
                 opp_def_reb_pct, opp_off_reb_pct,
                 opp_assists, opp_turnovers, opp_blocks, opp_steals, opp_fouls,
                 fetched_at)
                VALUES
                (:team_id, :season, :season_type, :as_of_date, :n_games, :def_poss, :pace,
                 :opp_points, :opp_efg, :opp_ts_pct,
                 :opp_at_rim_fga, :opp_at_rim_pct, :opp_short_mid_fga, :opp_short_mid_pct,
                 :opp_long_mid_fga, :opp_long_mid_pct, :opp_arc3_fga, :opp_arc3_pct,
                 :opp_corner3_fga, :opp_corner3_pct,
                 :opp_def_reb_pct, :opp_off_reb_pct,
                 :opp_assists, :opp_turnovers, :opp_blocks, :opp_steals, :opp_fouls,
                 :fetched_at)
            """, parsed)
            conn.commit()
            inserted += 1
            logger.info(f"  {abbr} {season}: n={parsed['n_games']} pace={parsed['pace']:.1f} opp_pts={parsed['opp_points']:.0f}")
    
    logger.info(f"Done: inserted {inserted}, failed {failed}")
    conn.close()


if __name__ == "__main__":
    sys.exit(main())
