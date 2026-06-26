"""Daily incremental update from basketball-reference.com.

Pulls yesterday's completed games + today's schedule.
"""
from __future__ import annotations
import sys
from datetime import datetime, timedelta
from functools import lru_cache

import pandas as pd

from src.nba_pipeline.config import setup_logging
from src.nba_pipeline.database import get_conn, init_db, upsert_many
from src.nba_pipeline.nba_client_br import (
    TEAM_ID_BY_ABBR,
    fetch_season_games,
)
from scripts.backfill import _ingest_box_score_br

logger = setup_logging("daily_update")


def get_current_season() -> str:
    today = datetime.now()
    if today.month >= 10:
        start = today.year
    else:
        start = today.year - 1
    return f"{start}-{str(start + 1)[2:]}"


# Cache the schedule fetch within a single script run
@lru_cache(maxsize=1)
def _cached_schedule(season: str) -> pd.DataFrame:
    return fetch_season_games(season)


def update_completed_games(target_date: str | None = None, lookback_days: int = 3) -> None:
    """Pull box scores for games on target_date, OR back-fill the last N days
    of games not yet in player_box."""
    season = get_current_season()
    df = _cached_schedule(season)
    if df.empty:
        logger.info(f"No games for season {season}")
        return
    
    # Build list of dates to check
    if target_date:
        dates_to_check = [target_date]
    else:
        # Look back N days from yesterday
        today = datetime.now()
        dates_to_check = [
            (today - timedelta(days=d)).strftime("%Y-%m-%d")
            for d in range(1, lookback_days + 1)
        ]
    
    logger.info(f"Checking dates: {dates_to_check}")
    
    for d in dates_to_check:
        df_target = df[df["GAME_DATE"] == d]
        if df_target.empty:
            logger.info(f"  No games on {d}")
            continue
        game_ids = df_target["GAME_ID"].unique().tolist()
        logger.info(f"  Ingesting up to {len(game_ids)} games from {d}")
        for gid in game_ids:
            try:
                _ingest_box_score_br(gid)
                logger.info(f"    ingested {gid}")
            except Exception as e:
                logger.error(f"    failed {gid}: {e}")


def update_today_schedule() -> None:
    season = get_current_season()
    today = datetime.now().strftime("%Y-%m-%d")
    df = _cached_schedule(season)
    if df.empty:
        logger.info("No schedule data")
        return
    df_today = df[df["GAME_DATE"] == today]
    if df_today.empty:
        logger.info("No games scheduled today")
        return
    rows = []
    seen = set()
    for _, r in df_today.iterrows():
        gid = r["GAME_ID"]
        if gid in seen:
            continue
        seen.add(gid)
        is_home = "vs." in r["MATCHUP"]
        if is_home:
            home_abbr = r["TEAM_ABBREVIATION"]
            away_abbr = r["MATCHUP"].split("vs.")[1].strip()
        else:
            away_abbr = r["TEAM_ABBREVIATION"]
            home_abbr = r["MATCHUP"].split("@")[1].strip()
        rows.append({
            "game_id": gid,
            "game_date": today,
            "home_team_id": TEAM_ID_BY_ABBR.get(home_abbr, 0),
            "away_team_id": TEAM_ID_BY_ABBR.get(away_abbr, 0),
            "tipoff_utc": None,
        })
    if rows:
        with get_conn() as conn:
            upsert_many(conn, "schedule", rows, ["game_id"])
        logger.info(f"Scheduled {len(rows)} games for today")


def main() -> int:
    init_db()
    update_completed_games()
    update_today_schedule()
    logger.info("Daily update complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
