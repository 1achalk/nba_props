"""Refresh player prop odds.

Only saves props for today's scheduled games — stale completed-game
props are cleared automatically on each run.

Usage:
    python -m scripts.refresh_odds
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime

from src.nba_pipeline.config import DB_PATH, setup_logging
from src.nba_pipeline.database import get_conn, init_db, upsert_many
from src.nba_pipeline.odds_client import SportsGameOddsClient

logger = setup_logging("refresh_odds")


def main() -> int:
    init_db()
    today = datetime.now().strftime("%Y-%m-%d")

    # Clear stale odds — anything older than today's first second
    yesterday_cutoff = today + "T00:00:00"
    with sqlite3.connect(DB_PATH) as conn:
        deleted = conn.execute(
            "DELETE FROM prop_odds WHERE snapshot_time < ?", (yesterday_cutoff,)
        ).rowcount
        conn.commit()
    if deleted:
        logger.info(f"Cleared {deleted} stale prop rows from previous days")

    client = SportsGameOddsClient()
    events = client.fetch_nba_events()
    rows = list(client.extract_player_props(events))

    if not rows:
        logger.warning("No prop rows extracted — check API plan supports player props")
        return 0

    with get_conn() as conn:
        upsert_many(
            conn,
            "prop_odds",
            rows,
            ["snapshot_time", "player_name", "market", "book", "line"],
        )
    logger.info(f"Saved {len(rows)} prop odds snapshots")
    return 0


if __name__ == "__main__":
    sys.exit(main())
