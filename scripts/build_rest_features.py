"""Build per-team-game rest and travel features into the team_rest table.

For each team-game we walk that team's games in chronological order and compute,
using only prior games (no leakage):
  - days of rest since the previous game
  - back-to-back / 3-in-4 / 4-in-6 schedule-density flags
  - miles traveled from the previous game's arena to this game's arena
  - timezone change between those two arenas

The geometry comes from src.nba_pipeline.travel (hardcoded arena coordinates).
Results are stored in the team_rest table for fast lookup, mirroring the pattern
in scripts/build_team_features.py.

Run after backfill, or any time new games have landed:
    python -m scripts.build_rest_features
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from src.nba_pipeline.config import DB_PATH, setup_logging
from src.nba_pipeline.travel import compute_rest_features

logger = setup_logging("build_rest_features")


def main() -> int:
    conn = sqlite3.connect(DB_PATH)

    # One row per (game, team). The "host" of a game is always the home team's
    # city, which is the destination a team travels to for that game.
    games = pd.read_sql("""
        SELECT game_id, game_date, home_team_id, away_team_id,
               home_team_abbr, away_team_abbr
        FROM games
        WHERE game_date IS NOT NULL
        ORDER BY game_date
    """, conn)

    if games.empty:
        logger.warning("No games found. Run the backfill first.")
        conn.close()
        return 1

    # The host of a game (the arena city travelled to) is always the home team's
    # abbr, for both the home and away team's rows.
    games["host_abbr"] = games["home_team_abbr"]

    # Reshape to long form: each game contributes a home-team row and an
    # away-team row.
    home = games.rename(columns={"home_team_id": "team_id", "home_team_abbr": "team_abbr"})
    away = games.rename(columns={"away_team_id": "team_id", "away_team_abbr": "team_abbr"})
    cols = ["game_id", "game_date", "team_id", "team_abbr", "host_abbr"]
    long = pd.concat([home[cols], away[cols]], ignore_index=True)
    long["game_dt"] = pd.to_datetime(long["game_date"])
    long = long.sort_values(["team_id", "game_dt"]).reset_index(drop=True)

    logger.info(f"Computing rest/travel for {len(long)} team-game rows...")

    rows = []
    for team_id, g in tqdm(long.groupby("team_id"), desc="teams"):
        # prior holds (game_date, host_abbr) for this team's earlier games,
        # most-recent-first — exactly what compute_rest_features expects.
        prior: list[tuple[datetime, str]] = []
        for _, r in g.iterrows():
            # The destination is this game's host city, so pass host_abbr as the
            # current location (not the team's own home city).
            feats = compute_rest_features(r["host_abbr"], r["game_dt"], prior)
            rows.append({
                "game_id": r["game_id"],
                "team_id": int(team_id),
                **feats,
            })
            prior.insert(0, (r["game_dt"], r["host_abbr"]))

    logger.info(f"Inserting {len(rows)} rows into team_rest...")
    conn.execute("DELETE FROM team_rest")
    conn.executemany("""
        INSERT INTO team_rest
        (game_id, team_id, days_rest, is_back_to_back, is_3in4, is_4in6,
         travel_miles, timezone_change)
        VALUES (:game_id, :team_id, :days_rest, :is_back_to_back, :is_3in4,
                :is_4in6, :travel_miles, :timezone_change)
    """, rows)
    conn.commit()

    n = conn.execute("SELECT COUNT(*) FROM team_rest").fetchone()[0]
    logger.info(f"team_rest now has {n} rows")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
