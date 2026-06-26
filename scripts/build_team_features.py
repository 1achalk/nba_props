"""Build team-level rolling features.

For each team-game, compute:
  - Team's rolling offensive efficiency (last 20 games)
  - Team's rolling defensive efficiency (last 20 games)
  - Team's rolling pace (last 20 games)
  - Opponent stats specific to player markets:
      - Rebounds allowed per game
      - Assists allowed per game
      - 3PM allowed per game

Stored in team_rolling table for fast lookup.
"""
from __future__ import annotations

import sqlite3

import pandas as pd

from src.nba_pipeline.config import setup_logging, DB_PATH

logger = setup_logging("build_team_features")

ROLLING_WINDOW = 20

SCHEMA = """
CREATE TABLE IF NOT EXISTS team_rolling (
    game_id TEXT NOT NULL,
    team_id INTEGER NOT NULL,
    game_date TEXT NOT NULL,
    -- Rolling team performance over last N games (excluding current)
    pace REAL,
    points_for REAL,
    points_against REAL,
    -- Stat-specific allowed (what opponent did against this team)
    opp_rebounds REAL,
    opp_assists REAL,
    opp_threes REAL,
    n_games INTEGER,
    PRIMARY KEY (game_id, team_id)
);
CREATE INDEX IF NOT EXISTS idx_team_rolling_team_date ON team_rolling(team_id, game_date);
"""


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)

    # Build a per-team-game frame: for each (team, game), what did they allow?
    # team_box has team-level data; if we don't have it populated, derive from player_box.
    logger.info("Aggregating team-game stats from player_box...")
    df = pd.read_sql("""
        SELECT
            pb.game_id,
            pb.team_id,
            g.game_date,
            g.home_team_id,
            g.away_team_id,
            SUM(pb.points) AS team_pts,
            SUM(pb.rebounds) AS team_reb,
            SUM(pb.assists) AS team_ast,
            SUM(pb.fg3m) AS team_3pm,
            SUM(pb.fga) AS team_fga,
            SUM(pb.fta) AS team_fta,
            SUM(pb.turnovers) AS team_tov,
            SUM(pb.minutes) AS team_min
        FROM player_box pb
        JOIN games g ON pb.game_id = g.game_id
        GROUP BY pb.game_id, pb.team_id
    """, conn)

    logger.info(f"Loaded {len(df)} team-game rows")

    # Pace estimate: 0.5 * (FGA + 0.44*FTA + TOV) is a common formula
    # Apply per team, total possessions ~= average of two teams' estimates
    df["possessions"] = df["team_fga"] + 0.44 * df["team_fta"] + df["team_tov"]

    # For each team-game, find the opponent-team-game row to get points allowed and
    # opponent counting stats
    df_paired = df.merge(
        df[["game_id", "team_id", "team_pts", "team_reb", "team_ast", "team_3pm", "possessions"]]
            .rename(columns={
                "team_id": "opp_id",
                "team_pts": "opp_pts",
                "team_reb": "opp_reb_made",
                "team_ast": "opp_ast_made",
                "team_3pm": "opp_3pm_made",
                "possessions": "opp_poss",
            }),
        on="game_id",
    )
    # Filter so each row pairs a team with its opponent (not itself)
    df_paired = df_paired[df_paired["team_id"] != df_paired["opp_id"]].copy()
    # Pace = avg of two teams' possession estimates per 48 min
    df_paired["game_pace"] = (df_paired["possessions"] + df_paired["opp_poss"]) / 2
    df_paired = df_paired.sort_values(["team_id", "game_date"]).reset_index(drop=True)

    logger.info(f"Computing rolling features (window={ROLLING_WINDOW} games)...")

    # Group by team and compute rolling means over PRIOR games (shift by 1 to exclude current)
    grouped = df_paired.groupby("team_id", group_keys=False)

    df_paired["pace_rolling"] = grouped["game_pace"].transform(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=5).mean()
    )
    df_paired["pts_for_rolling"] = grouped["team_pts"].transform(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=5).mean()
    )
    df_paired["pts_against_rolling"] = grouped["opp_pts"].transform(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=5).mean()
    )
    df_paired["opp_reb_rolling"] = grouped["opp_reb_made"].transform(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=5).mean()
    )
    df_paired["opp_ast_rolling"] = grouped["opp_ast_made"].transform(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=5).mean()
    )
    df_paired["opp_3pm_rolling"] = grouped["opp_3pm_made"].transform(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=5).mean()
    )
    df_paired["n_games_rolling"] = grouped["team_pts"].transform(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).count()
    )

    # Build insert rows
    rows = []
    for _, r in df_paired.iterrows():
        if pd.isna(r["pace_rolling"]):
            continue
        rows.append({
            "game_id": r["game_id"],
            "team_id": int(r["team_id"]),
            "game_date": r["game_date"],
            "pace": float(r["pace_rolling"]),
            "points_for": float(r["pts_for_rolling"]),
            "points_against": float(r["pts_against_rolling"]),
            "opp_rebounds": float(r["opp_reb_rolling"]),
            "opp_assists": float(r["opp_ast_rolling"]),
            "opp_threes": float(r["opp_3pm_rolling"]),
            "n_games": int(r["n_games_rolling"]),
        })

    logger.info(f"Inserting {len(rows)} rolling-feature rows...")
    conn.execute("DELETE FROM team_rolling")
    conn.executemany("""
        INSERT INTO team_rolling
        (game_id, team_id, game_date, pace, points_for, points_against,
         opp_rebounds, opp_assists, opp_threes, n_games)
        VALUES (:game_id, :team_id, :game_date, :pace, :points_for, :points_against,
                :opp_rebounds, :opp_assists, :opp_threes, :n_games)
    """, rows)
    conn.commit()

    n = conn.execute("SELECT COUNT(*) FROM team_rolling").fetchone()[0]
    logger.info(f"team_rolling now has {n} rows")
    conn.close()


if __name__ == "__main__":
    main()
