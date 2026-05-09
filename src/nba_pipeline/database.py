"""SQLite database: schema, connection, helpers."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .config import DB_PATH

SCHEMA = """
-- Games: one row per NBA game
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    season TEXT NOT NULL,
    season_type TEXT NOT NULL,
    game_date TEXT NOT NULL,
    home_team_id INTEGER NOT NULL,
    away_team_id INTEGER NOT NULL,
    home_team_abbr TEXT,
    away_team_abbr TEXT,
    home_score INTEGER,
    away_score INTEGER,
    status TEXT,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date);
CREATE INDEX IF NOT EXISTS idx_games_season ON games(season);

-- Teams reference table
CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY,
    abbreviation TEXT NOT NULL,
    full_name TEXT,
    city TEXT,
    arena_lat REAL,
    arena_lon REAL
);

-- Players reference table
CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER PRIMARY KEY,
    full_name TEXT NOT NULL,
    position TEXT,
    height_inches INTEGER,
    weight_lbs INTEGER,
    birthdate TEXT,
    is_active INTEGER DEFAULT 1
);

-- Player box score: one row per player per game
CREATE TABLE IF NOT EXISTS player_box (
    game_id TEXT NOT NULL,
    player_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    minutes REAL,
    points INTEGER,
    rebounds INTEGER,
    assists INTEGER,
    steals INTEGER,
    blocks INTEGER,
    turnovers INTEGER,
    fgm INTEGER, fga INTEGER,
    fg3m INTEGER, fg3a INTEGER,
    ftm INTEGER, fta INTEGER,
    plus_minus INTEGER,
    started INTEGER,
    PRIMARY KEY (game_id, player_id),
    FOREIGN KEY (game_id) REFERENCES games(game_id),
    FOREIGN KEY (player_id) REFERENCES players(player_id)
);
CREATE INDEX IF NOT EXISTS idx_pbox_player ON player_box(player_id);
CREATE INDEX IF NOT EXISTS idx_pbox_game ON player_box(game_id);

-- Player advanced stats per game
CREATE TABLE IF NOT EXISTS player_advanced (
    game_id TEXT NOT NULL,
    player_id INTEGER NOT NULL,
    usage_pct REAL,
    true_shooting_pct REAL,
    effective_fg_pct REAL,
    offensive_rating REAL,
    defensive_rating REAL,
    pace REAL,
    PRIMARY KEY (game_id, player_id)
);

-- Team box per game
CREATE TABLE IF NOT EXISTS team_box (
    game_id TEXT NOT NULL,
    team_id INTEGER NOT NULL,
    is_home INTEGER NOT NULL,
    points INTEGER,
    pace REAL,
    offensive_rating REAL,
    defensive_rating REAL,
    efg_pct REAL,
    tov_pct REAL,
    orb_pct REAL,
    ft_rate REAL,
    PRIMARY KEY (game_id, team_id)
);

-- Shots: one row per shot attempt
CREATE TABLE IF NOT EXISTS shots (
    game_id TEXT NOT NULL,
    player_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    period INTEGER,
    seconds_remaining INTEGER,
    shot_type TEXT,
    shot_zone TEXT,
    shot_distance INTEGER,
    loc_x INTEGER,
    loc_y INTEGER,
    made INTEGER NOT NULL,
    shot_idx INTEGER NOT NULL,
    PRIMARY KEY (game_id, player_id, shot_idx)
);
CREATE INDEX IF NOT EXISTS idx_shots_player ON shots(player_id);

-- Schedule (forward-looking, for travel/rest features)
CREATE TABLE IF NOT EXISTS schedule (
    game_id TEXT PRIMARY KEY,
    game_date TEXT NOT NULL,
    home_team_id INTEGER NOT NULL,
    away_team_id INTEGER NOT NULL,
    tipoff_utc TEXT
);
CREATE INDEX IF NOT EXISTS idx_schedule_date ON schedule(game_date);

-- Travel/rest features per team-game
CREATE TABLE IF NOT EXISTS team_rest (
    game_id TEXT NOT NULL,
    team_id INTEGER NOT NULL,
    days_rest INTEGER,
    is_back_to_back INTEGER,
    is_3in4 INTEGER,
    is_4in6 INTEGER,
    travel_miles REAL,
    timezone_change INTEGER,
    PRIMARY KEY (game_id, team_id)
);

-- Injury reports
CREATE TABLE IF NOT EXISTS injuries (
    report_date TEXT NOT NULL,
    player_id INTEGER NOT NULL,
    team_id INTEGER,
    status TEXT,
    reason TEXT,
    PRIMARY KEY (report_date, player_id)
);

-- Player injuries (current, refreshed from CBS)
CREATE TABLE IF NOT EXISTS player_injuries (
    player_id INTEGER PRIMARY KEY,
    player_name TEXT,
    team_abbr TEXT,
    position TEXT,
    injury TEXT,
    status TEXT,
    status_normalized TEXT,
    update_date TEXT,
    fetched_at TEXT,
    source TEXT DEFAULT 'cbssports'
);

-- Player prop odds
CREATE TABLE IF NOT EXISTS prop_odds (
    snapshot_time TEXT NOT NULL,
    game_id TEXT,
    player_id INTEGER,
    player_name TEXT,
    market TEXT NOT NULL,
    book TEXT NOT NULL,
    line REAL NOT NULL,
    over_odds INTEGER,
    under_odds INTEGER,
    PRIMARY KEY (snapshot_time, player_name, market, book, line)
);
CREATE INDEX IF NOT EXISTS idx_prop_player ON prop_odds(player_name);
CREATE INDEX IF NOT EXISTS idx_prop_game ON prop_odds(game_id);

-- Picks log: persists all generated picks for performance tracking
CREATE TABLE IF NOT EXISTS picks_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pick_date TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    player_name TEXT NOT NULL,
    player_id INTEGER,
    market TEXT NOT NULL,
    side TEXT NOT NULL,        -- 'over' or 'under'
    line REAL NOT NULL,
    best_book TEXT,
    book_odds INTEGER,
    book_implied REAL,
    model_prob REAL,
    edge REAL,
    model_pred REAL,
    model_std REAL,
    n_games INTEGER,
    opp_team TEXT,
    actual REAL,
    won INTEGER,               -- 1=win, 0=loss, NULL=pending
    pl REAL,                   -- profit/loss in units
    result_checked_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_picks_date ON picks_log(pick_date);
CREATE INDEX IF NOT EXISTS idx_picks_player ON picks_log(player_id);

-- PBP team defense (populated by fetch_pbp_defense.py)
CREATE TABLE IF NOT EXISTS pbp_team_defense (
    team_id INTEGER NOT NULL,
    season TEXT NOT NULL,
    season_type TEXT NOT NULL,
    as_of_date TEXT,
    n_games INTEGER,
    def_poss REAL,
    pace REAL,
    opp_points REAL,
    opp_efg REAL,
    opp_ts_pct REAL,
    opp_at_rim_fga REAL,
    opp_at_rim_pct REAL,
    opp_short_mid_fga REAL,
    opp_short_mid_pct REAL,
    opp_long_mid_fga REAL,
    opp_long_mid_pct REAL,
    opp_arc3_fga REAL,
    opp_arc3_pct REAL,
    opp_corner3_fga REAL,
    opp_corner3_pct REAL,
    opp_def_reb_pct REAL,
    opp_off_reb_pct REAL,
    opp_assists REAL,
    opp_turnovers REAL,
    opp_blocks REAL,
    opp_steals REAL,
    opp_fouls REAL,
    fetched_at TEXT,
    PRIMARY KEY (team_id, season, season_type, as_of_date)
);
"""


def init_db(db_path: Path = DB_PATH) -> None:
    """Create the database and tables if they don't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.commit()


@contextmanager
def get_conn(db_path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    """Context manager for DB connections."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def upsert_many(
    conn: sqlite3.Connection,
    table: str,
    rows: list[dict],
    conflict_cols: list[str],
) -> int:
    """INSERT ... ON CONFLICT UPDATE for a list of dict rows. Returns rows affected."""
    if not rows:
        return 0
    cols = list(rows[0].keys())
    placeholders = ", ".join(["?"] * len(cols))
    col_list = ", ".join(cols)
    update_cols = [c for c in cols if c not in conflict_cols]
    update_clause = ", ".join(f"{c}=excluded.{c}" for c in update_cols)
    conflict_clause = ", ".join(conflict_cols)
    sql = (
        f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
        f"ON CONFLICT ({conflict_clause}) DO UPDATE SET {update_clause}"
    )
    values = [tuple(r[c] for c in cols) for r in rows]
    cur = conn.executemany(sql, values)
    return cur.rowcount
