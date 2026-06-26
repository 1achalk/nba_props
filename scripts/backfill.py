"""Backfill historical NBA data from basketball-reference.com."""
from __future__ import annotations
import argparse
import hashlib
import re
import sys
from io import StringIO

import pandas as pd
from tqdm import tqdm

from src.nba_pipeline.config import setup_logging
from src.nba_pipeline.database import get_conn, init_db, upsert_many
from src.nba_pipeline.nba_client_br import (
    BR_TEAM_MAP, TEAM_ID_BY_ABBR, fetch_box_score, fetch_season_games,
    fetch_teams,
)

logger = setup_logging("backfill")

# Match player anchor: href="/players/x/slug.html">Name
PLAYER_ID_RE = re.compile(r'href="/players/[a-z]/([a-z0-9]+)\.html">([^<]+)<')

# Match a full-game basic box table block. We extract by ID so we ignore
# all the per-quarter and per-half splits.
BR_TO_NBA = {v: k for k, v in BR_TEAM_MAP.items()}


def seed_reference_tables():
    teams = fetch_teams()
    team_rows = [{"team_id":t["id"],"abbreviation":t["abbreviation"],
                  "full_name":t["full_name"],"city":t["city"],
                  "arena_lat":None,"arena_lon":None} for t in teams]
    # Players are added during box-score ingestion with stable slug-based IDs.
    # Skipping the season-totals seed avoids creating phantom IDs.
    with get_conn() as conn:
        upsert_many(conn, "teams", team_rows, ["team_id"])
    logger.info(f"Seeded {len(team_rows)} teams (players added during box ingestion)")


def backfill_season(season, skip_shots=False):
    df = fetch_season_games(season)
    if df.empty:
        logger.warning(f"No games found for {season}")
        return
    game_ids = df["GAME_ID"].unique().tolist()
    logger.info(f"Season {season}: {len(game_ids)} unique games")

    games_to_insert = []
    seen = set()
    for _, row in df.iterrows():
        gid = row["GAME_ID"]
        if gid in seen: continue
        seen.add(gid)
        is_home = "vs." in row["MATCHUP"]
        if is_home:
            home_abbr = row["TEAM_ABBREVIATION"]
            away_abbr = row["MATCHUP"].split("vs.")[1].strip()
        else:
            away_abbr = row["TEAM_ABBREVIATION"]
            home_abbr = row["MATCHUP"].split("@")[1].strip()
        games_to_insert.append({
            "game_id":gid,"season":season,"season_type":"Regular Season",
            "game_date":row["GAME_DATE"],
            "home_team_id":TEAM_ID_BY_ABBR.get(home_abbr,0),
            "away_team_id":TEAM_ID_BY_ABBR.get(away_abbr,0),
            "home_team_abbr":home_abbr,"away_team_abbr":away_abbr,
            "home_score":None,"away_score":None,"status":"Final",
        })

    with get_conn() as conn:
        upsert_many(conn, "games", games_to_insert, ["game_id"])

    # Skip games already in player_box for fast resume
    with get_conn() as conn:
        already_loaded = {r[0] for r in conn.execute(
            "SELECT DISTINCT game_id FROM player_box WHERE game_id IN ({})".format(
                ",".join("?" * len(game_ids))
            ),
            game_ids,
        )}
    todo = [g for g in game_ids if g not in already_loaded]
    skipped = len(game_ids) - len(todo)
    if skipped:
        logger.info(f"Skipping {skipped} games already ingested")
    logger.info(f"Pulling box scores for {len(todo)} games "
                f"(rate-limited; ~{len(todo)*3.5/60:.0f} minutes)")
    for gid in tqdm(todo, desc=f"{season} box scores"):
        try:
            _ingest_box_score_br(gid)
        except Exception as e:
            logger.error(f"Failed game {gid}: {e}")
            continue


def _extract_full_game_box(html: str, br_team_abbr: str) -> pd.DataFrame | None:
    """Pull just the full-game basic box for a single team.

    BR wraps these in HTML comments to defer rendering. We strip those out,
    locate the <table id="box-XXX-game-basic"> block, and parse it.
    """
    # BR puts boxes inside HTML comments to avoid auto-loading; strip them
    cleaned = html.replace("<!--", "").replace("-->", "")
    table_id = f"box-{br_team_abbr}-game-basic"
    # Greedy-but-bounded match for the table block
    pattern = re.compile(
        rf'<table[^>]*id="{re.escape(table_id)}"[^>]*>.*?</table>',
        re.DOTALL,
    )
    m = pattern.search(cleaned)
    if not m:
        return None
    table_html = m.group(0)
    try:
        tables = pd.read_html(StringIO(table_html))
    except ValueError:
        return None
    if not tables:
        return None
    df = tables[0]
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[1] if isinstance(c, tuple) else c for c in df.columns]
    return df


def _ingest_box_score_br(game_id):
    box = fetch_box_score(game_id)
    html = box["traditional_html"]

    # game_id format: YYYYMMDD-AWAY-HOME (NBA-style abbrs)
    parts = game_id.split("-")
    if len(parts) != 3: return
    _, away_abbr, home_abbr = parts

    # Convert to BR's abbreviations for table-id lookup
    br_away = BR_TEAM_MAP.get(away_abbr, away_abbr)
    br_home = BR_TEAM_MAP.get(home_abbr, home_abbr)

    teams_in_order = [(away_abbr, br_away), (home_abbr, br_home)]

    # Player slug map from anchor tags
    name_to_brslug = {}
    for slug, name in PLAYER_ID_RE.findall(html):
        name_clean = re.sub(r"<[^>]+>", "", name).strip()
        if name_clean:
            name_to_brslug[name_clean] = slug

    rows = []
    for team_abbr, br_abbr in teams_in_order:
        team_id = TEAM_ID_BY_ABBR.get(team_abbr, 0)
        df = _extract_full_game_box(html, br_abbr)
        if df is None or df.empty:
            logger.debug(f"{game_id}: no full-game box for {br_abbr}")
            continue

        df = df[df.iloc[:, 0].notna()]
        starter_idx = 0
        for _, r in df.iterrows():
            name = str(r.iloc[0]).strip()
            if name in ("Starters","Reserves","Team Totals","Player"): continue
            if name.startswith("Did Not") or name.startswith("Inactive"): continue
            slug = name_to_brslug.get(name)
            if not slug: continue
            pid = int(hashlib.md5(slug.encode()).hexdigest()[:8], 16)

            mp_str = str(r.get("MP","0"))
            try:
                if ":" in mp_str:
                    m, s = mp_str.split(":")
                    minutes = int(m) + int(s) / 60
                else:
                    minutes = float(mp_str) if mp_str.replace(".","").isdigit() else 0.0
            except (ValueError, AttributeError):
                minutes = 0.0

            rows.append({
                "game_id":game_id,"player_id":pid,"team_id":team_id,
                "minutes":minutes,
                "points":_safe_int(r.get("PTS")),"rebounds":_safe_int(r.get("TRB")),
                "assists":_safe_int(r.get("AST")),"steals":_safe_int(r.get("STL")),
                "blocks":_safe_int(r.get("BLK")),"turnovers":_safe_int(r.get("TOV")),
                "fgm":_safe_int(r.get("FG")),"fga":_safe_int(r.get("FGA")),
                "fg3m":_safe_int(r.get("3P")),"fg3a":_safe_int(r.get("3PA")),
                "ftm":_safe_int(r.get("FT")),"fta":_safe_int(r.get("FTA")),
                "plus_minus":_safe_int(r.get("+/-")),
                "started":1 if starter_idx < 5 else 0,
            })
            starter_idx += 1

    if name_to_brslug:
        player_ref_rows = []
        for name, slug in name_to_brslug.items():
            pid = int(hashlib.md5(slug.encode()).hexdigest()[:8], 16)
            player_ref_rows.append({
                "player_id":pid,"full_name":name,"position":None,
                "height_inches":None,"weight_lbs":None,"birthdate":None,
                "is_active":1,
            })
        with get_conn() as conn:
            upsert_many(conn, "players", player_ref_rows, ["player_id"])

    if rows:
        with get_conn() as conn:
            upsert_many(conn, "player_box", rows, ["game_id","player_id"])


def _safe_int(v):
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return 0
        return int(float(v))
    except (ValueError, TypeError):
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", required=True)
    parser.add_argument("--skip-shots", action="store_true")
    args = parser.parse_args()
    init_db()
    seed_reference_tables()
    for season in args.seasons:
        logger.info(f"=== Backfilling {season} ===")
        backfill_season(season, skip_shots=args.skip_shots)
    logger.info("Backfill complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
