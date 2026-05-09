"""Fetch latest NBA injuries and store in player_injuries table.

Run before generating picks. Idempotent — overwrites existing entries.
"""
from __future__ import annotations

import sqlite3
import sys
import unicodedata
from datetime import datetime, timezone

from src.nba_pipeline.config import DB_PATH, setup_logging
from src.nba_pipeline.injury_client import fetch_cbs_injuries

logger = setup_logging("refresh_injuries")


def normalize_player_name(name: str) -> str:
    """Normalize a name for fuzzy matching.
    
    - Lowercases
    - Strips accents (Nikola Jović -> nikola jovic)
    - Removes suffixes (Jr., Sr., II, III, IV)
    - Removes punctuation
    """
    # Decompose accented characters and strip the accent marks
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    
    # Lowercase and strip punctuation
    cleaned = ascii_name.lower().replace("-", "").replace(".", "").replace("'", "").replace("'", "")
    
    # Remove generational suffixes
    tokens = cleaned.split()
    suffix_tokens = {"jr", "sr", "ii", "iii", "iv", "v"}
    tokens = [t for t in tokens if t not in suffix_tokens]
    
    return "".join(tokens)


def find_player_id(prop_name: str, conn: sqlite3.Connection) -> int | None:
    target = normalize_player_name(prop_name)
    rows = conn.execute("""
        SELECT p.player_id, p.full_name FROM players p
        JOIN player_box pb ON p.player_id = pb.player_id
        GROUP BY p.player_id HAVING COUNT(pb.game_id) > 0
    """).fetchall()

    # Pass 1: exact normalized match
    for pid, name in rows:
        if normalize_player_name(name) == target:
            return int(pid)

    # Pass 2: first + last token match (handles middle names / extra tokens)
    prop_tokens = [t for t in prop_name.lower().split() if t not in {"jr.", "sr.", "jr", "sr", "ii", "iii", "iv"}]
    if len(prop_tokens) >= 2:
        first_tok = normalize_player_name(prop_tokens[0])
        last_tok = normalize_player_name(prop_tokens[-1])
        for pid, name in rows:
            name_tokens = [t for t in name.lower().split() if t not in {"jr.", "sr.", "jr", "sr", "ii", "iii", "iv"}]
            if len(name_tokens) >= 2:
                n_first = normalize_player_name(name_tokens[0])
                n_last = normalize_player_name(name_tokens[-1])
                if n_first == first_tok and n_last == last_tok:
                    return int(pid)

    # Pass 3: last name only (unique match only — avoids false positives)
    prop_last = normalize_player_name(prop_name.split()[-1])
    candidates = [int(pid) for pid, name in rows
                  if normalize_player_name(name.split()[-1]) == prop_last]
    if len(candidates) == 1:
        return candidates[0]

    return None


def main():
    logger.info("Fetching CBS injuries...")
    try:
        records = fetch_cbs_injuries()
    except Exception as e:
        logger.error(f"Failed to fetch injuries: {e}")
        return 1
    
    logger.info(f"Got {len(records)} injury records")

    fetched_at = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(DB_PATH)

    # Clear existing CBS records (full refresh)
    conn.execute("DELETE FROM player_injuries WHERE source = 'cbssports'")

    matched = 0
    unmatched = []
    for r in records:
        pid = find_player_id(r.player_name, conn)
        if pid is None:
            unmatched.append(r.player_name)
            continue
        conn.execute("""
            INSERT OR REPLACE INTO player_injuries
            (player_id, player_name, team_abbr, position, injury, status,
             status_normalized, update_date, fetched_at, source)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (pid, r.player_name, r.team_abbr, r.position, r.injury,
              r.status_raw, r.status_normalized, r.update_date,
              fetched_at, r.source))
        matched += 1

    conn.commit()
    logger.info(f"Matched {matched} players, {len(unmatched)} unmatched")
    if unmatched:
        logger.warning(f"Unmatched players: {unmatched}")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
