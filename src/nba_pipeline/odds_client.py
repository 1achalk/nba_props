"""SportsGameOdds API client.

Pulls player prop odds for NBA games. Returns normalized rows ready
for the prop_odds table.

Docs: https://sportsgameodds.com/docs
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterator

import requests

from .config import SPORTSGAMEODDS_API_KEY, setup_logging

logger = setup_logging(__name__)

BASE_URL = "https://api.sportsgameodds.com/v2"

# Only pull from DraftKings — most reliable lines, least stale/placeholder data.
# Expand this set later if you want to shop lines across books again.
ALLOWED_BOOKS = {"draftkings"}

# Hard sanity bounds per market (min_line, max_line).
# Lines outside these are almost certainly data errors or placeholder values.
LINE_SANITY_BOUNDS: dict[str, tuple[float, float]] = {
    "points":                  (0.5,  60.0),
    "rebounds":                (0.5,  25.0),
    "assists":                 (0.5,  20.0),
    "threes":                  (0.5,  10.0),
    "threes_made":             (0.5,  10.0),
    "steals":                  (0.5,   6.0),
    "blocks":                  (0.5,   6.0),
    "points_rebounds_assists": (1.5,  80.0),
}

# Reject odds more extreme than these — beyond +-600 is almost always a
# data error (missing line, misformatted value, etc.)
MIN_ODDS = -600
MAX_ODDS = +600

# Markets we care about for player props
PLAYER_PROP_MARKETS = [
    "points-{player}-game-ou-over",
    "rebounds-{player}-game-ou-over",
    "assists-{player}-game-ou-over",
    "threes_made-{player}-game-ou-over",
    "steals-{player}-game-ou-over",
    "blocks-{player}-game-ou-over",
    "points_rebounds_assists-{player}-game-ou-over",
]


class SportsGameOddsClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or SPORTSGAMEODDS_API_KEY
        if not self.api_key:
            raise ValueError(
                "No SPORTSGAMEODDS_API_KEY found. Set it in your .env file."
            )
        self.session = requests.Session()
        self.session.headers.update({"x-api-key": self.api_key})

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{BASE_URL}{path}"
        r = self.session.get(url, params=params or {}, timeout=30)
        r.raise_for_status()
        return r.json()

    def fetch_nba_events(self, include_alt_lines: bool = True) -> list[dict]:
        """Fetch upcoming NBA events with odds available."""
        params = {
            "leagueID": "NBA",
            "oddsAvailable": "true",
            "includeAltLines": "true" if include_alt_lines else "false",
            "limit": 25,
        }
        events = []
        cursor = None
        while True:
            if cursor:
                params["cursor"] = cursor
            data = self._get("/events", params=params)
            events.extend(data.get("data", []))
            cursor = data.get("nextCursor")
            if not cursor:
                break
        logger.info(f"Fetched {len(events)} NBA events from SportsGameOdds")
        return events

    def extract_player_props(self, events: list[dict]) -> Iterator[dict]:
        """Walk event payloads and yield normalized prop_odds rows.

        Each event has an 'odds' object keyed by oddID. Player prop oddIDs
        look like 'points-LEBRON_JAMES_1_NBA-game-ou-over' with bookmaker
        prices nested under byBookmaker.

        Only emits rows for books in ALLOWED_BOOKS and lines that pass
        LINE_SANITY_BOUNDS + odds range checks.
        """
        snapshot = datetime.now(timezone.utc).isoformat()
        skipped_book = skipped_line = skipped_odds = 0

        for event in events:
            game_id = event.get("eventID")
            odds_dict = event.get("odds", {}) or {}
            for odd_id, odd_data in odds_dict.items():
                # We only want over/under player props
                parts = odd_id.split("-")
                if len(parts) < 5:
                    continue
                stat, entity, period, bet_type, side = parts[0], parts[1], parts[2], parts[3], parts[4]
                if bet_type != "ou" or period != "game" or side != "over":
                    continue
                # Skip team-level (entity is team abbr) — we want player entities
                # Player entities look like 'LEBRON_JAMES_1_NBA'
                if "_NBA" not in entity:
                    continue

                player_name = entity.replace("_NBA", "").rsplit("_", 1)[0].replace("_", " ").title()

                # Find matching 'under' side for the same line
                under_id = f"{stat}-{entity}-{period}-{bet_type}-under"
                under_data = odds_dict.get(under_id, {})

                by_bookmaker = odd_data.get("byBookmaker", {}) or {}
                under_books = under_data.get("byBookmaker", {}) or {}

                for book, book_odds in by_bookmaker.items():
                    # --- Book filter ---
                    if book.lower() not in ALLOWED_BOOKS:
                        skipped_book += 1
                        continue

                    if not book_odds.get("available"):
                        continue

                    line = book_odds.get("overUnder")
                    over_price = book_odds.get("odds")
                    under_price = under_books.get(book, {}).get("odds")

                    if line is None:
                        continue

                    line = float(line)

                    # --- Line sanity check ---
                    bounds = LINE_SANITY_BOUNDS.get(stat)
                    if bounds:
                        lo, hi = bounds
                        if not (lo <= line <= hi):
                            skipped_line += 1
                            logger.debug(
                                f"Skipping {player_name} {stat} line={line} "
                                f"(outside bounds {lo}-{hi})"
                            )
                            continue

                    # --- Odds sanity check ---
                    over_int = _to_int(over_price)
                    under_int = _to_int(under_price)
                    if over_int is not None and not (MIN_ODDS <= over_int <= MAX_ODDS):
                        skipped_odds += 1
                        logger.debug(
                            f"Skipping {player_name} {stat} over_odds={over_int} out of range"
                        )
                        continue
                    if under_int is not None and not (MIN_ODDS <= under_int <= MAX_ODDS):
                        skipped_odds += 1
                        logger.debug(
                            f"Skipping {player_name} {stat} under_odds={under_int} out of range"
                        )
                        continue

                    yield {
                        "snapshot_time": snapshot,
                        "game_id": game_id,
                        "player_id": None,
                        "player_name": player_name,
                        "market": stat,
                        "book": book,
                        "line": line,
                        "over_odds": over_int,
                        "under_odds": under_int,
                    }

        logger.info(
            f"extract_player_props: skipped {skipped_book} non-DK rows, "
            f"{skipped_line} bad lines, {skipped_odds} bad odds"
        )


def _to_int(american_odds) -> int | None:
    """Convert '+150' / '-200' / 150 to int. None if missing."""
    if american_odds is None:
        return None
    try:
        s = str(american_odds).replace("+", "")
        return int(s)
    except (ValueError, TypeError):
        return None
