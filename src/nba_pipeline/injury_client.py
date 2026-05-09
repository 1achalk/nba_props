"""Injury feed client with status normalization.

Primary source: CBS Sports NBA injuries page (HTML scrape).
Falls back to: nothing for now (could add ESPN/Bleacher Report later).

Status normalization:
  OUT - definitely not playing
  DOUBTFUL - >75% chance not playing
  QUESTIONABLE - ~50% game-time decision
  PROBABLE - >75% chance playing
  HEALTHY - on injury report but expected to play
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup

CBS_URL = "https://www.cbssports.com/nba/injuries/"
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/120.0.0.0 Safari/537.36")

# Team name -> abbreviation mapping (CBS uses team names in headers)
CBS_TEAM_TO_ABBR = {
    "atlanta": "ATL", "boston": "BOS", "brooklyn": "BKN",
    "charlotte": "CHA", "chicago": "CHI", "cleveland": "CLE",
    "dallas": "DAL", "denver": "DEN", "detroit": "DET",
    "golden st.": "GSW", "houston": "HOU", "indiana": "IND",
    "l.a. clippers": "LAC", "l.a. lakers": "LAL",
    "memphis": "MEM", "miami": "MIA", "milwaukee": "MIL",
    "minnesota": "MIN", "new orleans": "NOP", "new york": "NYK",
    "oklahoma city": "OKC", "orlando": "ORL", "philadelphia": "PHI",
    "phoenix": "PHX", "portland": "POR", "sacramento": "SAC",
    "san antonio": "SAS", "toronto": "TOR", "utah": "UTA",
    "washington": "WAS",
}


@dataclass
class InjuryRecord:
    player_name: str
    team_abbr: str
    position: str
    injury: str
    status_raw: str
    status_normalized: str
    update_date: str
    source: str = "cbssports"


def normalize_status(raw: str) -> str:
    """Map CBS status strings to our normalized levels."""
    raw_lower = raw.lower().strip()
    if "out for the season" in raw_lower:
        return "OUT"
    if raw_lower.startswith("expected to be out"):
        return "OUT"
    if "game time decision" in raw_lower:
        return "QUESTIONABLE"
    if "doubtful" in raw_lower:
        return "DOUBTFUL"
    if "probable" in raw_lower:
        return "PROBABLE"
    if "questionable" in raw_lower:
        return "QUESTIONABLE"
    if "day-to-day" in raw_lower or "day to day" in raw_lower:
        return "QUESTIONABLE"
    return "QUESTIONABLE"  # default for unrecognized


def fetch_cbs_injuries(timeout: int = 30) -> list[InjuryRecord]:
    """Scrape CBS Sports NBA injuries page."""
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(CBS_URL, headers=headers, timeout=timeout)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    records = []

    # CBS structure: each team is in an h4 with team name, followed by a table
    # We look for tables and find the preceding team header
    for table in soup.find_all("table"):
        # Find the team name from the preceding header
        team_header = None
        for prev in table.find_all_previous(["h4", "h3", "h2"], limit=5):
            text = prev.get_text(strip=True).lower()
            for team_name, abbr in CBS_TEAM_TO_ABBR.items():
                if team_name in text:
                    team_header = abbr
                    break
            if team_header:
                break

        if not team_header:
            continue

        # Parse rows
        for tr in table.find_all("tr"):
            cells = tr.find_all("td")
            if len(cells) < 5:
                continue
            try:
                # Player name is in first cell, often duplicated (short name + full name)
                player_cell = cells[0]
                # Get the full name (typically the second link or full text after slash)
                links = player_cell.find_all("a")
                if len(links) >= 2:
                    player_name = links[1].get_text(strip=True)
                elif links:
                    player_name = links[0].get_text(strip=True)
                else:
                    player_name = player_cell.get_text(strip=True)

                position = cells[1].get_text(strip=True)
                update_date = cells[2].get_text(strip=True)
                injury = cells[3].get_text(strip=True)
                status_raw = cells[4].get_text(strip=True)

                if not player_name or not status_raw:
                    continue

                records.append(InjuryRecord(
                    player_name=player_name,
                    team_abbr=team_header,
                    position=position,
                    injury=injury,
                    status_raw=status_raw,
                    status_normalized=normalize_status(status_raw),
                    update_date=update_date,
                ))
            except Exception:
                continue  # skip malformed rows

    return records
