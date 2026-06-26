"""Travel and rest features.

Built from the schedule. NBA arenas are hardcoded with lat/lon.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Iterable

# NBA arena coordinates (lat, lon). Approximate arena locations.
# Used for travel distance calculation.
ARENA_COORDS = {
    "ATL": (33.7573, -84.3963),
    "BOS": (42.3662, -71.0621),
    "BKN": (40.6826, -73.9754),
    "CHA": (35.2251, -80.8392),
    "CHI": (41.8807, -87.6742),
    "CLE": (41.4965, -81.6882),
    "DAL": (32.7905, -96.8103),
    "DEN": (39.7487, -105.0077),
    "DET": (42.3411, -83.0553),
    "GSW": (37.7680, -122.3877),
    "HOU": (29.7508, -95.3621),
    "IND": (39.7640, -86.1555),
    "LAC": (33.9430, -118.3414),  # Intuit Dome
    "LAL": (34.0430, -118.2673),
    "MEM": (35.1382, -90.0506),
    "MIA": (25.7814, -80.1870),
    "MIL": (43.0451, -87.9173),
    "MIN": (44.9795, -93.2761),
    "NOP": (29.9490, -90.0820),
    "NYK": (40.7505, -73.9934),
    "OKC": (35.4634, -97.5151),
    "ORL": (28.5392, -81.3839),
    "PHI": (39.9012, -75.1719),
    "PHX": (33.4457, -112.0712),
    "POR": (45.5316, -122.6668),
    "SAC": (38.5802, -121.4997),
    "SAS": (29.4271, -98.4375),
    "TOR": (43.6435, -79.3791),
    "UTA": (40.7683, -111.9011),
    "WAS": (38.8981, -77.0209),
}

# Approximate timezone offsets from UTC (standard time, ignoring DST for simplicity)
ARENA_TZ = {
    "ATL": -5, "BOS": -5, "BKN": -5, "CHA": -5, "CHI": -6, "CLE": -5,
    "DAL": -6, "DEN": -7, "DET": -5, "GSW": -8, "HOU": -6, "IND": -5,
    "LAC": -8, "LAL": -8, "MEM": -6, "MIA": -5, "MIL": -6, "MIN": -6,
    "NOP": -6, "NYK": -5, "OKC": -6, "ORL": -5, "PHI": -5, "PHX": -7,
    "POR": -8, "SAC": -8, "SAS": -6, "TOR": -5, "UTA": -7, "WAS": -5,
}


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in miles."""
    r = 3958.8  # Earth radius in miles
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def distance_between(team_a: str, team_b: str) -> float:
    """Travel miles between two team cities. 0 if same team."""
    if team_a == team_b:
        return 0.0
    if team_a not in ARENA_COORDS or team_b not in ARENA_COORDS:
        return 0.0
    lat1, lon1 = ARENA_COORDS[team_a]
    lat2, lon2 = ARENA_COORDS[team_b]
    return haversine_miles(lat1, lon1, lat2, lon2)


def timezone_change(team_a: str, team_b: str) -> int:
    """Hours of timezone change moving from team_a's city to team_b's."""
    if team_a not in ARENA_TZ or team_b not in ARENA_TZ:
        return 0
    return ARENA_TZ[team_b] - ARENA_TZ[team_a]


def compute_rest_features(
    team_abbr: str,
    game_date: datetime,
    prior_games: Iterable[tuple[datetime, str]],
) -> dict:
    """Compute rest/travel features for a single team-game.

    Args:
        team_abbr: this team's abbreviation
        game_date: date of the current game
        prior_games: iterable of (date, opponent_or_host_abbr) for this team's
                     previous games, sorted descending (most recent first).
                     The 'host' is the team whose city the game was played in.
    Returns:
        dict ready to insert into team_rest table.
    """
    prior = list(prior_games)
    prior_dates = [d for d, _ in prior]

    days_rest = (
        (game_date - prior_dates[0]).days if prior_dates else 7
    )
    is_b2b = 1 if days_rest == 1 else 0

    # 3-in-4: this game + at least 2 of the prior 3 days had a game
    last_3_days = [d for d in prior_dates if (game_date - d).days <= 3]
    is_3in4 = 1 if len(last_3_days) >= 2 else 0

    # 4-in-6: this game + at least 3 of the prior 5 days had a game
    last_5_days = [d for d in prior_dates if (game_date - d).days <= 5]
    is_4in6 = 1 if len(last_5_days) >= 3 else 0

    # Travel: distance from previous host city to current host city
    if prior:
        prev_host = prior[0][1]
        travel_miles = distance_between(prev_host, team_abbr)
        tz_change = timezone_change(prev_host, team_abbr)
    else:
        travel_miles = 0.0
        tz_change = 0

    return {
        "days_rest": days_rest,
        "is_back_to_back": is_b2b,
        "is_3in4": is_3in4,
        "is_4in6": is_4in6,
        "travel_miles": travel_miles,
        "timezone_change": tz_change,
    }
