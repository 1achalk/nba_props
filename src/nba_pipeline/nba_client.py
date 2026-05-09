"""Rate-limited wrapper around nba_api.

The stats.nba.com API throttles aggressively. We:
  - sleep between requests
  - retry on transient failures
  - log everything
"""
from __future__ import annotations

import time
from typing import Any, Callable

from nba_api.stats.library.http import NBAStatsHTTP
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import (
    NBA_API_DELAY_SECONDS,
    NBA_API_MAX_RETRIES,
    NBA_API_TIMEOUT_SECONDS,
    setup_logging,
)

logger = setup_logging(__name__)

# Inject browser-like headers — stats.nba.com blocks default requests
NBAStatsHTTP.headers = {
    "Host": "stats.nba.com",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Connection": "keep-alive",
}


class NBAApiError(Exception):
    """Raised when an nba_api call fails after retries."""


@retry(
    stop=stop_after_attempt(NBA_API_MAX_RETRIES),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type((NBAApiError, ConnectionError, TimeoutError)),
    reraise=True,
)
def call_endpoint(endpoint_cls: Callable, **kwargs: Any) -> Any:
    """Instantiate an nba_api endpoint with retries + delay."""
    kwargs.setdefault("timeout", NBA_API_TIMEOUT_SECONDS)
    try:
        time.sleep(NBA_API_DELAY_SECONDS)
        result = endpoint_cls(**kwargs)
        return result
    except Exception as e:
        logger.warning(f"{endpoint_cls.__name__} failed: {e}")
        raise NBAApiError(f"{endpoint_cls.__name__}: {e}") from e


def fetch_season_games(season: str, season_type: str = "Regular Season"):
    """Fetch all games for a season."""
    from nba_api.stats.endpoints import LeagueGameLog
    logger.info(f"Fetching {season} {season_type} games")
    result = call_endpoint(
        LeagueGameLog,
        season=season,
        season_type_all_star=season_type,
    )
    return result.get_data_frames()[0]


def fetch_box_score(game_id: str):
    """Fetch traditional + advanced box score for one game."""
    from nba_api.stats.endpoints import (
        BoxScoreAdvancedV3,
        BoxScoreTraditionalV3,
    )
    trad = call_endpoint(BoxScoreTraditionalV3, game_id=game_id)
    adv = call_endpoint(BoxScoreAdvancedV3, game_id=game_id)
    return {
        "traditional": trad.get_data_frames(),
        "advanced": adv.get_data_frames(),
    }


def fetch_shot_chart(game_id: str, season: str):
    """Fetch all shots from a single game."""
    from nba_api.stats.endpoints import ShotChartDetail
    result = call_endpoint(
        ShotChartDetail,
        team_id=0,
        player_id=0,
        game_id_nullable=game_id,
        season_nullable=season,
        context_measure_simple="FGA",
    )
    return result.get_data_frames()[0]


def fetch_teams():
    """Get reference list of all NBA teams."""
    from nba_api.stats.static import teams as nba_teams
    return nba_teams.get_teams()


def fetch_players():
    """Get reference list of all players (active + retired)."""
    from nba_api.stats.static import players as nba_players
    return nba_players.get_players()
