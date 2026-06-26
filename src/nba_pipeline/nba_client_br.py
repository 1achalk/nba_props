"""Basketball Reference data client - replaces nba_api wrapper."""
from __future__ import annotations

import hashlib
import time
from io import StringIO

import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import NBA_API_MAX_RETRIES, NBA_API_TIMEOUT_SECONDS, setup_logging

logger = setup_logging(__name__)
BASE_URL = "https://www.basketball-reference.com"
BR_DELAY_SECONDS = 3.5

_session = requests.Session()
_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
})

BR_TEAM_MAP = {
    "ATL":"ATL","BOS":"BOS","BKN":"BRK","CHA":"CHO","CHI":"CHI","CLE":"CLE",
    "DAL":"DAL","DEN":"DEN","DET":"DET","GSW":"GSW","HOU":"HOU","IND":"IND",
    "LAC":"LAC","LAL":"LAL","MEM":"MEM","MIA":"MIA","MIL":"MIL","MIN":"MIN",
    "NOP":"NOP","NYK":"NYK","OKC":"OKC","ORL":"ORL","PHI":"PHI","PHX":"PHO",
    "POR":"POR","SAC":"SAC","SAS":"SAS","TOR":"TOR","UTA":"UTA","WAS":"WAS",
}

TEAMS_STATIC = [
    {"id":1610612737,"full_name":"Atlanta Hawks","abbreviation":"ATL","nickname":"Hawks","city":"Atlanta","state":"Georgia","year_founded":1949},
    {"id":1610612738,"full_name":"Boston Celtics","abbreviation":"BOS","nickname":"Celtics","city":"Boston","state":"Massachusetts","year_founded":1946},
    {"id":1610612751,"full_name":"Brooklyn Nets","abbreviation":"BKN","nickname":"Nets","city":"Brooklyn","state":"New York","year_founded":1976},
    {"id":1610612766,"full_name":"Charlotte Hornets","abbreviation":"CHA","nickname":"Hornets","city":"Charlotte","state":"North Carolina","year_founded":1988},
    {"id":1610612741,"full_name":"Chicago Bulls","abbreviation":"CHI","nickname":"Bulls","city":"Chicago","state":"Illinois","year_founded":1966},
    {"id":1610612739,"full_name":"Cleveland Cavaliers","abbreviation":"CLE","nickname":"Cavaliers","city":"Cleveland","state":"Ohio","year_founded":1970},
    {"id":1610612742,"full_name":"Dallas Mavericks","abbreviation":"DAL","nickname":"Mavericks","city":"Dallas","state":"Texas","year_founded":1980},
    {"id":1610612743,"full_name":"Denver Nuggets","abbreviation":"DEN","nickname":"Nuggets","city":"Denver","state":"Colorado","year_founded":1976},
    {"id":1610612765,"full_name":"Detroit Pistons","abbreviation":"DET","nickname":"Pistons","city":"Detroit","state":"Michigan","year_founded":1948},
    {"id":1610612744,"full_name":"Golden State Warriors","abbreviation":"GSW","nickname":"Warriors","city":"Golden State","state":"California","year_founded":1946},
    {"id":1610612745,"full_name":"Houston Rockets","abbreviation":"HOU","nickname":"Rockets","city":"Houston","state":"Texas","year_founded":1967},
    {"id":1610612754,"full_name":"Indiana Pacers","abbreviation":"IND","nickname":"Pacers","city":"Indiana","state":"Indiana","year_founded":1976},
    {"id":1610612746,"full_name":"LA Clippers","abbreviation":"LAC","nickname":"Clippers","city":"Los Angeles","state":"California","year_founded":1970},
    {"id":1610612747,"full_name":"Los Angeles Lakers","abbreviation":"LAL","nickname":"Lakers","city":"Los Angeles","state":"California","year_founded":1948},
    {"id":1610612763,"full_name":"Memphis Grizzlies","abbreviation":"MEM","nickname":"Grizzlies","city":"Memphis","state":"Tennessee","year_founded":1995},
    {"id":1610612748,"full_name":"Miami Heat","abbreviation":"MIA","nickname":"Heat","city":"Miami","state":"Florida","year_founded":1988},
    {"id":1610612749,"full_name":"Milwaukee Bucks","abbreviation":"MIL","nickname":"Bucks","city":"Milwaukee","state":"Wisconsin","year_founded":1968},
    {"id":1610612750,"full_name":"Minnesota Timberwolves","abbreviation":"MIN","nickname":"Timberwolves","city":"Minnesota","state":"Minnesota","year_founded":1989},
    {"id":1610612740,"full_name":"New Orleans Pelicans","abbreviation":"NOP","nickname":"Pelicans","city":"New Orleans","state":"Louisiana","year_founded":2002},
    {"id":1610612752,"full_name":"New York Knicks","abbreviation":"NYK","nickname":"Knicks","city":"New York","state":"New York","year_founded":1946},
    {"id":1610612760,"full_name":"Oklahoma City Thunder","abbreviation":"OKC","nickname":"Thunder","city":"Oklahoma City","state":"Oklahoma","year_founded":1967},
    {"id":1610612753,"full_name":"Orlando Magic","abbreviation":"ORL","nickname":"Magic","city":"Orlando","state":"Florida","year_founded":1989},
    {"id":1610612755,"full_name":"Philadelphia 76ers","abbreviation":"PHI","nickname":"76ers","city":"Philadelphia","state":"Pennsylvania","year_founded":1949},
    {"id":1610612756,"full_name":"Phoenix Suns","abbreviation":"PHX","nickname":"Suns","city":"Phoenix","state":"Arizona","year_founded":1968},
    {"id":1610612757,"full_name":"Portland Trail Blazers","abbreviation":"POR","nickname":"Trail Blazers","city":"Portland","state":"Oregon","year_founded":1970},
    {"id":1610612758,"full_name":"Sacramento Kings","abbreviation":"SAC","nickname":"Kings","city":"Sacramento","state":"California","year_founded":1948},
    {"id":1610612759,"full_name":"San Antonio Spurs","abbreviation":"SAS","nickname":"Spurs","city":"San Antonio","state":"Texas","year_founded":1976},
    {"id":1610612761,"full_name":"Toronto Raptors","abbreviation":"TOR","nickname":"Raptors","city":"Toronto","state":"Ontario","year_founded":1995},
    {"id":1610612762,"full_name":"Utah Jazz","abbreviation":"UTA","nickname":"Jazz","city":"Utah","state":"Utah","year_founded":1974},
    {"id":1610612764,"full_name":"Washington Wizards","abbreviation":"WAS","nickname":"Wizards","city":"Washington","state":"District of Columbia","year_founded":1961},
]
TEAM_ID_BY_ABBR = {t["abbreviation"]: t["id"] for t in TEAMS_STATIC}


class NBAApiError(Exception):
    pass


@retry(stop=stop_after_attempt(NBA_API_MAX_RETRIES),
       wait=wait_exponential(multiplier=2, min=2, max=30),
       retry=retry_if_exception_type((NBAApiError, ConnectionError, TimeoutError)),
       reraise=True)
def _fetch_html(path: str) -> str:
    url = f"{BASE_URL}{path}"
    try:
        time.sleep(BR_DELAY_SECONDS)
        r = _session.get(url, timeout=NBA_API_TIMEOUT_SECONDS)
        if r.status_code == 429:
            logger.warning("Rate limited by BR, backing off 60s")
            time.sleep(60)
            raise NBAApiError("Rate limited (429)")
        r.raise_for_status()
        return r.text
    except requests.exceptions.HTTPError as e:
        # 404 means a month has no games yet (e.g., June before playoffs end)
        # Log at debug level instead of warning to keep logs clean
        if e.response is not None and e.response.status_code == 404:
            logger.debug(f"BR fetch {url} 404 (no games for this month)")
        else:
            logger.warning(f"BR fetch {url} failed: {e}")
        raise NBAApiError(f"GET {url}: {e}") from e
    except requests.exceptions.RequestException as e:
        logger.warning(f"BR fetch {url} failed: {e}")
        raise NBAApiError(f"GET {url}: {e}") from e


def _season_to_br_year(season: str) -> int:
    return int(season.split("-")[0]) + 1


def fetch_season_games(season: str, season_type: str = "Regular Season"):
    """Fetch a season's schedule/results from Basketball-Reference.

    Walks the per-month schedule pages, concatenates them, and returns a
    normalized two-rows-per-game frame (one per team). Only Regular Season is
    supported. Months with no games yet (404) are skipped.
    """
    if season_type != "Regular Season":
        logger.warning(f"BR client only supports Regular Season; got {season_type}")
    year = _season_to_br_year(season)
    logger.info(f"Fetching {season} games from basketball-reference")
    months = ["october","november","december","january","february","march","april","may","june"]
    all_games = []
    for month in months:
        path = f"/leagues/NBA_{year}_games-{month}.html"
        try:
            html = _fetch_html(path)
        except NBAApiError:
            continue
        try:
            tables = pd.read_html(StringIO(html))
        except ValueError:
            continue
        if not tables:
            continue
        df = tables[0]
        df = df[df["Date"] != "Date"]
        if df.empty:
            continue
        all_games.append(df)
    if not all_games:
        return pd.DataFrame()
    df = pd.concat(all_games, ignore_index=True)
    return _normalize_schedule(df, season)


def _normalize_schedule(df, season):
    df = df.rename(columns={
        df.columns[0]: "Date",
        df.columns[2]: "Visitor",
        df.columns[3]: "VisitorPts",
        df.columns[4]: "Home",
        df.columns[5]: "HomePts",
    })
    rows = []
    name_to_abbr = {t["full_name"]: t["abbreviation"] for t in TEAMS_STATIC}
    aliases = {"Los Angeles Clippers":"LAC","LA Clippers":"LAC"}
    for _, r in df.iterrows():
        try:
            date = pd.to_datetime(r["Date"]).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
        home_name = r.get("Home","")
        away_name = r.get("Visitor","")
        home_abbr = aliases.get(home_name, name_to_abbr.get(home_name))
        away_abbr = aliases.get(away_name, name_to_abbr.get(away_name))
        if not home_abbr or not away_abbr:
            continue
        game_id = f"{date.replace('-','')}-{away_abbr}-{home_abbr}"
        rows.append({"SEASON_ID":season,"TEAM_ID":TEAM_ID_BY_ABBR[home_abbr],
                     "TEAM_ABBREVIATION":home_abbr,"GAME_ID":game_id,
                     "GAME_DATE":date,"MATCHUP":f"{home_abbr} vs. {away_abbr}","WL":None})
        rows.append({"SEASON_ID":season,"TEAM_ID":TEAM_ID_BY_ABBR[away_abbr],
                     "TEAM_ABBREVIATION":away_abbr,"GAME_ID":game_id,
                     "GAME_DATE":date,"MATCHUP":f"{away_abbr} @ {home_abbr}","WL":None})
    return pd.DataFrame(rows)


def fetch_box_score(game_id: str):
    """Fetch a single game's box score page, returning the raw HTML and all
    parsed tables. `game_id` must be the "YYYYMMDD-AWAY-HOME" form this client emits.
    """
    parts = game_id.split("-")
    if len(parts) != 3:
        raise ValueError(f"Unexpected game_id: {game_id}")
    date_str, _away, home = parts
    br_home = BR_TEAM_MAP.get(home, home)
    path = f"/boxscores/{date_str}0{br_home}.html"
    html = _fetch_html(path)
    tables = pd.read_html(StringIO(html))
    return {"traditional_html": html, "tables": tables}


def fetch_shot_chart(game_id: str, season: str):
    """Shot-chart stub. Basketball-Reference does not expose per-shot data the way
    the old nba_api source did, so this returns an empty frame to preserve the
    interface expected by callers.
    """
    return pd.DataFrame()


def fetch_teams():
    """Return the static list of 30 NBA teams (id, name, abbr, city, ...)."""
    return TEAMS_STATIC


def fetch_players():
    """Fetch the current season's active players from the per-game stats page.

    Returns a list of {id, full_name, is_active}; player ids are stable hashes of
    the player name (Basketball-Reference has no numeric player id we ingest here).
    """
    from datetime import datetime
    today = datetime.now()
    year = today.year + 1 if today.month >= 10 else today.year
    path = f"/leagues/NBA_{year}_per_game.html"
    try:
        html = _fetch_html(path)
        tables = pd.read_html(StringIO(html))
    except (NBAApiError, ValueError) as e:
        logger.warning(f"Could not fetch player list: {e}")
        return []
    if not tables:
        return []
    df = tables[0]
    df = df[df["Player"] != "Player"].dropna(subset=["Player"])
    players = []
    seen = set()
    for _, r in df.iterrows():
        name = r["Player"]
        if name in seen:
            continue
        seen.add(name)
        pid = int(hashlib.md5(name.encode()).hexdigest()[:8], 16)
        players.append({"id": pid, "full_name": name, "is_active": True})
    return players
