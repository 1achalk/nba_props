"""Daily bet recommendations from the model.

Usage:
    python -m scripts.daily_picks --teams SAS,MIN
    python -m scripts.daily_picks --teams SAS,MIN,DET,CLE,OKC,LAL
    python -m scripts.daily_picks --results
    python -m scripts.daily_picks --results --date 2026-05-07
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from src.nba_pipeline.config import DB_PATH, setup_logging
from src.nba_pipeline.database import init_db
from src.nba_pipeline.stat_model import project_stat

logger = setup_logging("daily_picks")

MARKET_MAP = {
    "points": "points", "rebounds": "rebounds", "assists": "assists",
    "threePointersMade": "threes", "threes_made": "threes", "threes": "threes",
    "steals": "steals", "blocks": "blocks",
}

STAT_COL_MAP = {
    "points": "points", "rebounds": "rebounds", "assists": "assists",
    "threes": "fg3m", "steals": "steals", "blocks": "blocks",
}

MAINSTREAM_BOOKS = {"draftkings", "fanduel", "betmgm", "caesars", "espnbet"}

EDGE_HIGH_THRESHOLD = 0.15
EDGE_EXTREME_THRESHOLD = 0.30


def implied_prob(american_odds: int) -> float:
    if american_odds > 0:
        return 100.0 / (american_odds + 100.0)
    return abs(american_odds) / (abs(american_odds) + 100.0)


def fair_prob_to_american(p: float) -> int:
    if p <= 0.001 or p >= 0.999:
        return 0
    if p >= 0.5:
        return int(round(-100 * p / (1 - p)))
    return int(round(100 * (1 - p) / p))


def edge_flag(edge: float) -> str:
    if edge >= EDGE_EXTREME_THRESHOLD:
        return "SKIP"
    if edge >= EDGE_HIGH_THRESHOLD:
        return "HIGH"
    return "OK"


def normalize(name: str) -> str:
    return name.lower().replace("-", "").replace(" ", "").replace(".", "").replace("'", "").replace("jr", "").replace("sr", "").replace("iii", "").replace("ii", "").replace("iv", "")


def find_player_id(prop_name: str, conn: sqlite3.Connection) -> Optional[int]:
    target = normalize(prop_name)
    rows = conn.execute("""
        SELECT p.player_id, p.full_name, COUNT(pb.game_id) AS games
        FROM players p LEFT JOIN player_box pb ON p.player_id = pb.player_id
        GROUP BY p.player_id HAVING games > 0
    """).fetchall()

    # Pass 1: exact normalized match
    for pid, name, _ in rows:
        if normalize(name) == target:
            return int(pid)

    # Pass 2: first + last token match
    prop_tokens = prop_name.lower().split()
    if len(prop_tokens) >= 2:
        first_tok = normalize(prop_tokens[0])
        last_tok = normalize(prop_tokens[-1])
        candidates = []
        for pid, name, games in rows:
            name_tokens = name.lower().split()
            if len(name_tokens) >= 2:
                n_first = normalize(name_tokens[0])
                n_last = normalize(name_tokens[-1])
                if n_first == first_tok and n_last == last_tok:
                    candidates.append((games, int(pid)))
        if candidates:
            return sorted(candidates, reverse=True)[0][1]

    # Pass 3: last name only (unique match only)
    last_tok = normalize(prop_name.split()[-1]) if " " in prop_name else target
    candidates = [(g, int(pid)) for pid, name, g in rows
                  if normalize(name.split()[-1]) == last_tok]
    if len(candidates) == 1:
        return int(candidates[0][1])

    return None


def get_player_team(player_id: int, as_of_date: str, conn: sqlite3.Connection) -> Optional[int]:
    """Get player's team for today — checks schedule first, falls back to last box score."""
    # Check if the player's most recent team is scheduled to play today
    row = conn.execute("""
        SELECT pb.team_id FROM player_box pb
        JOIN games g ON pb.game_id = g.game_id
        WHERE pb.player_id = ?
        ORDER BY g.game_date DESC LIMIT 1
    """, (player_id,)).fetchone()
    
    if not row:
        return None
    
    team_id = int(row[0])
    
    # Verify this team plays today (from schedule)
    sched = conn.execute("""
        SELECT game_id FROM schedule
        WHERE game_date = ? AND (home_team_id = ? OR away_team_id = ?)
        LIMIT 1
    """, (as_of_date, team_id, team_id)).fetchone()
    
    if sched:
        return team_id
    
    # Team not in today's schedule — player may have been traded, or team has a day off
    return team_id  # Still return it so we can attempt an opponent lookup


def get_opponent_team(player_team_id: int, as_of_date: str, conn: sqlite3.Connection) -> Optional[int]:
    """Find opponent team_id from schedule for today's game."""
    row = conn.execute("""
        SELECT 
            CASE WHEN home_team_id = ? THEN away_team_id ELSE home_team_id END
        FROM schedule
        WHERE game_date = ? AND (home_team_id = ? OR away_team_id = ?)
        LIMIT 1
    """, (player_team_id, as_of_date, player_team_id, player_team_id)).fetchone()
    return int(row[0]) if row else None


def get_team_abbr(team_id: int, conn: sqlite3.Connection) -> Optional[str]:
    row = conn.execute(
        "SELECT abbreviation FROM teams WHERE team_id = ?", (team_id,)
    ).fetchone()
    return row[0] if row else None


def sanity_check(pred: float, line: float, market: str) -> bool:
    """Reject projections that are wildly misaligned with the line.
    
    Uses market-specific bounds since different stats have different variance.
    The key insight: conditional projections (assuming player plays) should be
    within a reasonable range of the offered line.
    """
    if line <= 0 or pred < 0:
        return False
    
    # Market-specific ratio bounds (pred / line)
    bounds = {
        "points":   (0.30, 3.0),
        "rebounds": (0.30, 3.0),
        "assists":  (0.25, 3.5),
        "threes":   (0.20, 4.0),
        "steals":   (0.20, 5.0),
        "blocks":   (0.20, 5.0),
    }
    lo, hi = bounds.get(market, (0.25, 4.0))
    ratio = pred / line
    return lo <= ratio <= hi


def get_actual_stat(player_id: int, game_date: str, market: str,
                    conn: sqlite3.Connection) -> Optional[float]:
    col = STAT_COL_MAP.get(market)
    if not col:
        return None
    row = conn.execute(f"""
        SELECT pb.{col} FROM player_box pb
        JOIN games g ON pb.game_id = g.game_id
        WHERE pb.player_id = ? AND g.game_date = ?
        LIMIT 1
    """, (player_id, game_date)).fetchone()
    return float(row[0]) if row else None


def generate_picks(
    target_date: str,
    team_filter: Optional[list[str]],
    min_edge: float,
    markets: list[str],
    log_picks: bool = True,
    backtest: bool = False,
) -> pd.DataFrame:
    # Ensure picks_log table exists
    init_db()
    
    conn = sqlite3.connect(DB_PATH)

    # Resolve team filter to IDs
    filter_team_ids = set()
    if team_filter:
        for abbr in team_filter:
            row = conn.execute(
                "SELECT team_id FROM teams WHERE abbreviation = ?",
                (abbr.upper(),)
            ).fetchone()
            if row:
                filter_team_ids.add(int(row[0]))
            else:
                logger.warning(f"Team not found: {abbr}")

    # Pull odds — use the most recent snapshot for each player/market/line/book
    odds_df = pd.read_sql("""
        SELECT player_name, market, book, line, over_odds, under_odds,
               MAX(snapshot_time) AS snapshot_time
        FROM prop_odds
        WHERE over_odds IS NOT NULL AND under_odds IS NOT NULL
        GROUP BY player_name, market, book, line
    """, conn)

    odds_df = odds_df[odds_df["book"].str.lower().isin(MAINSTREAM_BOOKS)].copy()
    odds_df["our_market"] = odds_df["market"].map(MARKET_MAP)
    odds_df = odds_df.dropna(subset=["our_market"])
    odds_df = odds_df[odds_df["our_market"].isin(markets)]

    if odds_df.empty:
        logger.warning("No odds rows found matching mainstream books + requested markets.")
        conn.close()
        return pd.DataFrame()

    generated_at = datetime.now(timezone.utc).isoformat()
    rows = []
    rows_to_log = []
    skipped_no_player = 0
    skipped_sanity = 0
    skipped_team = 0
    skipped_no_game = 0

    grouped = odds_df.groupby(["player_name", "our_market", "line"])
    for (pname, market, line), group in grouped:
        pid = find_player_id(pname, conn)
        if pid is None:
            skipped_no_player += 1
            continue

        # Get player's team and check if they play today
        player_team_id = get_player_team(pid, target_date, conn)
        
        # Team filter check
        if filter_team_ids and player_team_id not in filter_team_ids:
            skipped_team += 1
            continue

        # Find opponent from today's schedule
        opp_id = None
        opp_abbr = None
        player_abbr = None
        if player_team_id:
            player_abbr = get_team_abbr(player_team_id, conn)
            opp_id = get_opponent_team(player_team_id, target_date, conn)
            if opp_id:
                opp_abbr = get_team_abbr(opp_id, conn)

        try:
            proj = project_stat(
                pid,
                market,
                as_of_date=target_date,
                opponent_team_id=opp_id,
                db_path=DB_PATH,
            )
        except Exception as e:
            logger.debug(f"Projection failed {pname} {market}: {e}")
            continue

        if proj.expected == 0.0 and proj.p_play == 0.0:
            # Player is injured/DNP — skip entirely
            continue

        # Sanity check with market-aware bounds
        if not sanity_check(proj.expected, line, market):
            skipped_sanity += 1
            logger.debug(f"Sanity fail: {pname} {market} pred={proj.expected:.1f} line={line}")
            continue

        model_p_over = proj.p_over(line)
        model_p_under = 1.0 - model_p_over

        # Find best odds across books for each side
        best_over_idx = group["over_odds"].idxmax()
        best_under_idx = group["under_odds"].idxmax()
        best_over = group.loc[best_over_idx]
        best_under = group.loc[best_under_idx]

        for side, price_row, model_p in [
            ("over", best_over, model_p_over),
            ("under", best_under, model_p_under),
        ]:
            book_odds = int(price_row["over_odds" if side == "over" else "under_odds"])
            book_implied = implied_prob(book_odds)
            edge = model_p - book_implied

            if edge < min_edge:
                continue

            actual = won = pl = None
            if backtest:
                actual = get_actual_stat(pid, target_date, market, conn)
                if actual is not None:
                    won = 1 if (side == "over" and actual > line) or \
                               (side == "under" and actual < line) else 0
                    pl = (book_odds/100.0 if book_odds > 0
                          else 100.0/abs(book_odds)) if won else -1.0

            row = {
                "date": target_date,
                "player": pname,
                "player_id": pid,
                "market": market,
                "side": side,
                "line": line,
                "opp": opp_abbr or "?",
                "best_book": price_row["book"],
                "book_odds": book_odds,
                "book_implied": round(book_implied, 3),
                "model_prob": round(model_p, 3),
                "edge": round(edge, 3),
                "flag": edge_flag(edge),
                "model_pred": proj.expected,
                "model_std": proj.std,
                "p_play": proj.p_play,
                "n_games": proj.n_games,
                "snapshot_time": group["snapshot_time"].max(),
                "actual": actual, "won": won, "pl": pl,
            }
            rows.append(row)

            if log_picks and edge_flag(edge) != "SKIP":
                rows_to_log.append({
                    "pick_date": target_date,
                    "generated_at": generated_at,
                    "player_name": pname,
                    "player_id": pid,
                    "market": market,
                    "side": side,
                    "line": line,
                    "best_book": price_row["book"],
                    "book_odds": book_odds,
                    "book_implied": round(book_implied, 3),
                    "model_prob": round(model_p, 3),
                    "edge": round(edge, 3),
                    "model_pred": proj.expected,
                    "model_std": proj.std,
                    "n_games": proj.n_games,
                    "opp_team": opp_abbr,
                })

    if log_picks and rows_to_log:
        log_conn = sqlite3.connect(DB_PATH)
        log_conn.executemany("""
            INSERT INTO picks_log
            (pick_date, generated_at, player_name, player_id, market, side,
             line, best_book, book_odds, book_implied, model_prob, edge,
             model_pred, model_std, n_games, opp_team)
            VALUES
            (:pick_date, :generated_at, :player_name, :player_id, :market,
             :side, :line, :best_book, :book_odds, :book_implied, :model_prob,
             :edge, :model_pred, :model_std, :n_games, :opp_team)
        """, rows_to_log)
        log_conn.commit()
        log_conn.close()
        logger.info(f"Logged {len(rows_to_log)} picks")

    conn.close()
    logger.info(
        f"Skipped: {skipped_no_player} no-match, {skipped_sanity} sanity-fail, "
        f"{skipped_team} wrong-team, {skipped_no_game} no-game-today"
    )

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values("edge", ascending=False).reset_index(drop=True)


def check_results(target_date: Optional[str] = None) -> None:
    conn = sqlite3.connect(DB_PATH)
    checked_at = datetime.now(timezone.utc).isoformat()

    query = "SELECT id, pick_date, player_id, market, side, line FROM picks_log WHERE won IS NULL"
    params = []
    if target_date:
        query += " AND pick_date = ?"
        params.append(target_date)

    pending = conn.execute(query, params).fetchall()
    if not pending:
        print("No pending picks to check.")
        conn.close()
        return

    print(f"Checking {len(pending)} picks...")
    updated = 0
    for pick_id, pick_date, player_id, market, side, line in pending:
        actual = get_actual_stat(player_id, pick_date, market, conn)
        if actual is None:
            continue
        won = 1 if (side == "over" and actual > line) or \
                   (side == "under" and actual < line) else 0
        book_odds = conn.execute(
            "SELECT book_odds FROM picks_log WHERE id = ?", (pick_id,)
        ).fetchone()[0]
        pl = (book_odds/100.0 if book_odds > 0 else 100.0/abs(book_odds)) if won else -1.0
        conn.execute("""
            UPDATE picks_log SET actual=?, won=?, pl=?, result_checked_at=?
            WHERE id=?
        """, (actual, won, pl, checked_at, pick_id))
        updated += 1

    conn.commit()
    print(f"Updated {updated} picks with results.")
    print_running_stats(conn, target_date)
    conn.close()


def print_running_stats(conn: sqlite3.Connection,
                        filter_date: Optional[str] = None) -> None:
    query = """
        SELECT pick_date, market, side, line, model_pred, edge,
               book_odds, book_implied, model_prob, player_name,
               opp_team, best_book, actual, won, pl
        FROM picks_log WHERE won IS NOT NULL
    """
    params = []
    if filter_date:
        query += " AND pick_date = ?"
        params.append(filter_date)

    df = pd.read_sql(query, conn, params=params)
    if len(df) == 0:
        print("No completed picks yet.")
        return

    print(f"\n{'=' * 65}")
    print("RUNNING PERFORMANCE STATS")
    print(f"{'=' * 65}")
    print(f"Total picks:  {len(df)}")
    print(f"Win rate:     {df['won'].mean()*100:.1f}%")
    print(f"Total P/L:    {df['pl'].sum():+.2f} units")
    print(f"ROI:          {df['pl'].sum()/len(df)*100:+.1f}%")

    avg_imp = df["book_implied"].mean()
    actual_wr = df["won"].mean()
    print(f"Realized edge: {(actual_wr - avg_imp)*100:+.1f}% "
          f"(model predicted {df['edge'].mean()*100:+.1f}%)")

    print(f"\nBy market:")
    for mkt, g in df.groupby("market"):
        wr = g["won"].mean() * 100
        roi = g["pl"].sum() / len(g) * 100
        print(f"  {mkt:<12} {len(g):>3} picks  WR={wr:.0f}%  ROI={roi:+.0f}%")

    print(f"\nBy edge bucket:")
    df["edge_pct"] = df["edge"] * 100
    for lo, hi in [(5,8),(8,11),(11,15),(15,30)]:
        b = df[(df["edge_pct"] >= lo) & (df["edge_pct"] < hi)]
        if len(b):
            wr = b["won"].mean() * 100
            roi = b["pl"].sum() / len(b) * 100
            print(f"  {lo}-{hi}%:  {len(b):>3} picks  WR={wr:.0f}%  ROI={roi:+.0f}%")

    print(f"\nLast 10 results:")
    print(f"{'Date':<11}{'Player':<20}{'Mkt':<10}{'Side':<6}"
          f"{'Line':<7}{'Pred':<7}{'Actual':<8}{'Result'}")
    for _, r in df.tail(10).iterrows():
        result = "WIN " if r["won"] == 1 else "LOSS"
        print(f"  {r['pick_date']:<9}{r['player_name'][:18]:<20}"
              f"{r['market']:<10}{r['side']:<6}{r['line']:<7.1f}"
              f"{r['model_pred']:<7.1f}{r['actual']:<8.0f}{result}")


def _odds_age_warning(df: pd.DataFrame) -> None:
    """Warn if the odds snapshot is stale — lines move a lot close to tipoff."""
    if "snapshot_time" not in df.columns or df["snapshot_time"].isna().all():
        return
    try:
        from datetime import timezone as tz
        latest = pd.to_datetime(df["snapshot_time"].dropna()).max()
        if latest.tzinfo is None:
            latest = latest.replace(tzinfo=tz.utc)
        age_mins = (datetime.now(tz.utc) - latest).total_seconds() / 60
        if age_mins > 60:
            print(f"\n⚠️  ODDS ARE {age_mins:.0f} MINUTES OLD — run refresh_odds before betting")
        else:
            print(f"  Odds snapshot: {age_mins:.0f} min ago")
    except Exception:
        pass


def print_picks_report(df: pd.DataFrame, backtest: bool = False) -> None:
    if len(df) == 0:
        print("\nNo bets meet the criteria.")
        return

    _odds_age_warning(df)

    ok   = df[df["flag"] == "OK"].copy()
    high = df[df["flag"] == "HIGH"].copy()

    def _print_section(section_df: pd.DataFrame, label: str, emoji: str) -> None:
        if len(section_df) == 0:
            return

        # Group by opponent so picks are organised by game
        section_df = section_df.copy()
        section_df["opp"] = section_df["opp"].fillna("?")
        games = section_df["opp"].unique()

        print(f"\n{emoji}  {label}  ({len(section_df)} bets)")
        print("=" * 78)

        for game in sorted(games):
            game_df = section_df[section_df["opp"] == game]
            print(f"\n  vs {game}")
            print(f"  {'─' * 74}")
            print(f"  {'Player':<22} {'Mkt':<10} {'Side':<6} {'Line':>5}  "
                  f"{'Pred':>5}  {'Edge':>6}  {'Odds':>6}  {'Model%':>7}")
            print(f"  {'─' * 74}")

            for _, r in game_df.iterrows():
                edge_str  = f"+{r['edge']*100:.1f}%"
                odds_val  = r['book_odds']
                odds_str  = f"+{odds_val}" if odds_val > 0 else str(odds_val)
                model_str = f"{r['model_prob']*100:.1f}%"
                side_str  = r['side'].capitalize()
                print(f"  {r['player'][:21]:<22} {r['market']:<10} {side_str:<6} "
                      f"{r['line']:>5.1f}  {r['model_pred']:>5.1f}  "
                      f"{edge_str:>6}  {odds_str:>6}  {model_str:>7}", end="")
                if backtest and r.get("actual") is not None:
                    result = " WIN" if r["won"] == 1 else " LOSS"
                    print(f"  {r['actual']:.0f}{result}", end="")
                print()

        print()

    _print_section(ok,   "TRUST (5–15% edge)", "✅")
    _print_section(high, "HIGH EDGE — verify line is current (15–30%)", "⚠️")

    # SKIP section: just a compact count, no detail
    skip = df[df["flag"] == "SKIP"]
    if len(skip) > 0:
        print(f"🚫  {len(skip)} bets flagged as likely model error (>30% edge) — not shown")

    if backtest and "won" in df.columns and df["won"].notna().any():
        scored = df[df["won"].notna()]
        wr  = scored["won"].mean() * 100
        roi = scored["pl"].sum() / len(scored) * 100
        print(f"\nBacktest: {int(scored['won'].sum())}/{len(scored)} "
              f"({wr:.1f}% WR)  ROI: {roi:+.1f}%")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None)
    parser.add_argument("--teams", default=None,
                        help="Comma-separated abbreviations e.g. SAS,MIN")
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--results", action="store_true")
    parser.add_argument("--min-edge", type=float, default=0.05)
    parser.add_argument("--markets",
                        default="points,rebounds,assists,threes")
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args()

    target_date = args.date or datetime.now().strftime("%Y-%m-%d")

    if args.results:
        check_results(target_date if args.date else None)
        return 0

    team_filter = [t.strip().upper() for t in args.teams.split(",")] \
        if args.teams else None
    markets = [m.strip() for m in args.markets.split(",")]

    print(f"Generating picks for {target_date}")
    print(f"  Teams:   {', '.join(team_filter) if team_filter else 'ALL'}")
    print(f"  Markets: {', '.join(markets)}")
    print(f"  Min edge: {args.min_edge*100:.0f}%")

    df = generate_picks(
        target_date, team_filter, args.min_edge, markets,
        log_picks=not args.no_log,
        backtest=args.backtest,
    )
    print_picks_report(df, backtest=args.backtest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
