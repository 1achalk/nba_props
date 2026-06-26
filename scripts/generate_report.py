"""Generate a static HTML report for a slate of picks.

Usage:
    python -m scripts.generate_report                    # today
    python -m scripts.generate_report --date 2026-05-09  # specific date

Produces a self-contained HTML file with:
  - Performance summary (historical wins/losses)
  - Per-game cards with all picks for that matchup
  - Per-pick details: distribution SVG, recent form sparkline, opponent context
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom, norm

from src.nba_pipeline.config import DB_PATH, PROJECT_ROOT
from src.nba_pipeline.opponent_context import get_opponent_context
from src.nba_pipeline.stat_model import project_stat

STAT_COL_MAP = {
    "points": "points", "rebounds": "rebounds", "assists": "assists",
    "threes": "fg3m", "steals": "steals", "blocks": "blocks",
}

EDGE_HIGH = 0.15
EDGE_EXTREME = 0.30


def implied_prob(american_odds: int) -> float:
    if american_odds > 0:
        return 100.0 / (american_odds + 100.0)
    return abs(american_odds) / (abs(american_odds) + 100.0)


def build_distribution_svg(proj_dist: str, params: dict, line: float, side: str,
                           width: int = 220, height: int = 70) -> str:
    """Render the model's probability distribution as inline SVG.
    
    Shows the PDF/PMF as a filled curve, with a vertical bar at the line.
    The shaded area on the side we're betting is highlighted.
    """
    if proj_dist == "normal":
        mean = params.get("mean", 0)
        std = max(params.get("std", 1), 0.5)
        # Sample 100 points across mean ± 4 std
        x_min = max(0, mean - 3.5 * std)
        x_max = mean + 3.5 * std
        xs = np.linspace(x_min, x_max, 100)
        ys = norm.pdf(xs, mean, std)
    elif proj_dist == "nbinom":
        n = params.get("n", 1)
        p = params.get("p", 0.5)
        mean = n * (1 - p) / p if p > 0 else 5
        std = np.sqrt(n * (1 - p) / (p ** 2)) if p > 0 else 2
        x_max = int(min(mean + 4 * std, 30))
        xs_int = np.arange(0, x_max + 1)
        ys = nbinom.pmf(xs_int, n, p)
        xs = xs_int.astype(float)
    else:
        return f'<svg width="{width}" height="{height}"></svg>'
    
    # Normalize y to fit
    if ys.max() <= 0:
        return f'<svg width="{width}" height="{height}"></svg>'
    
    pad_x, pad_y = 12, 8
    plot_w = width - 2 * pad_x
    plot_h = height - 2 * pad_y
    
    x_range = xs.max() - xs.min() if xs.max() > xs.min() else 1
    y_max = ys.max() * 1.1
    
    # Map data → SVG coords
    def to_x(v):
        return pad_x + (v - xs.min()) / x_range * plot_w
    
    def to_y(v):
        return pad_y + plot_h - (v / y_max) * plot_h
    
    # Build curve path
    if proj_dist == "nbinom":
        # Bar chart for discrete
        bars = []
        bar_w = max(2, plot_w / len(xs) - 1)
        for x, y in zip(xs, ys):
            bx = to_x(x) - bar_w / 2
            by = to_y(y)
            bh = pad_y + plot_h - by
            color = _bar_color(x, line, side)
            bars.append(
                f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="{color}"/>'
            )
        curve_svg = "".join(bars)
    else:
        # Continuous: filled area under curve, split at line
        # Build two polygons: one for "win" side, one for "lose" side
        line_x = to_x(line)
        win_pts, lose_pts = [], []
        for x, y in zip(xs, ys):
            sx = to_x(x)
            sy = to_y(y)
            if (side == "over" and x > line) or (side == "under" and x < line):
                if not win_pts:
                    win_pts.append(f"{line_x:.1f},{pad_y + plot_h:.1f}")
                win_pts.append(f"{sx:.1f},{sy:.1f}")
            else:
                if not lose_pts:
                    lose_pts.append(f"{pad_x:.1f},{pad_y + plot_h:.1f}")
                lose_pts.append(f"{sx:.1f},{sy:.1f}")
        if win_pts:
            win_pts.append(f"{pad_x + plot_w:.1f},{pad_y + plot_h:.1f}" if side == "over"
                           else f"{line_x:.1f},{pad_y + plot_h:.1f}")
        if lose_pts:
            lose_pts.append(f"{line_x:.1f},{pad_y + plot_h:.1f}" if side == "over"
                            else f"{pad_x + plot_w:.1f},{pad_y + plot_h:.1f}")
        curve_svg = ""
        if lose_pts:
            curve_svg += f'<polygon points="{" ".join(lose_pts)}" fill="rgba(180,180,180,0.3)"/>'
        if win_pts:
            curve_svg += f'<polygon points="{" ".join(win_pts)}" fill="rgba(34,197,94,0.5)"/>'
    
    # Vertical line at the betting line
    line_x = to_x(line)
    line_marker = (
        f'<line x1="{line_x:.1f}" y1="{pad_y}" x2="{line_x:.1f}" '
        f'y2="{pad_y + plot_h}" stroke="#1a1a1a" stroke-width="1.5" stroke-dasharray="3,2"/>'
        f'<text x="{line_x:.1f}" y="{pad_y - 1}" text-anchor="middle" '
        f'font-size="9" fill="#1a1a1a" font-weight="600">{line:g}</text>'
    )
    
    # X-axis ticks at min, mean, max
    axis_y = pad_y + plot_h
    axis = (
        f'<line x1="{pad_x}" y1="{axis_y}" x2="{pad_x + plot_w}" y2="{axis_y}" '
        f'stroke="#888" stroke-width="0.5"/>'
    )
    
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'{curve_svg}{axis}{line_marker}</svg>'
    )


def _bar_color(x, line, side):
    if side == "over" and x > line:
        return "rgba(34,197,94,0.7)"
    if side == "under" and x < line:
        return "rgba(34,197,94,0.7)"
    return "rgba(180,180,180,0.4)"


def build_form_svg(values: list[float], width: int = 100, height: int = 28) -> str:
    """Render last-N games as a sparkline."""
    if not values or len(values) < 2:
        return f'<svg width="{width}" height="{height}"></svg>'
    
    y_min, y_max = min(values), max(values)
    y_range = y_max - y_min if y_max > y_min else 1
    
    pad = 4
    plot_w = width - 2 * pad
    plot_h = height - 2 * pad
    
    pts = []
    for i, v in enumerate(values):
        x = pad + i / max(1, len(values) - 1) * plot_w
        y = pad + plot_h - (v - y_min) / y_range * plot_h
        pts.append(f"{x:.1f},{y:.1f}")
    
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<polyline points="{" ".join(pts)}" fill="none" stroke="#0a0a0a" '
        f'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>'
        f'</svg>'
    )


def get_recent_form(player_id: int, market: str, as_of_date: str,
                    conn: sqlite3.Connection, n: int = 10) -> list[float]:
    """Return a player's last `n` values for a market (oldest-first), for the
    form sparkline. Only games with >5 minutes before `as_of_date` are included.
    """
    col = STAT_COL_MAP.get(market)
    if not col:
        return []
    rows = conn.execute(f"""
        SELECT pb.{col} FROM player_box pb
        JOIN games g ON pb.game_id = g.game_id
        WHERE pb.player_id = ? AND g.game_date < ? AND pb.minutes > 5
        ORDER BY g.game_date DESC LIMIT ?
    """, (player_id, as_of_date, n)).fetchall()
    return [float(r[0]) for r in reversed(rows)]


def get_performance_stats(conn: sqlite3.Connection) -> dict:
    """Pull all-time stats from picks_log."""
    df = pd.read_sql("""
        SELECT market, side, edge, won, pl, book_implied
        FROM picks_log WHERE won IS NOT NULL
    """, conn)
    if len(df) == 0:
        return {"n": 0}
    
    by_market = []
    for mkt, g in df.groupby("market"):
        by_market.append({
            "market": mkt,
            "n": len(g),
            "wr": g["won"].mean() * 100,
            "roi": g["pl"].sum() / len(g) * 100,
        })
    
    by_edge = []
    df["edge_pct"] = df["edge"] * 100
    for lo, hi in [(5, 8), (8, 11), (11, 15), (15, 30)]:
        b = df[(df["edge_pct"] >= lo) & (df["edge_pct"] < hi)]
        if len(b) > 0:
            by_edge.append({
                "bucket": f"{lo}-{hi}%",
                "n": len(b),
                "wr": b["won"].mean() * 100,
                "roi": b["pl"].sum() / len(b) * 100,
            })
    
    return {
        "n": len(df),
        "wr": df["won"].mean() * 100,
        "pl": df["pl"].sum(),
        "roi": df["pl"].sum() / len(df) * 100,
        "implied_avg": df["book_implied"].mean() * 100,
        "realized_edge": (df["won"].mean() - df["book_implied"].mean()) * 100,
        "predicted_edge": df["edge"].mean() * 100,
        "by_market": by_market,
        "by_edge": by_edge,
    }


def build_pick_card(pick: dict, proj, recent_form: list[float],
                    opp_ctx) -> str:
    """One pick → one HTML card."""
    edge_pct = pick["edge"] * 100
    if edge_pct >= EDGE_EXTREME * 100:
        flag_class = "skip"
        flag_text = "SKIP"
    elif edge_pct >= EDGE_HIGH * 100:
        flag_class = "high"
        flag_text = "HIGH"
    else:
        flag_class = "ok"
        flag_text = "TRUST"
    
    odds_str = f"+{pick['book_odds']}" if pick['book_odds'] > 0 else str(pick['book_odds'])
    side_arrow = "▲" if pick["side"] == "over" else "▼"
    
    # Distribution SVG
    dist_svg = build_distribution_svg(
        proj.distribution, proj.dist_params, pick["line"], pick["side"]
    )
    
    # Form sparkline + last value
    if recent_form:
        form_svg = build_form_svg(recent_form)
        form_avg = np.mean(recent_form)
        form_last = recent_form[-1]
        form_meta = f"avg {form_avg:.1f} • last {form_last:g}"
    else:
        form_svg = ""
        form_meta = "no data"
    
    # Opponent context line
    if opp_ctx:
        opp_meta = (
            f'def_rtg <strong>{opp_ctx.def_rtg:.1f}</strong> • '
            f'pace <strong>{opp_ctx.pace:.1f}</strong>'
        )
        # Adjustment direction
        if pick["market"] in ("points", "threes"):
            adj = opp_ctx.pts_adj if pick["market"] == "points" else opp_ctx.threes_adj
            if adj > 1.02:
                opp_meta += f' • <span class="adj-up">+{(adj-1)*100:.1f}% favorable</span>'
            elif adj < 0.98:
                opp_meta += f' • <span class="adj-dn">{(adj-1)*100:.1f}% tough</span>'
    else:
        opp_meta = "no opponent data"
    
    market_label = pick["market"].upper()
    
    return f"""
    <div class="pick-card flag-{flag_class}">
      <div class="pick-header">
        <div class="pick-player">{escape(pick['player_name'])}</div>
        <div class="pick-flag flag-{flag_class}">{flag_text}</div>
      </div>
      <div class="pick-line">
        <span class="market">{market_label}</span>
        <span class="side-arrow {pick['side']}">{side_arrow}</span>
        <span class="line-value">{pick['line']:g}</span>
        <span class="odds">{odds_str}</span>
      </div>
      <div class="pick-edge">
        <div class="edge-num">+{edge_pct:.1f}%</div>
        <div class="edge-detail">
          model {pick['model_prob']*100:.1f}% • implied {pick['book_implied']*100:.1f}%
        </div>
      </div>
      <div class="pick-projection">
        <div class="proj-label">model projects</div>
        <div class="proj-value">{pick['model_pred']:.1f} ± {pick['model_std']:.1f}</div>
      </div>
      <div class="pick-distribution">
        {dist_svg}
      </div>
      <div class="pick-meta">
        <div class="meta-row">
          <span class="meta-label">L10</span>
          {form_svg}
          <span class="meta-detail">{form_meta}</span>
        </div>
        <div class="meta-row">
          <span class="meta-label">vs {escape(pick.get('opp_team') or '?')}</span>
          <span class="meta-detail">{opp_meta}</span>
        </div>
      </div>
    </div>
    """


def build_game_section(game_key: str, picks: list[dict], conn: sqlite3.Connection,
                       target_date: str) -> str:
    """One game → top 15 picks by edge."""
    picks = sorted(picks, key=lambda p: -p["edge"])[:15]
    pick_html = []
    for pick in picks:
        try:
            # Need to resolve opponent_team_id for projection
            opp_team_id = None
            if pick.get("opp_team"):
                row = conn.execute(
                    "SELECT team_id FROM teams WHERE abbreviation = ?",
                    (pick["opp_team"],)
                ).fetchone()
                if row:
                    opp_team_id = int(row[0])
            
            proj = project_stat(
                pick["player_id"], pick["market"],
                as_of_date=target_date, opponent_team_id=opp_team_id
            )
            recent = get_recent_form(pick["player_id"], pick["market"],
                                     target_date, conn)
            opp_ctx = get_opponent_context(opp_team_id, target_date) if opp_team_id else None
            pick_html.append(build_pick_card(pick, proj, recent, opp_ctx))
        except Exception as e:
            pick_html.append(
                f'<div class="pick-card error">Error rendering pick for '
                f'{escape(pick["player_name"])}: {escape(str(e))}</div>'
            )
    
    avg_edge = np.mean([p["edge"] for p in picks]) * 100
    
    return f"""
    <section class="game">
      <header class="game-header">
        <h2>{escape(game_key)}</h2>
        <div class="game-stats">
          <span>{len(picks)} picks</span>
          <span>•</span>
          <span>avg edge +{avg_edge:.1f}%</span>
        </div>
      </header>
      <div class="picks-grid">
        {''.join(pick_html)}
      </div>
    </section>
    """


def build_top_picks_section(all_picks: list[dict]) -> str:
    """Top 10 picks across all games, ranked by edge (excluding SKIP-flagged)."""
    sorted_picks = sorted(
        [p for p in all_picks if p["edge"] < EDGE_EXTREME],
        key=lambda p: -p["edge"]
    )[:10]
    
    if not sorted_picks:
        return ""
    
    rows = []
    for i, p in enumerate(sorted_picks, 1):
        edge_pct = p["edge"] * 100
        flag = "high" if edge_pct >= EDGE_HIGH * 100 else "ok"
        side_arrow = "▲" if p["side"] == "over" else "▼"
        odds_str = f"+{p['book_odds']}" if p['book_odds'] > 0 else str(p['book_odds'])
        rows.append(f"""
          <tr class="flag-{flag}">
            <td class="rank">#{i}</td>
            <td class="player">{escape(p['player_name'])}</td>
            <td class="matchup">vs {escape(p.get('opp_team') or '?')}</td>
            <td class="market">{p['market']}</td>
            <td class="line"><span class="side-arrow {p['side']}">{side_arrow}</span> {p['line']:g}</td>
            <td class="odds">{odds_str}</td>
            <td class="edge"><strong>+{edge_pct:.1f}%</strong></td>
            <td class="model">{p['model_pred']:.1f}</td>
          </tr>
        """)
    
    return f"""
    <section class="top-picks">
      <header><h2>Top picks across all games</h2></header>
      <table>
        <thead>
          <tr>
            <th></th><th>Player</th><th>Game</th><th>Market</th>
            <th>Line</th><th>Odds</th><th>Edge</th><th>Model</th>
          </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </section>
    """


def build_performance_section(stats: dict) -> str:
    """Historical performance summary."""
    if stats["n"] == 0:
        return f"""
        <section class="performance">
          <header><h2>Track Record</h2></header>
          <div class="perf-empty">
            No graded picks yet. Run <code>daily_picks --results</code> after games complete.
          </div>
        </section>
        """
    
    pl_class = "positive" if stats["pl"] > 0 else "negative"
    edge_class = "positive" if stats["realized_edge"] > 0 else "negative"
    
    market_rows = "".join([
        f'<tr><td>{m["market"]}</td><td>{m["n"]}</td>'
        f'<td>{m["wr"]:.0f}%</td><td class="{"positive" if m["roi"] > 0 else "negative"}">'
        f'{"+" if m["roi"] >= 0 else ""}{m["roi"]:.0f}%</td></tr>'
        for m in stats["by_market"]
    ])
    
    edge_rows = "".join([
        f'<tr><td>{e["bucket"]}</td><td>{e["n"]}</td>'
        f'<td>{e["wr"]:.0f}%</td><td class="{"positive" if e["roi"] > 0 else "negative"}">'
        f'{"+" if e["roi"] >= 0 else ""}{e["roi"]:.0f}%</td></tr>'
        for e in stats["by_edge"]
    ])
    
    return f"""
    <section class="performance">
      <header><h2>Track Record</h2></header>
      <div class="perf-headline">
        <div class="perf-stat"><span class="num">{stats["n"]}</span><span class="lbl">picks</span></div>
        <div class="perf-stat"><span class="num">{stats["wr"]:.1f}%</span><span class="lbl">win rate</span></div>
        <div class="perf-stat"><span class="num {pl_class}">{stats["pl"]:+.1f}</span><span class="lbl">units P/L</span></div>
        <div class="perf-stat"><span class="num {pl_class}">{stats["roi"]:+.1f}%</span><span class="lbl">ROI</span></div>
        <div class="perf-stat"><span class="num {edge_class}">{stats["realized_edge"]:+.1f}%</span><span class="lbl">realized edge</span></div>
      </div>
      <div class="perf-detail-row">
        <div class="perf-table">
          <h3>By market</h3>
          <table>
            <thead><tr><th>Market</th><th>N</th><th>WR</th><th>ROI</th></tr></thead>
            <tbody>{market_rows}</tbody>
          </table>
        </div>
        <div class="perf-table">
          <h3>By edge bucket</h3>
          <table>
            <thead><tr><th>Edge</th><th>N</th><th>WR</th><th>ROI</th></tr></thead>
            <tbody>{edge_rows}</tbody>
          </table>
        </div>
      </div>
      <div class="perf-context">
        Model predicted edge averaged <strong>{stats["predicted_edge"]:+.1f}%</strong>,
        realized was <strong>{stats["realized_edge"]:+.1f}%</strong>.
        Average book implied probability: <strong>{stats["implied_avg"]:.1f}%</strong>.
      </div>
    </section>
    """


CSS = r"""
:root {
  --ink: #0a0a0a;
  --paper: #faf8f3;
  --rule: #1a1a1a;
  --accent: #1d4ed8;
  --green: #16a34a;
  --red: #b91c1c;
  --amber: #b45309;
  --muted: #6b6b6b;
  --soft: #ebe7df;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: 'Iowan Old Style', 'Palatino Linotype', Palatino, 'Book Antiqua', Georgia, serif;
  background: var(--paper);
  color: var(--ink);
  line-height: 1.45;
  padding: 32px 24px 80px;
  max-width: 1280px;
  margin: 0 auto;
}

header.masthead {
  border-top: 4px solid var(--ink);
  border-bottom: 1px solid var(--ink);
  padding: 14px 0 18px;
  margin-bottom: 28px;
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
}
.masthead h1 {
  font-family: 'Bodoni 72', 'Bodoni Moda', Bodoni, Didot, serif;
  font-size: 38px;
  font-weight: 900;
  letter-spacing: -0.02em;
  line-height: 1;
}
.masthead .subtitle {
  font-style: italic;
  color: var(--muted);
  font-size: 14px;
}
.masthead .date {
  font-family: 'IBM Plex Mono', 'Menlo', monospace;
  font-size: 13px;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--ink);
  border-left: 2px solid var(--ink);
  padding-left: 10px;
}

section { margin-bottom: 36px; }
section > header {
  border-bottom: 2px solid var(--ink);
  padding-bottom: 8px;
  margin-bottom: 18px;
}
section h2 {
  font-family: 'Bodoni 72', 'Bodoni Moda', Bodoni, serif;
  font-size: 26px;
  font-weight: 900;
  letter-spacing: -0.01em;
}

/* Performance section */
.perf-empty {
  font-style: italic; color: var(--muted);
  padding: 18px; background: var(--soft); border-radius: 4px;
}
.perf-empty code {
  font-family: 'IBM Plex Mono', monospace; font-size: 13px;
  background: white; padding: 2px 6px; border-radius: 3px;
}
.perf-headline {
  display: flex; gap: 32px; flex-wrap: wrap;
  padding: 16px 0; border-bottom: 1px solid var(--soft);
  margin-bottom: 18px;
}
.perf-stat { display: flex; flex-direction: column; }
.perf-stat .num {
  font-family: 'Bodoni 72', serif; font-size: 30px; font-weight: 900;
  letter-spacing: -0.02em; line-height: 1;
}
.perf-stat .lbl {
  font-family: 'IBM Plex Mono', monospace; font-size: 11px;
  text-transform: uppercase; letter-spacing: 0.1em;
  color: var(--muted); margin-top: 4px;
}
.positive { color: var(--green); }
.negative { color: var(--red); }
.perf-detail-row {
  display: grid; grid-template-columns: 1fr 1fr; gap: 24px;
}
.perf-table h3 {
  font-family: 'IBM Plex Mono', monospace; font-size: 11px;
  text-transform: uppercase; letter-spacing: 0.1em;
  color: var(--muted); margin-bottom: 8px; font-weight: 600;
}
.perf-table table { width: 100%; border-collapse: collapse; }
.perf-table th {
  font-family: 'IBM Plex Mono', monospace; font-size: 10px;
  text-transform: uppercase; letter-spacing: 0.05em;
  font-weight: 600; padding: 6px 8px; text-align: left;
  color: var(--muted); border-bottom: 1px solid var(--soft);
}
.perf-table td {
  padding: 6px 8px; font-size: 14px; border-bottom: 1px solid var(--soft);
}
.perf-context {
  margin-top: 12px; font-style: italic; color: var(--muted); font-size: 13px;
}

/* Top picks table */
.top-picks table {
  width: 100%; border-collapse: collapse;
  font-size: 14px;
}
.top-picks th {
  font-family: 'IBM Plex Mono', monospace; font-size: 11px;
  text-transform: uppercase; letter-spacing: 0.05em;
  text-align: left; padding: 8px 10px;
  border-bottom: 2px solid var(--ink); font-weight: 600;
}
.top-picks td { padding: 10px; border-bottom: 1px solid var(--soft); }
.top-picks tr.flag-high { background: #fef9f0; }
.top-picks .rank {
  font-family: 'IBM Plex Mono', monospace; color: var(--muted);
  font-size: 11px; width: 28px;
}
.top-picks .player { font-weight: 600; }
.top-picks .matchup { color: var(--muted); font-size: 13px; }
.top-picks .market {
  font-family: 'IBM Plex Mono', monospace; font-size: 11px;
  text-transform: uppercase; letter-spacing: 0.05em;
}
.top-picks .edge { font-size: 16px; }
.top-picks .odds {
  font-family: 'IBM Plex Mono', monospace; font-size: 13px;
}

/* Game sections */
.game-header {
  display: flex; align-items: baseline; justify-content: space-between;
  border-bottom: 2px solid var(--ink); padding-bottom: 8px; margin-bottom: 18px;
}
.game-header h2 {
  font-family: 'Bodoni 72', serif; font-size: 24px;
}
.game-stats {
  font-family: 'IBM Plex Mono', monospace; font-size: 12px;
  color: var(--muted); display: flex; gap: 8px;
  text-transform: uppercase; letter-spacing: 0.05em;
}

.picks-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
  gap: 14px;
}

/* Pick cards */
.pick-card {
  border: 1px solid var(--ink);
  background: white;
  padding: 14px 16px;
  display: grid;
  grid-template-columns: 1fr auto;
  grid-template-rows: auto auto auto auto auto;
  grid-template-areas:
    "header header"
    "line projection"
    "edge edge"
    "dist dist"
    "meta meta";
  gap: 10px 12px;
}
.pick-card.flag-high { border-color: var(--amber); border-width: 1px 1px 1px 4px; }
.pick-card.flag-skip { border-color: var(--red); border-width: 1px 1px 1px 4px; opacity: 0.85; }
.pick-card.error { color: var(--red); font-style: italic; padding: 8px; font-size: 12px; }
.flag-skip { background: var(--red); color: white; }

.pick-header {
  grid-area: header;
  display: flex; align-items: baseline; justify-content: space-between;
  border-bottom: 1px solid var(--soft); padding-bottom: 6px;
}
.pick-player {
  font-family: 'Bodoni 72', serif; font-size: 18px; font-weight: 700;
  letter-spacing: -0.01em;
}
.pick-flag {
  font-family: 'IBM Plex Mono', monospace; font-size: 9px;
  text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700;
  padding: 2px 6px; border-radius: 2px;
}
.pick-flag.flag-ok { background: var(--accent); color: white; }
.pick-flag.flag-high { background: var(--amber); color: var(--bg); }
.pick-flag.flag-skip { background: var(--red); color: white; }

.pick-line {
  grid-area: line;
  display: flex; align-items: baseline; gap: 6px; flex-wrap: wrap;
}
.pick-line .market {
  font-family: 'IBM Plex Mono', monospace; font-size: 10px;
  text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted);
}
.pick-line .side-arrow {
  font-size: 16px; font-weight: 700;
}
.pick-line .side-arrow.over { color: var(--green); }
.pick-line .side-arrow.under { color: var(--red); }
.pick-line .line-value {
  font-family: 'Bodoni 72', serif; font-size: 22px; font-weight: 800;
}
.pick-line .odds {
  font-family: 'IBM Plex Mono', monospace; font-size: 12px;
  color: var(--muted); margin-left: auto;
}

.pick-projection {
  grid-area: projection;
  text-align: right;
  border-left: 1px solid var(--soft); padding-left: 12px;
}
.proj-label {
  font-family: 'IBM Plex Mono', monospace; font-size: 9px;
  text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted);
}
.proj-value {
  font-family: 'Bodoni 72', serif; font-size: 16px; font-weight: 700;
}

.pick-edge {
  grid-area: edge;
  display: flex; align-items: baseline; gap: 12px;
  padding: 8px 0;
  border-top: 1px solid var(--soft); border-bottom: 1px solid var(--soft);
}
.edge-num {
  font-family: 'Bodoni 72', serif; font-size: 24px; font-weight: 900;
  color: var(--accent);
}
.edge-detail {
  font-family: 'IBM Plex Mono', monospace; font-size: 11px;
  color: var(--muted);
}

.pick-distribution { grid-area: dist; }
.pick-distribution svg { display: block; }

.pick-meta {
  grid-area: meta;
  font-family: 'IBM Plex Mono', monospace; font-size: 11px;
  display: flex; flex-direction: column; gap: 4px;
}
.meta-row { display: flex; align-items: center; gap: 8px; }
.meta-label {
  text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted);
  min-width: 36px; font-weight: 600;
}
.meta-detail { color: var(--ink); }
.meta-detail strong { font-weight: 700; }
.adj-up { color: var(--green); }
.adj-dn { color: var(--red); }

footer.colophon {
  margin-top: 60px; padding-top: 16px;
  border-top: 1px solid var(--ink);
  font-family: 'IBM Plex Mono', monospace; font-size: 11px;
  color: var(--muted);
  display: flex; justify-content: space-between;
}
"""


def main() -> int:
    """Build the static HTML report for a slate and write it to the output dir."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "reports"))
    args = parser.parse_args()
    
    target_date = args.date or datetime.now().strftime("%Y-%m-%d")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    
    # Pull picks for this date
    picks_df = pd.read_sql("""
        SELECT player_name, player_id, market, side, line, best_book,
               book_odds, book_implied, model_prob, edge, model_pred,
               model_std, n_games, opp_team
        FROM picks_log
        WHERE pick_date = ?
        ORDER BY edge DESC
    """, conn, params=(target_date,))
    
    # Deduplicate: same player+market only the highest-edge side
    if not picks_df.empty:
        picks_df = (picks_df
            .sort_values("edge", ascending=False)
            .drop_duplicates(subset=["player_name", "market"], keep="first"))
    
    all_picks = picks_df.to_dict("records")
    
    # Group by game using opp_team — find player team via player_box
    def get_player_team_abbr(pid):
        r = conn.execute("""
            SELECT t.abbreviation FROM player_box pb
            JOIN games g ON pb.game_id = g.game_id
            JOIN teams t ON pb.team_id = t.team_id
            WHERE pb.player_id = ? AND g.game_date < ?
            ORDER BY g.game_date DESC LIMIT 1
        """, (pid, target_date)).fetchone()
        return r[0] if r else "?"
    
    games = {}
    for pick in all_picks:
        team = get_player_team_abbr(pick["player_id"])
        opp = pick.get("opp_team") or "?"
        # Use sorted abbreviations as canonical game key
        key = " vs ".join(sorted([team, opp]))
        if key not in games:
            games[key] = []
        games[key].append(pick)
    
    # Build sections
    perf_stats = get_performance_stats(conn)
    perf_section = build_performance_section(perf_stats)
    
    top_section = build_top_picks_section(all_picks) if all_picks else ""
    
    game_sections = []
    for game_key in sorted(games.keys()):
        game_picks = sorted(games[game_key], key=lambda p: -p["edge"])
        game_sections.append(build_game_section(game_key, game_picks, conn, target_date))
    
    if not game_sections:
        game_sections.append(
            '<section><div class="perf-empty">No picks found for this date. '
            'Run <code>daily_picks --date ' + target_date + '</code> first.</div></section>'
        )
    
    # Generated timestamp
    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M %Z")
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Picks Report — {target_date}</title>
  <style>{CSS}</style>
</head>
<body>
  <header class="masthead">
    <div>
      <h1>The Daily Edge</h1>
      <div class="subtitle">A model's view of the slate · {len(all_picks)} picks across {len(games)} games</div>
    </div>
    <div class="date">{target_date}</div>
  </header>
  
  {perf_section}
  {top_section}
  {''.join(game_sections)}
  
  <footer class="colophon">
    <span>Generated {escape(gen_time)}</span>
    <span>Model: stat_model w/ PBP defensive context · DraftKings only</span>
  </footer>
</body>
</html>
"""
    
    output_path = output_dir / f"report_{target_date}.html"
    output_path.write_text(html)
    
    conn.close()
    
    print(f"\nReport generated: {output_path}")
    print(f"  Picks: {len(all_picks)}")
    print(f"  Games: {len(games)}")
    print(f"  Historical track record: {perf_stats['n']} graded picks")
    print(f"\nOpen with:")
    print(f"  open {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
