# NBA Player Props Model

A from-scratch pipeline and probabilistic model for NBA player prop bets (points,
rebounds, assists, threes, steals, blocks). It ingests box scores, play-by-play
defensive data, injury reports, and sportsbook odds into a local SQLite database,
projects a full outcome distribution for each player/market, and compares the
model's implied probabilities against book lines to find theoretical edges.

It then **forward-tests those edges honestly** — and the headline result is the
most important thing in this repository.

## Headline result: no durable edge

Across **739 forward-tested picks** graded against closing lines, the model's
**realized edge was −1.0%**, versus a **predicted edge of +11.1%**.

In plain terms: the model believed it was finding large mispricings, but once
those picks were settled against sharp closing lines, the apparent edge
evaporated. The +11.1% was an artifact of the model disagreeing with the market,
not of the model being right.

**That negative result is the point of this project.** Sharp prop markets are
extremely efficient, and a single-developer model built on public data should be
expected to lose to the closing line. The value here is not a winning betting
system — it's the discipline of building a complete, calibrated pipeline,
forward-testing it without fooling myself, and reporting the result truthfully
even though it isn't the result I was hoping for. I would rather show a rigorous
negative finding than a backtest-overfit "edge" that wouldn't survive contact
with a real book.

See [Evaluation methodology](#evaluation-methodology) for how the numbers are
computed and why I trust the negative result more than the backtest.

## What the project does

1. **Collects data** — historical and daily box scores scraped from
   Basketball-Reference, team defensive profiles from the pbpstats.com
   play-by-play API, injury reports, and player prop odds from the
   SportsGameOdds API. Everything lands in a local SQLite DB.
2. **Projects minutes** — a positional minutes model that redistributes a team's
   240 available minutes when players are injured or rested (see below).
3. **Projects each stat** — a per-minute Bayesian rate is multiplied by projected
   minutes to produce an expected value *and a full distribution* for every market.
4. **Prices and compares** — converts distributions into over/under probabilities,
   compares them to the book's implied probabilities, and ranks the largest
   disagreements as candidate bets.
5. **Forward-tests** — logs every pick, grades it once the game settles, and
   tracks realized vs. predicted edge over time.

## Modeling approach

### Minutes model — positional cascading spillover

[`minutes_model.py`](src/nba_pipeline/minutes_model.py) builds a per-player
minutes baseline by blending season average, a recent-form window, career
average, and a prior, weighted by sample size (shrinkage toward the prior when
data is thin). When players are `OUT`/`DOUBTFUL`, their minutes are freed and
**redistributed to teammates in the same position bucket** (guards → guards,
etc.), with overflow cascading league-wide, while respecting realistic per-player
minute caps. The team total is normalized to 240. Playoff rotations tighten, so a
player-specific playoff multiplier is applied during the postseason.

### Per-minute Bayesian rate projection

Rather than projecting a counting stat directly, [`stat_model.py`](src/nba_pipeline/stat_model.py)
projects a **per-minute rate** and scales it by projected minutes. The rate is a
weighted blend of season, recent, career, and league-prior rates — each weight
shrunk by how much data backs it — so a player with five games leans on the prior
while a veteran leans on his own track record. The rate is then adjusted for
opponent defensive strength (from play-by-play profiles) and, in the postseason,
for the player's historical regular-season-vs-playoff scoring ratio.

Because props are conditional on the player actually playing, projections use a
**conditional expectation**: expected minutes are divided by the probability of
playing, so the distribution reflects "given he plays, here's the outcome," and
the play probability is applied separately.

### Market-specific distributions

Different stats have different shapes, so each market uses an appropriate
distribution rather than forcing one everywhere:

| Markets | Distribution | Why |
|---|---|---|
| Points, rebounds, assists | **Normal** | Higher-count stats; roughly symmetric around the mean |
| Threes, steals, blocks | **Negative binomial** | Low-count, over-dispersed (variance > mean); a Poisson would understate the tails |

The negative binomial is parameterized from the projected mean and an empirical
dispersion estimate, so the tail probabilities — which is where prop value lives —
reflect each player's actual variance rather than a Poisson assumption.

## Evaluation methodology

The model is judged two ways, and I deliberately trust the second one more.

**Backtesting** ([`backtest_minutes.py`](scripts/backtest_minutes.py),
[`backtest_points.py`](scripts/backtest_points.py)) replays historical games using
only data available before each game (point-in-time queries, no leakage) and
scores projection error. Backtests are useful for catching regressions but are
easy to overfit, so they are not the verdict.

**Forward-testing** ([`track_edge.py`](scripts/track_edge.py)) is the verdict.
Every recommended pick is logged before tip-off and graded after settlement. For
each pick it records the model's predicted edge and, after grading, the
**realized edge = (win rate − book-implied probability)**. Picks generated before
a known wrong-side bug fix are filtered out so the clean sample is honest. The
tracker reports cumulative realized edge per slate; the rule set in advance was
simple: *if realized edge stabilizes positive over enough slates, the edge is
real; if it decays toward zero, it isn't.* It decayed toward zero.

This is the discipline the project is meant to demonstrate: pre-commit to a
falsifiable success criterion, forward-test against closing lines, and accept the
answer.

## Tech stack

- **Python 3.12**
- **SQLite** — local store for box scores, schedule, odds, injuries, and the picks log
- **pandas / NumPy** — data wrangling and feature construction
- **SciPy** (`scipy.stats`) — normal / negative-binomial / Poisson distributions
- **BeautifulSoup + lxml** — HTML table parsing for Basketball-Reference box scores
- **requests + tenacity** — rate-limited, retrying API/scraper clients
- **python-dotenv** — secret/config management

## Setup

Requires Python 3.12+.

```bash
# 1. Clone and enter the repo
git clone <your-repo-url> nba_props
cd nba_props

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure secrets
cp .env.example .env
# Open .env and add your SportsGameOdds API key (https://sportsgameodds.com).
# .env is gitignored — never commit it.
```

## Running the pipeline

```bash
# One-time historical backfill (rate-limited; takes a while)
python -m scripts.backfill --seasons 2022-23 2023-24 2024-25 2025-26

# Build derived features (run after a backfill or when new games land)
python -m scripts.build_team_features   # rolling team offense/defense
python -m scripts.build_rest_features    # rest + travel (team_rest table)

# Daily incremental update (yesterday's results, today's schedule, injuries)
python -m scripts.daily_update

# Refresh current prop odds (run a few times pre-tipoff to track line movement)
python -m scripts.refresh_odds

# Generate the day's candidate picks
python -m scripts.daily_picks

# Score realized vs. predicted edge on graded picks
python -m scripts.track_edge
```

The SQLite database and any generated reports are created locally on first run.
They are intentionally **not** committed — the raw data is scraped from
third-party sources and is not redistributed here; regenerate it with the steps
above.

## Project structure

```
nba_props/
├── README.md
├── LICENSE                       # MIT
├── requirements.txt
├── .env.example                  # Template for API keys (.env is gitignored)
├── src/nba_pipeline/             # Library
│   ├── config.py                 # Env/config loading, paths, logging
│   ├── database.py               # SQLite schema and helpers
│   ├── nba_client_br.py          # Basketball-Reference box-score scraper
│   ├── odds_client.py            # SportsGameOdds API client
│   ├── injury_client.py          # Injury report ingestion
│   ├── opponent_context.py       # Opponent defensive adjustments
│   ├── travel.py                 # Travel / rest feature geometry
│   ├── minutes_model.py          # Positional minutes projection
│   ├── stat_model.py             # Per-minute Bayesian rate → distributions
│   ├── points_model.py           # Opponent-adjusted points projection
│   └── playoff_*.py              # Postseason minutes / rate adjustments
├── scripts/                      # Runnable entry points
│   ├── backfill.py               # Historical data pull
│   ├── daily_update.py           # Daily incremental update
│   ├── refresh_odds.py           # Odds-only refresh
│   ├── refresh_injuries.py       # Injury-only refresh
│   ├── build_team_features.py    # Rolling team offense/defense features
│   ├── build_rest_features.py    # Rest / travel features (team_rest table)
│   ├── daily_picks.py            # Candidate bet generation
│   ├── backtest_*.py             # Historical backtests
│   ├── compare_models.py         # Model comparison harness
│   ├── track_edge.py             # Realized-vs-predicted edge tracker
│   └── generate_report.py        # HTML slate reports
├── data/                         # SQLite DB (gitignored)
└── logs/                         # Run logs (gitignored)
```

## Limitations & honest caveats

- **No live edge.** As above, the model does not beat closing lines. Do not bet
  real money on it.
- **Public data only.** No proprietary tracking data, no real-time injury feeds
  faster than the public report.
- **Single-season-depth features.** Opponent and playoff adjustments are
  heuristic, not learned, and the sample behind them is modest.
- **Markets are efficient.** The closing line already incorporates most of what a
  public model can know; this project is a study in *just how hard* that wall is.

## License

[MIT](LICENSE) © 2026 Aidan Chalk
