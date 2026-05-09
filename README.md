# NBA Player Props Pipeline

A data pipeline for modeling NBA player props. Pulls game data, box scores, shot data, and odds — stores everything in a local SQLite database you can query and feed into models.

## What this is (and isn't)

**This is:** a foundation. It collects and organizes data so you can build models on top.

**This is not:** the model itself. Once data is flowing, we build the projection model, calibration pipeline, and bet selection layer as a next step.

---

## First-time setup (do this once)

### 1. Install Homebrew (Mac package manager)

Open **Terminal** (Cmd+Space, type "Terminal"). Paste this and hit Enter:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

It will ask for your Mac password. After it finishes, it prints "Next steps" — run those two `eval` commands it tells you to. This adds Homebrew to your shell.

### 2. Install Python

```bash
brew install python@3.12
```

Verify:

```bash
python3 --version
```

You should see `Python 3.12.x`.

### 3. Open the project in VSCode

- Open VSCode
- File → Open Folder → navigate to `/Users/achalk/Documents/nba_props`
- When prompted "Do you trust the authors", click Yes

### 4. Install Python extensions in VSCode

Extensions sidebar (Cmd+Shift+X), install:
- **Python** (by Microsoft)
- **Pylance** (usually auto-installs)

### 5. Create the virtual environment

Open VSCode's terminal: **Terminal → New Terminal** (or `` Ctrl+` ``). Make sure you're in the project folder (the prompt should show `nba_props`). Run:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

You'll see `(.venv)` appear in your prompt — that means the virtual environment is active.

### 6. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs `nba_api`, `pandas`, `requests`, and everything else. Takes a minute or two.

### 7. Set up your API key

Copy the example config to your real config:

```bash
cp .env.example .env
```

Open `.env` in VSCode and paste your SportsGameOdds API key. **The .env file is gitignored — never commit it.**

> ⚠️ If you previously shared your key anywhere (chat, email, screenshot), regenerate it at sportsgameodds.com first.

### 8. Tell VSCode about the virtual environment

- Press Cmd+Shift+P
- Type "Python: Select Interpreter"
- Choose the one with `.venv` in the path

Now VSCode knows to use this project's Python.

---

## Running the pipeline

Every time you open a new terminal in this project, activate the venv first:

```bash
source .venv/bin/activate
```

### One-time backfill (pull historical data)

```bash
python -m scripts.backfill --seasons 2022-23 2023-24 2024-25 2025-26
```

This takes 30-60 minutes (rate-limited to be polite to NBA's API). Run once, you're done.

### Daily updates

```bash
python -m scripts.daily_update
```

Pulls yesterday's completed games + today's schedule + injury report. Takes a few minutes.

### Pre-game odds refresh

```bash
python -m scripts.refresh_odds
```

Pulls current player prop lines. Run this 2-3 times before tipoff to track line movement.

---

## Automating updates (cron)

To run automatically: open Terminal and type `crontab -e`. Add these lines (adjust paths):

```cron
# Daily update at 3am
0 3 * * * cd /Users/achalk/Documents/nba_props && /Users/achalk/Documents/nba_props/.venv/bin/python -m scripts.daily_update >> logs/daily.log 2>&1

# Odds refresh at 10am, 2pm, 6pm
0 10,14,18 * * * cd /Users/achalk/Documents/nba_props && /Users/achalk/Documents/nba_props/.venv/bin/python -m scripts.refresh_odds >> logs/odds.log 2>&1
```

Save and exit (in vim: `:wq`). The first time, Mac will ask for "Full Disk Access" for cron — grant it in System Settings → Privacy.

---

## Project structure

```
nba_props/
├── README.md              # This file
├── requirements.txt       # Python packages
├── .env.example          # Template for API keys
├── .env                  # Your real API keys (gitignored)
├── .gitignore
├── src/
│   └── nba_pipeline/
│       ├── __init__.py
│       ├── nba_client.py      # Rate-limited nba_api wrapper
│       ├── odds_client.py     # SportsGameOdds API client
│       ├── database.py        # SQLite schema and helpers
│       ├── travel.py          # Travel/rest feature builder
│       └── config.py          # Loads .env, paths, constants
├── scripts/
│   ├── __init__.py
│   ├── backfill.py            # Historical data pull
│   ├── daily_update.py        # Daily incremental update
│   └── refresh_odds.py        # Odds-only refresh
├── data/
│   └── nba.db                 # SQLite database (created on first run)
└── logs/                      # Log output from cron jobs
```

---

## What's next (after data is flowing)

1. **Minutes projection model** — Bayesian blend of season minutes, recent games, matchup adjustments
2. **Component-level prop projections** — model points = sum of (shot type × make rate × point value), assists/rebounds/threes similarly, output full distributions
3. **Calibration pipeline** — Brier score, reliability diagrams, log loss tracking
4. **Bet selection** — fractional Kelly sizing, CLV tracking, line shopping across books
5. **Daily report** — pre-game writeup of model picks with confidence bands

These come after the foundation works. Don't skip ahead.
