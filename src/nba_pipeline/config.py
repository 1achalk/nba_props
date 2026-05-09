"""Configuration: loads environment variables and defines paths/constants."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = parent of src/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load .env from project root
load_dotenv(PROJECT_ROOT / ".env")

# Paths
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
DB_PATH = PROJECT_ROOT / os.getenv("DB_PATH", "data/nba.db")

# Make sure dirs exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# API keys
SPORTSGAMEODDS_API_KEY = os.getenv("SPORTSGAMEODDS_API_KEY", "").strip()

# NBA API rate limit: be polite. ~1 req per 0.6s is generally safe.
NBA_API_DELAY_SECONDS = 0.7
NBA_API_TIMEOUT_SECONDS = 90
NBA_API_MAX_RETRIES = 3

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def setup_logging(name: str = "nba_pipeline") -> logging.Logger:
    """Configure logging once. Returns a logger you can use."""
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger

    logger.setLevel(LOG_LEVEL)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    return logger
