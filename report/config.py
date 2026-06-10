#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: config.py
=============================================================================

INPUT FILES:
    - /Users/arjundivecha/Dropbox/AAA Backup/A Working/News/.env
      API keys (ANTHROPIC_API_KEY, SCHWAB_APP_KEY, SCHWAB_APP_SECRET).

OUTPUT FILES:
    (none - this is a configuration module)

VERSION: 1.0
LAST UPDATED: 2026-06-09
AUTHOR: Arjun Divecha

DESCRIPTION:
    Single source of configuration for the unified daily report system.
    Every other module imports paths and constants from here, so there is
    exactly one place to change a path, a window length, or a model name.

DEPENDENCIES:
    - python-dotenv

USAGE:
    from config import PATHS, FACTORS, SETTINGS
=============================================================================
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths - everything is relative to the repo root (parent of report/)
# ---------------------------------------------------------------------------
REPORT_DIR = Path(__file__).resolve().parent          # .../News/report
ROOT_DIR = REPORT_DIR.parent                          # .../News
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs" / "unified"
TESTS_DIR = ROOT_DIR / "tests"

load_dotenv(ROOT_DIR / ".env")

PATHS = {
    "universe": DATA_DIR / "universe.xlsx",
    "holdings": DATA_DIR / "holdings.xlsx",            # written by holdings.py each run
    "db": DATA_DIR / "report.db",
    "output_dir": OUTPUT_DIR,
    "system_prompt": REPORT_DIR / "prompts" / "system.md",
    # Legacy databases (read-only, used by migrate_history.py)
    "legacy_market_db": ROOT_DIR / "Step 4 Report Generation" / "database" / "market_data.db",
    "legacy_portfolio_db": ROOT_DIR / "Phase 2 Portfolio Reports" / "database" / "portfolio.db",
    # Legacy holdings file (stale-fallback seed for first run)
    "legacy_client_xlsx": ROOT_DIR / "Client.xlsx",
    # IBKR fetch runs in its own interpreter (ib_insync needs Python 3.12)
    "ibkr_python": ROOT_DIR / ".venv-ibkr312" / "bin" / "python3",
}

# ---------------------------------------------------------------------------
# Factor definitions - the 15 factor ETFs (validated in the ETF migration)
# factor_name -> yfinance ticker
# ---------------------------------------------------------------------------
FACTORS = {
    "SPX": "SPY",
    "Russell2000": "IWM",
    "Nasdaq100": "QQQ",
    "Value": "IWD",
    "Growth": "IWF",
    "EAFE": "EFA",
    "EM": "EEM",
    "HY_Credit": "HYG",
    "Treasuries": "BNDX",
    "TIPS": "TIP",
    "Commodities": "BCI",
    "Agriculture": "DBA",
    "Crypto": "IBIT",
    "REIT_US": "VNQ",
    "REIT_Global": "VNQI",
}

# ---------------------------------------------------------------------------
# Analytics & pipeline settings
# ---------------------------------------------------------------------------
SETTINGS = {
    # Price fetch
    "fetch_period": "1y",            # always refetch 1y; idempotent upsert self-heals gaps
    "min_coverage": 0.90,            # FAIL the run if < 90% of tickers return prices
    # Analytics windows (trading days)
    "vol_window": 60,
    "beta_window": 60,
    "percentile_window": 60,
    "min_beta_obs": 30,              # minimum observations to trust a beta
    # Report
    "model": os.getenv("REPORT_MODEL", "claude-opus-4-6"),
    "max_tokens": 16000,
    "thinking_budget": 8000,         # extended thinking budget for the report call
    "llm_retries": 3,
    "llm_timeout_s": 600,
    "continuity_days": 5,            # prior executive summaries fed back into prompt
    # Brokers
    "ibkr_port": 7496,
    "ibkr_client_id": 103,
    "schwab_app_key_env": "SCHWAB_APP_KEY",
    "schwab_app_secret_env": "SCHWAB_APP_SECRET",
}


def ensure_dirs() -> None:
    """Create output/data directories if missing (safe to call repeatedly)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
