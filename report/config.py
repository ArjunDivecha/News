#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: config.py
=============================================================================

INPUT FILES:
    - /Users/arjundivecha/Dropbox/AAA Backup/A Working/News/.env
      API keys: ANTHROPIC_API_KEY, SCHWAB_APP_KEY, SCHWAB_APP_SECRET.
      IBKR: IBKR_FLEX_TOKEN + IBKR_FLEX_QUERY_ID (primary, no TWS needed);
             fallback still supported via .venv-ibkr312 subprocess.

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
    # Legacy databases (read-only, used by migrate_history.py; archived 2026-06-10)
    "legacy_market_db": ROOT_DIR / "archive" / "Step 4 Report Generation" / "database" / "market_data.db",
    "legacy_portfolio_db": ROOT_DIR / "archive" / "Phase 2 Portfolio Reports" / "database" / "portfolio.db",
    # GMO holdings (static positions with tickers, updated manually)
    "gmo_xlsx": ROOT_DIR / "GMO.xlsx",
    # Legacy holdings file (stale-fallback seed for first run)
    "legacy_client_xlsx": ROOT_DIR / "Client.xlsx",
    # IBKR fetch runs in its own interpreter (ib_insync needs Python 3.12)
    # Kept as FALLBACK only — Flex Web Service (ibkr_flex.py) is primary
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
# ---------------------------------------------------------------------------
# Sub-portfolio labels — maps (broker, account_number) to a friendly name.
# Update these when accounts change. Unlisted accounts show as "broker account".
# ---------------------------------------------------------------------------
ACCOUNT_NAMES = {
    ("Schwab", "12790167"): "Country Value",
    ("Schwab", "28739966"): "Schwab Main",
    ("Schwab", "28739970"): "IRA Mom",
    ("Schwab", "36563696"): "Muni",
    ("Schwab", "36959647"): "Dancing Elephant",
    ("Schwab", "50913476"): "Schwab 50913476",
    ("Schwab", "76705090"): "Country Momentum",
    ("IBKR", "U1399611"): "IBKR Main",
    ("IBKR", "U14983106"): "IBKR Experiment",
    ("IBKR", "U24887919"): "IBKR Trading",
    ("Baupost", "Baupost"): "Baupost (LP)",
}

# ---------------------------------------------------------------------------
# Cash-equivalent symbols — treated as cash in analytics (no return computed)
# ---------------------------------------------------------------------------
CASH_EQUIVALENTS = {"CASH", "SNSXX", "SNAXX"}

# ---------------------------------------------------------------------------
# Tier-3 tag views (multi-label analytics) — the report's benchmark for the
# portfolio-side tag tilts is a blended 60% global equity / 40% long Treasury.
# yf_ticker -> benchmark weight fraction (must sum to 1.0). Their tags are
# pinned in report/tags.py MANUAL_OVERRIDES so the benchmark is exact.
# ---------------------------------------------------------------------------
BENCHMARK = [("ACWI", 0.60), ("TLT", 0.40)]

# ---------------------------------------------------------------------------
# Fund look-through for the asset-allocation report. Multi-asset funds (GMO
# Benchmark-Free etc.) and global-equity funds are distributed into their
# underlying asset classes and equity regions instead of one wholesale bucket.
#   ticker -> {"asof": "YYYY-MM-DD", "source": "<url>",
#              "class": {"Equities":f, "Bonds":f, "Cash":f, "Alternatives":f},
#              "equity_region": {"US":f, "International":f, "EM":f}}
# `class` fractions sum to 1.0 (of the fund); `equity_region` fractions sum to
# 1.0 (of the fund's EQUITY sleeve). Numbers are from published fact sheets with
# the as-of date recorded — refresh them when GMO/the issuers publish new sheets.
# NEVER put a fabricated figure here; leave a fund out and it stays wholesale.
# ---------------------------------------------------------------------------
FUND_LOOKTHROUGH = {
    # GMO Benchmark-Free Allocation Fund — from GMO's Portfolio Composition
    # workbook (user-provided), as of 2026-05-31.
    #   Asset class (Portfolio Allocation tab): Equity strategies 44.1%,
    #   Fixed Income 27.8% (US Treasuries 25.9 + ABS 1.9), Alternative
    #   strategies 28.3% (Equity Dislocation 14.3 + Emerging FX 1.9 + Alt
    #   Allocation 12.1).
    #   Equity regions (Equity-Regions tab): US 14.2, EM 25.0, International
    #   60.8 (Japan 22.9 + Europe ex-UK 20.4 + Other Intl 12.0 + UK 5.5).
    "GBMBX": {
        "asof": "2026-05-31",
        "source": "GMO Benchmark-Free Allocation Fund — Portfolio Composition",
        "class": {"Equities": 0.440, "Bonds": 0.277, "Alternatives": 0.283},
        "equity_region": {"US": 0.142, "International": 0.608, "EM": 0.250},
        "bond_region": {"US": 1.0},   # US Treasury Notes 25.9 + ABS 1.9
    },
    # GMO Equity Dislocation — market-neutral long/short (104% long / 100%
    # short; net regional exposures all within ±3pp => ~0 net equity beta).
    # Its NAV is an ALTERNATIVE strategy, not directional equity, so the whole
    # position is Alternatives (looking through the 100% long book would
    # overstate directional equity by ~30pp of the household). Sourced from the
    # GMO Equity Dislocation Portfolio Composition file.
    "IE00BF199475": {
        "asof": "2026-05-31",
        "source": "GMO Equity Dislocation — Portfolio Composition (market-neutral)",
        "class": {"Alternatives": 1.0},
        "equity_region": {},
    },
    # GMO Emerging Country Debt — EM sovereign/quasi-sovereign USD debt (97%
    # USD, sovereign 72% / quasi 28%). All Fixed Income.
    "GMOQX": {
        "asof": "2026-05-31",
        "source": "GMO Emerging Country Debt — Portfolio Composition",
        "class": {"Bonds": 1.0},
        "equity_region": {},
        "bond_region": {"EM": 1.0},
    },
    # Global-equity funds (100% equity) — regional splits from composition/fact
    # sheets, ex-cash normalized so US+International+EM = 1.0.
    "GCCHX": {  # GMO Climate Change Fund
        "asof": "2026-05-31",
        "source": "GMO Climate Change — Portfolio Composition",
        "class": {"Equities": 1.0},
        "equity_region": {"US": 0.509, "International": 0.306, "EM": 0.185},
    },
    "BCHI": {  # GMO Beyond China ETF — 100% emerging markets (ex-China)
        "asof": "2026-05-31",
        "source": "GMO Beyond China ETF — Portfolio Composition",
        "class": {"Equities": 1.0},
        "equity_region": {"US": 0.0, "International": 0.0, "EM": 1.0},
    },
    "COPX": {  # Global X Copper Miners ETF (miner-heavy: Canada/Australia dev)
        "asof": "2026-05-31",
        "source": "Global X COPX country breakdown",
        "class": {"Equities": 1.0},
        "equity_region": {"US": 0.093, "International": 0.745, "EM": 0.162},
    },
    "ICLN": {  # iShares Global Clean Energy ETF
        "asof": "2026-03-31",
        "source": "iShares ICLN fact sheet",
        "class": {"Equities": 1.0},
        "equity_region": {"US": 0.448, "International": 0.211, "EM": 0.342},
    },
    # Baupost Value Partners LP II — off-broker hedge fund, no daily mark.
    # Policy allocation supplied by the owner (Equity 30% split US 15 / Intl 10 /
    # EM 5 of TOTAL -> 50/33.3/16.7 of the equity sleeve; Bonds/credit 35%;
    # Cash/opportunistic reserve 35%).
    "BAUPOST": {
        "asof": "policy",
        "source": "Owner policy allocation (Baupost Value Partners LP II)",
        "class": {"Equities": 0.30, "Bonds": 0.35, "Cash": 0.35},
        "equity_region": {"US": 0.5, "International": 1 / 3, "EM": 1 / 6},
        "bond_region": {"US": 1.0},   # owner: treat the credit sleeve as US
    },
}

# ---------------------------------------------------------------------------
# Manual off-broker holdings — assets not in any broker/GMO feed, carried at a
# fixed value (no daily price). Included in the HOUSEHOLD total and the asset
# allocation (looked through via FUND_LOOKTHROUGH), NOT in the live "Portfolio"
# section. Update market_value when a new statement arrives.
# ---------------------------------------------------------------------------
MANUAL_HOLDINGS = [
    {"account": "Baupost", "broker": "Baupost", "symbol": "BAUPOST",
     "quantity": float("nan"), "avg_price": float("nan"),
     "market_value": 13_806_000.0, "open_pnl": float("nan"),
     "fetched_at": "", "name": "Baupost Value Partners LP II"},
]
# Extra series pulled purely for the tag views (never held): the benchmark legs
# plus the VIX for the Rule-of-16 noise gate.
# ACWI/TLT = benchmark legs; ^VIX = noise gate; SPY/EFA/EEM/AGG = generic
# asset-class return proxies for looking through no-daily-mark holdings
# (e.g. the Baupost LP): US eq/Intl eq/EM eq/US bonds respectively.
TAG_VIEW_EXTRA_TICKERS = ["ACWI", "TLT", "^VIX", "SPY", "EFA", "EEM", "AGG"]

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
    "model": os.getenv("REPORT_MODEL", "claude-opus-4-8"),
    # LLM backend: Claude CLI uses the local Claude Code subscription path.
    # The direct Anthropic API remains available only as a fallback.
    "llm_backend": os.getenv("REPORT_LLM_BACKEND", "claude_cli"),
    "cli_model": os.getenv("REPORT_CLI_MODEL", "opus"),
    "llm_api_fallback": os.getenv("REPORT_LLM_API_FALLBACK", "1") != "0",
    # max_tokens bounds thinking + visible output TOGETHER on a thinking model.
    # WARNING: 32000 was insufficient with thinking_effort=max — the 2026-06-18
    # report hit EXACTLY 32000 output tokens and truncated mid-table (Risks &
    # Watchlist and Bottom Line were silently dropped). opus-4-8 allows up to
    # 128000; 64000 leaves ample headroom for max-effort thinking plus the full
    # ~9KB report. High max_tokens REQUIRES streaming (see llm.py) so the
    # multi-minute generation does not trip the SDK's non-streaming timeout guard.
    "max_tokens": 64000,
    "thinking_effort": os.getenv("REPORT_THINKING_EFFORT", "high"),  # low/medium/high/max; "high" ~halves runtime vs "max" with negligible quality loss for table interpretation
    "llm_retries": 3,
    "llm_timeout_s": 900,            # streaming generation can run several minutes
    "continuity_days": 5,            # prior executive summaries fed back into prompt
    # Tier-3 tag views: adds portfolio tag-tilt / bridge / concentration and
    # market day-type sections. Purely additive; set REPORT_ENABLE_TAG_VIEWS=0
    # to fall back to the exact prior report.
    "enable_tag_views": os.getenv("REPORT_ENABLE_TAG_VIEWS", "1") != "0",
    "holding_price_aliases": {
        # Schwab reports Vietnam Enterprise Investments as the OTC line VTMEF,
        # which Yahoo no longer prices reliably. Use the London line for
        # returns, then anchor the synthetic VTMEF price series to broker value.
        "VTMEF": "VEIL.L",
    },
    # Brokers
    # IBKR Flex Web Service (primary — no TWS login required, token-based)
    # Set both in .env to skip TWS entirely:
    "ibkr_flex_token_env": "IBKR_FLEX_TOKEN",
    "ibkr_flex_query_id_env": "IBKR_FLEX_QUERY_ID",
    # IB Gateway API port (live=4002, paper=4001; override with IBKR_GATEWAY_PORT)
    "ibkr_port": int(os.getenv("IBKR_GATEWAY_PORT", "4002")),
    "ibkr_client_id": 103,
    "schwab_app_key_env": "SCHWAB_APP_KEY",
    "schwab_app_secret_env": "SCHWAB_APP_SECRET",
    # Schwab auto-auth via Playwright (primary — no interactive browser prompts)
    # Set all three in .env to skip interactive OAuth entirely:
    "schwab_username_env": "SCHWAB_USERNAME",
    "schwab_password_env": "SCHWAB_PASSWORD",
    "schwab_totp_secret_env": "SCHWAB_TOTP_SECRET",
}


def ensure_dirs() -> None:
    """Create output/data directories if missing (safe to call repeatedly)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
