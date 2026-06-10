#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: data.py
=============================================================================

INPUT FILES:
    - data/universe.xlsx   (via load_universe)
    - Yahoo Finance API    (via fetch_prices - batched download)

OUTPUT FILES:
    - data/report.db       (prices table, via store_prices)

VERSION: 1.0
LAST UPDATED: 2026-06-09
AUTHOR: Arjun Divecha

DESCRIPTION:
    Market data acquisition for the unified report. ONE batched yfinance
    download covers the entire universe plus portfolio holdings (~800
    tickers, 1 year of daily closes). Coverage is validated and the run
    FAILS LOUDLY if more than 10% of requested tickers return nothing -
    no silent NaN-filling, per the project's fail-is-fail policy.

DEPENDENCIES:
    - yfinance, pandas

USAGE:
    from data import load_universe, fetch_prices, store_prices
=============================================================================
"""

import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PATHS, SETTINGS
import db


class DataCoverageError(RuntimeError):
    """Raised when the price fetch returns insufficient coverage."""


def load_universe() -> pd.DataFrame:
    """Load the universe file (fails if missing - run build_universe.py)."""
    path = PATHS["universe"]
    if not path.exists():
        raise FileNotFoundError(
            f"Universe file not found: {path}\nRun: python report/build_universe.py")
    uni = pd.read_excel(path)
    if uni["yf_ticker"].duplicated().any():
        dupes = uni[uni["yf_ticker"].duplicated()]["yf_ticker"].tolist()
        raise ValueError(f"Universe contains duplicate tickers: {dupes}")
    return uni


def fetch_prices(tickers: Iterable[str], period: str = None) -> pd.DataFrame:
    """
    Batched download of adjusted daily closes for all tickers.

    Returns:
        Long-format DataFrame [date, yf_ticker, close, volume].

    Raises:
        DataCoverageError if coverage falls below SETTINGS['min_coverage'].
    """
    tickers = sorted(set(t for t in tickers if t and isinstance(t, str)))
    period = period or SETTINGS["fetch_period"]

    print(f"  Downloading {len(tickers)} tickers, period={period} (batched)...")
    raw = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    if raw is None or raw.empty:
        raise DataCoverageError("yfinance returned an empty frame - no data at all")

    # Normalize to long format
    frames = []
    if isinstance(raw.columns, pd.MultiIndex):
        available = raw.columns.get_level_values(0).unique()
        for t in available:
            sub = raw[t][["Close", "Volume"]].dropna(subset=["Close"])
            if sub.empty:
                continue
            sub = sub.reset_index()
            sub.columns = ["date", "close", "volume"]
            sub["yf_ticker"] = t
            frames.append(sub)
    else:  # single-ticker shape
        sub = raw[["Close", "Volume"]].dropna(subset=["Close"]).reset_index()
        sub.columns = ["date", "close", "volume"]
        sub["yf_ticker"] = tickers[0]
        frames.append(sub)

    if not frames:
        raise DataCoverageError("No ticker returned any usable price rows")

    long_df = pd.concat(frames, ignore_index=True)
    long_df["date"] = pd.to_datetime(long_df["date"]).dt.strftime("%Y-%m-%d")

    # Coverage check - LOUD failure, with the list of what's missing
    got = set(long_df["yf_ticker"].unique())
    missing = sorted(set(tickers) - got)
    coverage = len(got) / len(tickers)
    print(f"  Coverage: {len(got)}/{len(tickers)} tickers ({coverage:.1%})")
    if missing:
        print(f"  Missing ({len(missing)}): {', '.join(missing[:25])}"
              + (" ..." if len(missing) > 25 else ""))
    if coverage < SETTINGS["min_coverage"]:
        raise DataCoverageError(
            f"Price coverage {coverage:.1%} below required "
            f"{SETTINGS['min_coverage']:.0%}. Missing: {missing}")

    return long_df


def store_prices(long_df: pd.DataFrame) -> int:
    """Upsert fetched prices into report.db."""
    n = db.upsert_prices(long_df)
    print(f"  Stored {n} price rows -> {PATHS['db'].name}")
    return n


def latest_trading_date(prices_wide: pd.DataFrame) -> str:
    """Most recent date with data for at least half the universe."""
    counts = prices_wide.notna().sum(axis=1)
    valid = counts[counts >= counts.max() * 0.5]
    return str(valid.index[-1])
