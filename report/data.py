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

VERSION: 1.1
LAST UPDATED: 2026-07-05
AUTHOR: Arjun Divecha

DESCRIPTION:
    Market data acquisition for the unified report. ONE batched yfinance
    download covers the entire universe plus portfolio holdings (~800
    tickers, 1 year of daily closes). Coverage is validated and the run
    FAILS LOUDLY if more than 10% of requested tickers return nothing -
    no silent NaN-filling, per the project's fail-is-fail policy.

    ROOT-CAUSE FIX (2026-07-05): the launchd daily runs (07/01-07/04) failed
    with ~10-17% coverage and per-ticker
    `OperationalError('unable to open database file')`. That SQLite message is
    the classic symptom of FILE-DESCRIPTOR EXHAUSTION (EMFILE): launchd starts
    jobs with a low default soft limit (`launchctl limit maxfiles` -> 256),
    and yfinance's threaded download of ~800 tickers opens far more than 256
    concurrent sockets + per-thread SQLite cache handles, so most tickers fail
    to open the tz-cache DB. Interactive shells inherit a huge FD limit
    (1048576) which is why the pipeline always worked when run by hand.
    `_ensure_fd_limit()` raises the process soft RLIMIT_NOFILE to 8192 (well
    under macOS kern.maxfilesperproc 245760) before any download. A bounded
    retry of the still-missing tickers with exponential backoff is layered on
    top so a transient Yahoo error storm degrades gracefully instead of zeroing
    coverage. The 90% coverage gate is UNCHANGED - the fix makes real data
    arrive, it does not lower the bar or fabricate prices.

DEPENDENCIES:
    - yfinance, pandas

USAGE:
    from data import load_universe, fetch_prices, store_prices
=============================================================================
"""

import resource
import sys
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PATHS, SETTINGS
import db


class DataCoverageError(RuntimeError):
    """Raised when the price fetch returns insufficient coverage."""


# Target soft FD limit for the download. 8192 comfortably covers a threaded
# ~800-ticker pull (sockets + per-thread SQLite cache handles) and stays far
# below macOS kern.maxfilesperproc (245760). launchd's default is 256.
_FD_LIMIT_TARGET = 8192


def _ensure_fd_limit(target: int = _FD_LIMIT_TARGET) -> None:
    """
    Raise this process's soft open-files limit so the threaded yfinance
    download cannot exhaust file descriptors.

    Under launchd the soft limit is 256; yfinance's threaded fetch of the full
    universe opens many more concurrent sockets + SQLite cache handles than
    that, and SQLite reports the resulting EMFILE as
    `OperationalError('unable to open database file')`, silently dropping ~90%
    of tickers. Raising the soft limit (bounded by the hard limit) is the
    root-cause fix; it is a no-op if the limit is already high enough.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft != resource.RLIM_INFINITY and soft >= target:
        return
    new_soft = target
    if hard != resource.RLIM_INFINITY:
        new_soft = min(target, hard)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        print(f"  Raised open-files limit: {soft} -> {new_soft} (hard={hard})")
    except (ValueError, OSError) as e:
        # Fail LOUDLY per fail-is-fail: if we cannot raise the limit, the
        # download will likely under-cover and the coverage gate will catch it,
        # but surface the reason here so it is diagnosable.
        print(f"  !! Could not raise open-files limit from {soft} to {new_soft}: {e}")


def _download_long(tickers: list, period: str) -> pd.DataFrame:
    """
    One batched yfinance download of `tickers`, normalized to long format
    [date, close, volume, yf_ticker]. Returns an empty frame if nothing came
    back (caller handles retry / coverage). Never raises on partial results.
    """
    empty = pd.DataFrame(columns=["date", "close", "volume", "yf_ticker"])
    if not tickers:
        return empty

    # yf.download can itself RAISE (e.g. YFRateLimitError, or a failed ISIN
    # lookup for ISIN-format tickers like GMO's IE00BF199475) rather than just
    # returning a sparse frame. Swallow that here and return empty so the
    # caller's bounded-backoff retry loop can handle it, instead of letting one
    # raise kill the whole run before any retry.
    try:
        raw = yf.download(
            tickers=tickers,
            period=period,
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
        )
    except Exception as e:  # noqa: BLE001 - intentional: convert to retryable
        print(f"  yfinance download raised ({type(e).__name__}: {e}); "
              f"treating as empty for retry")
        return empty
    if raw is None or raw.empty:
        return empty

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
        return pd.DataFrame(columns=["date", "close", "volume", "yf_ticker"])

    long_df = pd.concat(frames, ignore_index=True)
    long_df["date"] = pd.to_datetime(long_df["date"]).dt.strftime("%Y-%m-%d")
    return long_df


def filter_sparse_rows(prices_wide: pd.DataFrame,
                       min_coverage: float = 0.5) -> pd.DataFrame:
    """
    Drop date rows whose ticker coverage is below `min_coverage` of the
    matrix's best-covered day. These are US market-HOLIDAY rows (e.g.
    2026-06-19 Juneteenth: only ~125/808 tickers print - the Canadian/intl
    names that trade through US closures) sitting between two full US
    sessions. Left in place, pct_change computes each US ticker's next-session
    return against the holiday row's NaN -> NaN, poisoning the entire day
    (this is the bug that filled the 2026-06-22 report with n/a).

    Empirically (report.db, 2026): this drops exactly the US holidays at
    ~125 tickers; the thinnest REAL trading day has 790, so 0.5*max never
    drops a genuine session. Dropping (not forward-filling) is correct -
    forward-fill would fabricate 0% holiday returns; dropping makes the "1d"
    return on a post-holiday day correctly span the last two REAL sessions
    (e.g. 2026-06-22 vs 2026-06-18 across the Juneteenth long weekend).
    """
    if prices_wide is None or prices_wide.empty:
        return prices_wide
    counts = prices_wide.notna().sum(axis=1)
    keep = counts >= counts.max() * min_coverage
    return prices_wide[keep]


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

    # ROOT-CAUSE FIX: raise the FD limit before the threaded download so we do
    # not hit launchd's 256-fd cap (which surfaces as SQLite
    # "unable to open database file" and drops ~90% of tickers).
    _ensure_fd_limit()

    print(f"  Downloading {len(tickers)} tickers, period={period} (batched)...")
    long_df = _download_long(tickers, period)

    # Bounded retry with exponential backoff for whatever is still missing, so
    # a transient Yahoo error storm or a partial connection reset retries the
    # gap instead of zeroing coverage. This does NOT relax the coverage gate.
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        got = set(long_df["yf_ticker"].unique()) if not long_df.empty else set()
        missing = sorted(set(tickers) - got)
        if not missing:
            break
        if len(got) / len(tickers) >= SETTINGS["min_coverage"]:
            break  # already over the gate; the remainder are genuinely dead symbols
        backoff = 2 ** attempt
        print(f"  Retry {attempt}/{max_retries}: re-fetching {len(missing)} "
              f"missing tickers after {backoff}s backoff...")
        time.sleep(backoff)
        retry_df = _download_long(missing, period)
        if not retry_df.empty:
            long_df = pd.concat([long_df, retry_df], ignore_index=True)

    if long_df.empty:
        raise DataCoverageError("yfinance returned no usable price rows at all")

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


def apply_holding_price_aliases(long_df: pd.DataFrame,
                                holdings_df: pd.DataFrame,
                                aliases: dict = None) -> pd.DataFrame:
    """
    Add synthetic price series for held symbols that Yahoo does not price.

    The source ticker supplies returns. The synthetic target series is scaled so
    its latest close equals the broker's current market_value / quantity. That
    preserves exposure sizing in compute_portfolio while making daily/YTD
    returns available for the originally held symbol.
    """
    aliases = aliases if aliases is not None else SETTINGS.get("holding_price_aliases", {})
    if not aliases:
        return long_df

    out = long_df.copy()
    held = holdings_df.copy()
    held["symbol"] = held["symbol"].astype(str).str.strip()

    for target, source in aliases.items():
        source_rows = out[out["yf_ticker"] == source].copy()
        target_rows = held[held["symbol"] == target].copy()
        if source_rows.empty or target_rows.empty:
            continue

        qty = pd.to_numeric(target_rows["quantity"], errors="coerce").sum()
        mv = pd.to_numeric(target_rows["market_value"], errors="coerce").sum()
        if not qty or pd.isna(qty) or pd.isna(mv):
            print(f"  Price alias skipped for {target}: missing broker anchor")
            continue

        latest_close = source_rows.sort_values("date")["close"].dropna().iloc[-1]
        if not latest_close or pd.isna(latest_close):
            print(f"  Price alias skipped for {target}: missing {source} close")
            continue

        broker_unit_price = mv / qty
        scale = broker_unit_price / latest_close
        alias_rows = source_rows.copy()
        alias_rows["yf_ticker"] = target
        alias_rows["close"] = alias_rows["close"] * scale
        out = out[out["yf_ticker"] != target]
        out = pd.concat([out, alias_rows], ignore_index=True)
        print(f"  Price alias: {target} <- {source} "
              f"(scaled to broker unit price {broker_unit_price:.4f})")

    return out


def store_prices(long_df: pd.DataFrame) -> int:
    """Upsert fetched prices into report.db."""
    n = db.upsert_prices(long_df)
    print(f"  Stored {n} price rows -> {PATHS['db'].name}")
    return n


def latest_trading_date(prices_wide: pd.DataFrame) -> str:
    """Most recent REAL trading date (sparse US-holiday rows excluded)."""
    clean = filter_sparse_rows(prices_wide)
    counts = clean.notna().sum(axis=1)
    valid = counts[counts >= counts.max() * 0.5]
    return str(valid.index[-1])
