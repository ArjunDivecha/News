#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: names.py
=============================================================================

INPUT FILES:
    - data/report.db   (security_names cache table, via db.py)
    - Yahoo Finance     (longName / shortName lookups for uncached symbols)

OUTPUT FILES:
    - data/report.db   (security_names cache is populated as a side effect)

VERSION: 1.0
LAST UPDATED: 2026-06-20
AUTHOR: Arjun Divecha

DESCRIPTION:
    Resolves ticker symbols to full, human-readable security names so the
    report can refer to holdings by NAME, not ticker (e.g. EWY ->
    "iShares MSCI South Korea ETF", INTC -> "Intel Corporation").

    The stored universe names are polluted with Goldman-basket codenames
    ("GS Korea L PB H Profit"), so Yahoo Finance's longName is the source of
    truth here. Results are cached per ticker in report.db, so Yahoo is hit
    once per symbol, not every run.

    Symbols Yahoo cannot resolve (CUSIP cash lines, some OTC/GMO funds) map
    to themselves - i.e. the report keeps the raw ticker/ID for those (the
    approved fallback). A cache row with name='' records a tried-and-failed
    lookup so we don't refetch it every run.

DEPENDENCIES:
    - yfinance (only for uncached symbols)

USAGE:
    from names import resolve_names
    name_map = resolve_names(["EWY", "INTC", "12464X101"])
=============================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import db


def _fetch_yahoo_names(symbols: list) -> dict:
    """Best-effort Yahoo longName/shortName lookup. Failures -> '' (not fatal:
    a missing name is allowed to fall back to the ticker, per spec)."""
    import yfinance as yf
    out = {}
    for s in symbols:
        name = ""
        try:
            info = yf.Ticker(s).get_info()
            name = (info.get("longName") or info.get("shortName") or "").strip()
        except Exception as e:
            print(f"    name lookup failed for {s}: {e}")
            name = ""
        out[s] = name
    return out


def resolve_names(symbols, fetch: bool = True) -> dict:
    """
    Return {symbol: full_name}. Cached Yahoo lookups; unresolved symbols map to
    themselves (the raw ticker/ID).

    Args:
        symbols: iterable of ticker symbols.
        fetch:   if False, only use the cache (no network) - uncached symbols
                 fall back to themselves. Used by tests.
    """
    syms = sorted({str(s).strip() for s in symbols if str(s).strip()})
    if not syms:
        return {}

    cached = db.get_cached_names(syms)             # {sym: name or ''}
    missing = [s for s in syms if s not in cached]

    if fetch and missing:
        print(f"  Resolving {len(missing)} new security name(s) via Yahoo...")
        fetched = _fetch_yahoo_names(missing)
        db.upsert_names(fetched)                    # cache successes AND ''-failures
        cached.update(fetched)

    # name='' (tried+failed) or uncached(no-fetch) -> keep the ticker/ID
    return {s: (cached.get(s) or s) for s in syms}
