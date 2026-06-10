#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: db.py
=============================================================================

INPUT FILES:
    - /Users/arjundivecha/Dropbox/AAA Backup/A Working/News/data/report.db
      (created automatically if missing)

OUTPUT FILES:
    - /Users/arjundivecha/Dropbox/AAA Backup/A Working/News/data/report.db
      Single SQLite database for the unified report system. Tables:
        assets              - universe snapshot (refreshed from universe.xlsx)
        prices              - adjusted close history per yf_ticker
        portfolio_snapshots - per-position daily record (weights, contributions)
        portfolio_summary   - one row per day of portfolio-level results
        reports             - archive of generated reports (continuity source)

VERSION: 1.0
LAST UPDATED: 2026-06-09
AUTHOR: Arjun Divecha

DESCRIPTION:
    All database access for the unified report system. Connections are
    context-managed, WAL mode is enabled, foreign keys are enforced, and
    bulk writes use executemany. Every writer is idempotent (upsert), so
    re-running a day is always safe.

DEPENDENCIES:
    - pandas

USAGE:
    from db import connect, init_schema, upsert_prices, ...
=============================================================================
"""

import sqlite3
from contextlib import contextmanager
from typing import Iterator, Optional

import pandas as pd

from config import PATHS

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
    yf_ticker     TEXT PRIMARY KEY,
    name          TEXT,
    description   TEXT,
    tier1         TEXT,
    tier2         TEXT,
    tags          TEXT,
    source        TEXT,
    tracking_score INTEGER,
    is_factor     INTEGER DEFAULT 0,
    factor_name   TEXT,
    proxied_tickers TEXT,
    updated_at    TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS prices (
    date      TEXT NOT NULL,
    yf_ticker TEXT NOT NULL,
    close     REAL,
    volume    REAL,
    PRIMARY KEY (date, yf_ticker)
);
CREATE INDEX IF NOT EXISTS idx_prices_ticker ON prices(yf_ticker);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    date          TEXT NOT NULL,
    account       TEXT NOT NULL DEFAULT '',
    symbol        TEXT NOT NULL,
    position_type TEXT NOT NULL,         -- LONG / SHORT
    quantity      REAL,
    price         REAL,
    market_value  REAL,                  -- signed (negative for shorts)
    weight        REAL,                  -- signed, vs gross exposure
    return_1d     REAL,                  -- %
    return_ytd    REAL,                  -- %
    contribution_bps REAL,               -- signed weight x return, in bps
    open_pnl      REAL,
    holdings_stale INTEGER DEFAULT 0,    -- 1 when run used stale holdings
    PRIMARY KEY (date, account, symbol, position_type)
);

CREATE TABLE IF NOT EXISTS portfolio_summary (
    date            TEXT PRIMARY KEY,
    total_value     REAL,
    gross_exposure  REAL,
    net_exposure    REAL,
    long_value      REAL,
    short_value     REAL,
    n_positions     INTEGER,
    return_1d       REAL,                -- %
    return_ytd      REAL,                -- %
    expected_return_1d REAL,             -- % (factor-implied)
    alpha_1d        REAL,                -- % (actual - expected)
    total_open_pnl  REAL,
    holdings_stale  INTEGER DEFAULT 0,
    holdings_as_of  TEXT
);

CREATE TABLE IF NOT EXISTS reports (
    date              TEXT PRIMARY KEY,
    generated_at      TEXT DEFAULT (datetime('now')),
    executive_summary TEXT,
    content_md        TEXT,
    model             TEXT,
    tokens_input      INTEGER,
    tokens_output     INTEGER,
    generation_time_ms INTEGER,
    pdf_path          TEXT
);
"""


@contextmanager
def connect(db_path=None) -> Iterator[sqlite3.Connection]:
    """Context-managed connection with WAL + foreign keys + busy timeout."""
    conn = sqlite3.connect(str(db_path or PATHS["db"]), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_schema() -> None:
    """Create all tables if they do not exist."""
    with connect() as conn:
        conn.executescript(SCHEMA)


def sync_assets(universe: pd.DataFrame) -> int:
    """Refresh the assets table from the universe DataFrame."""
    cols = ["yf_ticker", "name", "description", "tier1", "tier2", "tags",
            "source", "tracking_score", "is_factor", "factor_name", "proxied_tickers"]
    records = universe[cols].where(pd.notna(universe[cols]), None).values.tolist()
    with connect() as conn:
        conn.executemany("""
            INSERT INTO assets (yf_ticker, name, description, tier1, tier2, tags,
                                source, tracking_score, is_factor, factor_name,
                                proxied_tickers, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(yf_ticker) DO UPDATE SET
                name=excluded.name, description=excluded.description,
                tier1=excluded.tier1, tier2=excluded.tier2, tags=excluded.tags,
                source=excluded.source, tracking_score=excluded.tracking_score,
                is_factor=excluded.is_factor, factor_name=excluded.factor_name,
                proxied_tickers=excluded.proxied_tickers, updated_at=datetime('now')
        """, records)
    return len(records)


def upsert_prices(prices: pd.DataFrame) -> int:
    """
    Bulk-upsert price history.

    Args:
        prices: long-format DataFrame with columns [date, yf_ticker, close, volume]
    """
    df = prices.dropna(subset=["close"])
    records = df[["date", "yf_ticker", "close", "volume"]].values.tolist()
    with connect() as conn:
        conn.executemany("""
            INSERT INTO prices (date, yf_ticker, close, volume)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(date, yf_ticker) DO UPDATE SET
                close=excluded.close, volume=excluded.volume
        """, records)
    return len(records)


def load_prices(min_date: Optional[str] = None) -> pd.DataFrame:
    """Load price history as a wide DataFrame (index=date, columns=yf_ticker)."""
    query = "SELECT date, yf_ticker, close FROM prices"
    params: tuple = ()
    if min_date:
        query += " WHERE date >= ?"
        params = (min_date,)
    with connect() as conn:
        df = pd.read_sql_query(query, conn, params=params)
    if df.empty:
        return pd.DataFrame()
    return df.pivot(index="date", columns="yf_ticker", values="close").sort_index()


def save_portfolio_snapshot(date: str, positions: pd.DataFrame,
                            summary: dict) -> None:
    """Save per-position snapshot and the portfolio summary for one date."""
    pos_cols = ["account", "symbol", "position_type", "quantity", "price",
                "market_value", "weight", "return_1d", "return_ytd",
                "contribution_bps", "open_pnl", "holdings_stale"]
    records = [
        [date] + [None if pd.isna(v) else v for v in row]
        for row in positions[pos_cols].itertuples(index=False, name=None)
    ]
    with connect() as conn:
        conn.execute("DELETE FROM portfolio_snapshots WHERE date = ?", (date,))
        conn.executemany(f"""
            INSERT INTO portfolio_snapshots (date, {', '.join(pos_cols)})
            VALUES ({', '.join(['?'] * (len(pos_cols) + 1))})
        """, records)
        conn.execute("""
            INSERT INTO portfolio_summary
                (date, total_value, gross_exposure, net_exposure, long_value,
                 short_value, n_positions, return_1d, return_ytd,
                 expected_return_1d, alpha_1d, total_open_pnl,
                 holdings_stale, holdings_as_of)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                total_value=excluded.total_value,
                gross_exposure=excluded.gross_exposure,
                net_exposure=excluded.net_exposure,
                long_value=excluded.long_value,
                short_value=excluded.short_value,
                n_positions=excluded.n_positions,
                return_1d=excluded.return_1d,
                return_ytd=excluded.return_ytd,
                expected_return_1d=excluded.expected_return_1d,
                alpha_1d=excluded.alpha_1d,
                total_open_pnl=excluded.total_open_pnl,
                holdings_stale=excluded.holdings_stale,
                holdings_as_of=excluded.holdings_as_of
        """, (
            date,
            summary.get("total_value"), summary.get("gross_exposure"),
            summary.get("net_exposure"), summary.get("long_value"),
            summary.get("short_value"), summary.get("n_positions"),
            summary.get("return_1d"), summary.get("return_ytd"),
            summary.get("expected_return_1d"), summary.get("alpha_1d"),
            summary.get("total_open_pnl"),
            summary.get("holdings_stale", 0), summary.get("holdings_as_of"),
        ))


def get_portfolio_history(n_days: int = 30) -> pd.DataFrame:
    """Recent portfolio summary rows, oldest first."""
    with connect() as conn:
        df = pd.read_sql_query("""
            SELECT * FROM (
                SELECT * FROM portfolio_summary ORDER BY date DESC LIMIT ?
            ) ORDER BY date ASC
        """, conn, params=(n_days,))
    return df


def save_report(date: str, executive_summary: str, content_md: str,
                model: str, tokens_input: int, tokens_output: int,
                generation_time_ms: int, pdf_path: str) -> None:
    """Archive a generated report (idempotent per date)."""
    with connect() as conn:
        conn.execute("""
            INSERT INTO reports (date, executive_summary, content_md, model,
                                 tokens_input, tokens_output, generation_time_ms,
                                 pdf_path, generated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(date) DO UPDATE SET
                executive_summary=excluded.executive_summary,
                content_md=excluded.content_md, model=excluded.model,
                tokens_input=excluded.tokens_input,
                tokens_output=excluded.tokens_output,
                generation_time_ms=excluded.generation_time_ms,
                pdf_path=excluded.pdf_path, generated_at=datetime('now')
        """, (date, executive_summary, content_md, model, tokens_input,
              tokens_output, generation_time_ms, pdf_path))


def get_prior_summaries(before_date: str, n: int = 5) -> pd.DataFrame:
    """Executive summaries of the n reports before a date (for continuity)."""
    with connect() as conn:
        df = pd.read_sql_query("""
            SELECT date, executive_summary FROM reports
            WHERE date < ? AND executive_summary IS NOT NULL
            ORDER BY date DESC LIMIT ?
        """, conn, params=(before_date, n))
    return df.sort_values("date")
