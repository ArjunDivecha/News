#!/usr/bin/env python3
"""
=============================================================================
DATABASE UTILITIES
=============================================================================

PURPOSE:
Common database operations for the report generation pipeline.

USAGE:
    from utils.db import get_db, get_assets, get_latest_prices, save_daily_prices
=============================================================================
"""

import sqlite3
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

# Database path
DB_PATH = Path(__file__).parent.parent.parent / "database" / "market_data.db"
NEWS_ROOT = Path(__file__).parent.parent.parent.parent

MEME_SOCIAL_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meme_social_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_ts TEXT NOT NULL UNIQUE,
    snapshot_date TEXT NOT NULL,
    source_file TEXT,
    search_queries TEXT,
    total_found INTEGER,
    ingested_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS meme_social_assets (
    snapshot_id INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    name TEXT,
    mentions INTEGER,
    source_domains TEXT,
    volume_spike REAL,
    price_change_1d REAL,
    price_change_5d REAL,
    current_volume INTEGER,
    avg_volume INTEGER,
    market_cap_b REAL,
    reasons TEXT,
    sample_context TEXT,
    meme_score REAL,
    in_universe INTEGER DEFAULT 0,
    universe_ticker TEXT,
    tier1 TEXT,
    tier2 TEXT,
    tier3_tags TEXT,
    PRIMARY KEY (snapshot_id, symbol)
);

CREATE INDEX IF NOT EXISTS idx_meme_social_snapshots_date
    ON meme_social_snapshots(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_meme_social_assets_snapshot
    ON meme_social_assets(snapshot_id);
CREATE INDEX IF NOT EXISTS idx_meme_social_assets_universe
    ON meme_social_assets(in_universe, tier1, tier2);
"""


def get_db() -> sqlite3.Connection:
    """Get database connection with row factory for dict-like access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_meme_social_schema(conn: Optional[sqlite3.Connection] = None) -> None:
    """Ensure meme/social tables exist on the active database."""
    owns_conn = conn is None
    conn = conn or get_db()
    conn.executescript(MEME_SOCIAL_SCHEMA_SQL)
    conn.commit()
    if owns_conn:
        conn.close()


def _json_dumps(value: Any) -> str:
    """Serialize a Python object to compact JSON for SQLite storage."""
    return json.dumps(value or [])


def _json_loads_list(value: Any) -> List[Any]:
    """Parse a JSON-encoded list field from SQLite/JSON input."""
    if value in (None, "", []):
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except (TypeError, json.JSONDecodeError):
        return []


def _to_optional_float(value: Any) -> Optional[float]:
    """Coerce a scalar to float when possible, otherwise return None."""
    if value is None or pd.isna(value):
        return None
    return float(value)


def _clean_domain(value: Any) -> str:
    """Normalize a source domain for prompt-friendly display."""
    if value in (None, ""):
        return ""
    domain = str(value).strip().lower()
    if domain.startswith("https://"):
        domain = domain[8:]
    elif domain.startswith("http://"):
        domain = domain[7:]
    domain = domain.split("/", 1)[0]
    if domain.startswith("www."):
        domain = domain[4:]
    parts = [part for part in domain.split(".") if part]
    if len(parts) >= 2:
        domain = ".".join(parts[-2:])
    return domain


def _clean_context_snippet(value: Any, limit: int = 180) -> str:
    """Compress noisy search-result fragments into short evidence snippets."""
    if value in (None, ""):
        return ""
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _series_mean(series: pd.Series) -> Optional[float]:
    """Return a float mean for a numeric series, or None when empty."""
    numeric = pd.to_numeric(series, errors='coerce').dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


def _business_day_gap(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """
    Return signed business-day gap from start to end.

    Positive means end is later than start, negative means earlier.
    """
    start = pd.to_datetime(start).normalize()
    end = pd.to_datetime(end).normalize()
    if start == end:
        return 0
    if end > start:
        return max(len(pd.bdate_range(start, end)) - 1, 0)
    return -max(len(pd.bdate_range(end, start)) - 1, 0)


def resolve_meme_social_snapshot_path(snapshot_path: Optional[str] = None) -> Optional[Path]:
    """Find the latest meme/social JSON snapshot in the repo."""
    if snapshot_path:
        path = Path(snapshot_path).expanduser()
        return path if path.exists() else None

    candidates = [
        NEWS_ROOT / "search_based_meme_stocks.json",
        NEWS_ROOT / "Step 2 Data Processing - Final1000" / "search_based_meme_stocks.json",
        NEWS_ROOT / "Step 3 Data Analysis" / "search_based_meme_stocks.json",
    ]

    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None

    return max(existing, key=lambda path: path.stat().st_mtime)


def ingest_meme_social_snapshot(snapshot_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load a MemeFinder JSON snapshot into SQLite and enrich it against assets.

    Returns:
        Dict with ingestion status and counts.
    """
    path = resolve_meme_social_snapshot_path(snapshot_path)
    if path is None:
        return {
            'status': 'missing_file',
            'message': 'No meme/social snapshot file found',
        }

    payload = json.loads(path.read_text())
    snapshot_ts = payload.get('timestamp')
    if not snapshot_ts:
        raise ValueError(f"Snapshot file is missing timestamp: {path}")

    snapshot_dt = pd.to_datetime(snapshot_ts)
    snapshot_date = snapshot_dt.strftime('%Y-%m-%d')
    search_queries = payload.get('search_queries', [])
    meme_stocks = payload.get('meme_stocks', [])
    total_found = int(payload.get('total_found', len(meme_stocks)))

    conn = get_db()
    ensure_meme_social_schema(conn)
    cursor = conn.cursor()

    assets_df = pd.read_sql_query(
        """
        SELECT ticker, name, tier1, tier2, tier3_tags
        FROM assets
        """,
        conn,
    )
    asset_lookup = {
        str(row['ticker']).upper(): row
        for _, row in assets_df.iterrows()
        if pd.notna(row['ticker'])
    }

    cursor.execute(
        """
        INSERT INTO meme_social_snapshots
            (snapshot_ts, snapshot_date, source_file, search_queries, total_found, ingested_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(snapshot_ts) DO UPDATE SET
            snapshot_date = excluded.snapshot_date,
            source_file = excluded.source_file,
            search_queries = excluded.search_queries,
            total_found = excluded.total_found,
            ingested_at = excluded.ingested_at
        """,
        (
            snapshot_dt.isoformat(),
            snapshot_date,
            str(path),
            _json_dumps(search_queries),
            total_found,
            datetime.now().isoformat(),
        ),
    )
    cursor.execute(
        "SELECT snapshot_id FROM meme_social_snapshots WHERE snapshot_ts = ?",
        (snapshot_dt.isoformat(),),
    )
    snapshot_id = cursor.fetchone()[0]

    cursor.execute(
        "DELETE FROM meme_social_assets WHERE snapshot_id = ?",
        (snapshot_id,),
    )

    matched_count = 0
    for stock in meme_stocks:
        symbol = str(stock.get('symbol', '')).upper().strip()
        if not symbol:
            continue

        matched = asset_lookup.get(symbol)
        in_universe = 1 if matched is not None else 0
        matched_count += in_universe

        cursor.execute(
            """
            INSERT OR REPLACE INTO meme_social_assets (
                snapshot_id, symbol, name, mentions, source_domains,
                volume_spike, price_change_1d, price_change_5d,
                current_volume, avg_volume, market_cap_b,
                reasons, sample_context, meme_score,
                in_universe, universe_ticker, tier1, tier2, tier3_tags
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot_id,
                symbol,
                stock.get('name'),
                stock.get('mentions'),
                _json_dumps(stock.get('sources', [])),
                stock.get('volume_spike'),
                stock.get('price_change_1d'),
                stock.get('price_change_5d'),
                stock.get('current_volume'),
                stock.get('avg_volume'),
                stock.get('market_cap_b'),
                _json_dumps(stock.get('reasons', [])),
                _json_dumps(stock.get('sample_context', [])),
                stock.get('meme_score'),
                in_universe,
                matched['ticker'] if matched is not None else None,
                matched['tier1'] if matched is not None else None,
                matched['tier2'] if matched is not None else None,
                matched['tier3_tags'] if matched is not None else None,
            ),
        )

    conn.commit()
    conn.close()

    return {
        'status': 'ingested',
        'snapshot_id': snapshot_id,
        'snapshot_ts': snapshot_dt.isoformat(),
        'snapshot_date': snapshot_date,
        'source_file': str(path),
        'total_found': total_found,
        'rows_loaded': len(meme_stocks),
        'matched_count': matched_count,
        'unmatched_count': max(len(meme_stocks) - matched_count, 0),
    }


def get_meme_social_context(report_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Return the latest meme/social snapshot context.

    This intentionally uses the newest available snapshot regardless of the
    market-data date so current meme/social flow can still be reported when
    market data is lagging.
    """
    conn = get_db()
    ensure_meme_social_schema(conn)

    report_ts = pd.to_datetime(report_date).normalize() if report_date else pd.Timestamp.now().normalize()
    snapshot_df = pd.read_sql_query(
        """
        SELECT *
        FROM meme_social_snapshots
        ORDER BY snapshot_date DESC, snapshot_ts DESC
        LIMIT 1
        """,
        conn,
    )

    if snapshot_df.empty:
        conn.close()
        return {
            'status': 'missing',
            'summary': 'No meme/social snapshot has been ingested.',
        }

    snapshot = snapshot_df.iloc[0].to_dict()
    assets_df = pd.read_sql_query(
        """
        SELECT *
        FROM meme_social_assets
        WHERE snapshot_id = ?
        ORDER BY meme_score DESC, mentions DESC
        """,
        conn,
        params=[snapshot['snapshot_id']],
    )
    conn.close()

    if assets_df.empty:
        return {
            'status': 'empty',
            'snapshot_ts': snapshot['snapshot_ts'],
            'snapshot_date': snapshot['snapshot_date'],
            'summary': 'Meme/social snapshot exists but contains no meme-stock rows.',
        }

    snapshot_date = pd.to_datetime(snapshot['snapshot_date']).normalize()
    today_ts = pd.Timestamp.now().normalize()
    business_days_old = max(_business_day_gap(snapshot_date, today_ts), 0)
    calendar_days_old = max((today_ts - snapshot_date).days, 0)
    market_gap_business_days = _business_day_gap(report_ts, snapshot_date)
    market_gap_calendar_days = int((snapshot_date - report_ts).days)

    if business_days_old == 0:
        freshness = 'fresh'
    elif business_days_old == 1:
        freshness = 'recent'
    elif business_days_old == 2:
        freshness = 'aging'
    else:
        freshness = 'stale'

    if market_gap_business_days > 0:
        market_alignment = 'newer_than_market'
    elif market_gap_business_days < 0:
        market_alignment = 'older_than_market'
    else:
        market_alignment = 'aligned_with_market'

    matched_df = assets_df[assets_df['in_universe'] == 1].copy()
    unmatched_count = int(len(assets_df) - len(matched_df))

    tier1_breakdown = []
    if not matched_df.empty:
        tier1_grouped = (
            matched_df.groupby('tier1', dropna=True)
            .agg(
                count=('symbol', 'size'),
                mentions=('mentions', 'sum'),
                avg_1d=('price_change_1d', 'mean'),
                avg_5d=('price_change_5d', 'mean'),
                avg_volume_spike=('volume_spike', 'mean'),
                avg_score=('meme_score', 'mean'),
            )
            .reset_index()
            .sort_values(['mentions', 'count'], ascending=False)
        )
        tier1_breakdown = tier1_grouped.to_dict('records')

    tier2_breakdown = []
    if not matched_df.empty:
        tier2_grouped = (
            matched_df.groupby('tier2', dropna=True)
            .agg(
                count=('symbol', 'size'),
                mentions=('mentions', 'sum'),
                avg_1d=('price_change_1d', 'mean'),
                avg_5d=('price_change_5d', 'mean'),
            )
            .reset_index()
            .sort_values(['mentions', 'count'], ascending=False)
        )
        tier2_breakdown = tier2_grouped.head(5).to_dict('records')

    reason_flags = {
        'volume_spike': 0,
        'one_day_move': 0,
        'five_day_move': 0,
    }
    for _, row in assets_df.iterrows():
        for reason in _json_loads_list(row.get('reasons')):
            reason_text = str(reason).lower()
            if 'volume' in reason_text:
                reason_flags['volume_spike'] += 1
            elif '(1d)' in reason_text:
                reason_flags['one_day_move'] += 1
            elif '(5d)' in reason_text:
                reason_flags['five_day_move'] += 1

    top_tags = []
    if not matched_df.empty:
        tag_counts: Dict[str, int] = {}
        for raw_tags in matched_df['tier3_tags']:
            for tag in _json_loads_list(raw_tags):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        top_tags = [
            {'tag': tag, 'count': count}
            for tag, count in sorted(tag_counts.items(), key=lambda item: (-item[1], item[0]))[:8]
        ]

    source_counts: Dict[str, int] = {}
    for raw_sources in assets_df['source_domains']:
        for domain in _json_loads_list(raw_sources):
            label = _clean_domain(domain)
            if not label:
                continue
            source_counts[label] = source_counts.get(label, 0) + 1

    source_summary = [
        {'domain': domain, 'count': count}
        for domain, count in sorted(source_counts.items(), key=lambda item: (-item[1], item[0]))[:8]
    ]

    top_signals: List[Dict[str, Any]] = []
    for _, row in assets_df.head(6).iterrows():
        sources = [_clean_domain(domain) for domain in _json_loads_list(row.get('source_domains'))]
        reasons = [str(reason) for reason in _json_loads_list(row.get('reasons')) if str(reason).strip()]
        contexts = [
            snippet
            for snippet in (
                _clean_context_snippet(value)
                for value in _json_loads_list(row.get('sample_context'))
            )
            if snippet
        ]

        top_signals.append(
            {
                'symbol': str(row.get('symbol') or '').upper(),
                'name': row.get('name'),
                'mentions': int(row.get('mentions') or 0),
                'source_count': len(sources),
                'sources': sources[:4],
                'price_change_1d': _to_optional_float(row.get('price_change_1d')),
                'price_change_5d': _to_optional_float(row.get('price_change_5d')),
                'volume_spike': _to_optional_float(row.get('volume_spike')),
                'meme_score': _to_optional_float(row.get('meme_score')),
                'reasons': reasons[:3],
                'sample_context': contexts[:2],
                'in_universe': bool(row.get('in_universe')),
                'tier1': row.get('tier1'),
                'tier2': row.get('tier2'),
            }
        )

    return {
        'status': 'ok',
        'snapshot_id': int(snapshot['snapshot_id']),
        'snapshot_ts': snapshot['snapshot_ts'],
        'snapshot_date': snapshot['snapshot_date'],
        'search_queries': _json_loads_list(snapshot.get('search_queries')),
        'source_file': snapshot.get('source_file'),
        'total_found': int(snapshot.get('total_found') or len(assets_df)),
        'total_rows': int(len(assets_df)),
        'freshness': freshness,
        'business_days_old': int(business_days_old),
        'calendar_days_old': int(calendar_days_old),
        'as_of_date': today_ts.strftime('%Y-%m-%d'),
        'market_alignment': market_alignment,
        'market_gap_business_days': int(market_gap_business_days),
        'market_gap_calendar_days': int(market_gap_calendar_days),
        'report_date': report_ts.strftime('%Y-%m-%d'),
        'matched_count': int(len(matched_df)),
        'unmatched_count': unmatched_count,
        'total_mentions': int(pd.to_numeric(assets_df['mentions'], errors='coerce').fillna(0).sum()),
        'avg_volume_spike': _series_mean(assets_df['volume_spike']),
        'avg_price_change_1d': _series_mean(assets_df['price_change_1d']),
        'avg_price_change_5d': _series_mean(assets_df['price_change_5d']),
        'tier1_breakdown': tier1_breakdown,
        'tier2_breakdown': tier2_breakdown,
        'top_tags': top_tags,
        'top_signals': top_signals,
        'source_summary': source_summary,
        'reason_flags': reason_flags,
    }


def get_assets(tier1: Optional[str] = None, tier2: Optional[str] = None) -> pd.DataFrame:
    """
    Get assets from database with optional filtering.
    
    Args:
        tier1: Filter by Tier-1 category
        tier2: Filter by Tier-2 category
        
    Returns:
        DataFrame with asset data
    """
    conn = get_db()
    
    query = "SELECT * FROM assets WHERE 1=1"
    params = []
    
    if tier1:
        query += " AND tier1 = ?"
        params.append(tier1)
    if tier2:
        query += " AND tier2 = ?"
        params.append(tier2)
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    return df


def get_tickers() -> List[str]:
    """Get list of all tickers in database."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT ticker FROM assets ORDER BY ticker")
    tickers = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tickers


def get_latest_prices(date: Optional[str] = None) -> pd.DataFrame:
    """
    Get latest daily prices, optionally for a specific date.
    
    Args:
        date: Date string (YYYY-MM-DD). If None, gets most recent date.
        
    Returns:
        DataFrame with price data joined with asset info
    """
    conn = get_db()
    
    if date is None:
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(date) FROM daily_prices")
        result = cursor.fetchone()
        date = result[0] if result[0] else None
    
    if date is None:
        conn.close()
        return pd.DataFrame()
    
    query = """
        SELECT 
            dp.*,
            a.name, a.tier1, a.tier2, a.tier3_tags, a.source,
            a.beta_spx, a.beta_russell2000, a.beta_nasdaq100,
            a.beta_eafe, a.beta_em
        FROM daily_prices dp
        JOIN assets a ON dp.ticker = a.ticker
        WHERE dp.date = ?
    """
    
    df = pd.read_sql_query(query, conn, params=[date])
    conn.close()
    
    return df


def save_daily_prices(df: pd.DataFrame, date: str) -> int:
    """
    Save daily price data to database.
    
    Args:
        df: DataFrame with columns matching daily_prices schema
        date: Date string (YYYY-MM-DD)
        
    Returns:
        Number of rows inserted/updated
    """
    conn = get_db()
    cursor = conn.cursor()
    
    # Ensure date column
    df = df.copy()
    df['date'] = date
    
    # Required columns
    required = ['date', 'ticker']
    if not all(col in df.columns for col in required):
        raise ValueError(f"DataFrame must have columns: {required}")
    
    # Get valid columns from schema
    cursor.execute("PRAGMA table_info(daily_prices)")
    valid_columns = [row[1] for row in cursor.fetchall()]
    
    # Filter to valid columns
    cols = [c for c in df.columns if c in valid_columns]
    df = df[cols]
    
    # Insert or replace
    placeholders = ','.join(['?' for _ in cols])
    cols_str = ','.join(cols)
    
    count = 0
    for _, row in df.iterrows():
        values = [None if pd.isna(v) else v for v in row.values]
        cursor.execute(
            f"INSERT OR REPLACE INTO daily_prices ({cols_str}) VALUES ({placeholders})",
            values
        )
        count += 1
    
    conn.commit()
    conn.close()
    
    return count


def save_category_stats(stats: List[Dict], date: str) -> int:
    """
    Save category statistics to database.
    
    Args:
        stats: List of dicts with category stats
        date: Date string (YYYY-MM-DD)
        
    Returns:
        Number of rows inserted
    """
    conn = get_db()
    cursor = conn.cursor()
    
    count = 0
    for stat in stats:
        cursor.execute("""
            INSERT OR REPLACE INTO category_stats 
            (date, category_type, category_value, count, avg_return, median_return,
             std_return, min_return, max_return, best_ticker, best_return,
             worst_ticker, worst_return, percentile_60d, streak_days, streak_direction)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            date,
            stat.get('category_type'),
            stat.get('category_value'),
            stat.get('count'),
            stat.get('avg_return'),
            stat.get('median_return'),
            stat.get('std_return'),
            stat.get('min_return'),
            stat.get('max_return'),
            stat.get('best_ticker'),
            stat.get('best_return'),
            stat.get('worst_ticker'),
            stat.get('worst_return'),
            stat.get('percentile_60d'),
            stat.get('streak_days'),
            stat.get('streak_direction')
        ))
        count += 1
    
    conn.commit()
    conn.close()
    
    return count


def save_factor_returns(factors: Dict[str, float], date: str) -> int:
    """
    Save factor returns to database.
    
    Args:
        factors: Dict mapping factor name to return
        date: Date string (YYYY-MM-DD)
        
    Returns:
        Number of rows inserted
    """
    conn = get_db()
    cursor = conn.cursor()
    
    count = 0
    for factor_name, return_1d in factors.items():
        cursor.execute("""
            INSERT OR REPLACE INTO factor_returns (date, factor_name, return_1d)
            VALUES (?, ?, ?)
        """, (date, factor_name, return_1d))
        count += 1
    
    conn.commit()
    conn.close()
    
    return count


def save_report(report_id: str, report_type: str, report_date: str,
                content_md: str, model_name: str, **kwargs) -> None:
    """
    Save generated report to database.
    
    Args:
        report_id: Unique identifier
        report_type: 'daily' or 'flash'
        report_date: Date of the report
        content_md: Markdown content
        model_name: LLM model used
        **kwargs: Additional fields (pdf_path, tokens_input, etc.)
    """
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT OR REPLACE INTO reports 
        (report_id, report_type, report_date, generated_at, content_md,
         model_name, pdf_path, tokens_input, tokens_output, generation_time_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        report_id,
        report_type,
        report_date,
        datetime.now().isoformat(),
        content_md,
        model_name,
        kwargs.get('pdf_path'),
        kwargs.get('tokens_input'),
        kwargs.get('tokens_output'),
        kwargs.get('generation_time_ms')
    ))
    
    conn.commit()
    conn.close()


def get_historical_stats(category_type: str, category_value: str, 
                         days: int = 60) -> pd.DataFrame:
    """
    Get historical category stats for pattern analysis.
    
    Args:
        category_type: 'tier1', 'tier2', or 'tier3_tag'
        category_value: The category value to look up
        days: Number of days of history
        
    Returns:
        DataFrame with historical stats
    """
    conn = get_db()
    
    query = """
        SELECT * FROM category_stats
        WHERE category_type = ? AND category_value = ?
        ORDER BY date DESC
        LIMIT ?
    """
    
    df = pd.read_sql_query(query, conn, params=[category_type, category_value, days])
    conn.close()
    
    return df


def get_tier1_distribution() -> Dict[str, int]:
    """Get count of assets by Tier-1 category."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT tier1, COUNT(*) FROM assets GROUP BY tier1")
    result = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()
    return result


def get_tier2_distribution() -> Dict[str, int]:
    """Get count of assets by Tier-2 category."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT tier2, COUNT(*) FROM assets GROUP BY tier2 ORDER BY COUNT(*) DESC")
    result = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()
    return result
