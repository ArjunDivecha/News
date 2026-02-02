-- =============================================================================
-- PHASE 2: PORTFOLIO REPORTS DATABASE SCHEMA
-- =============================================================================
-- 
-- Purpose: Store portfolio data, holdings, daily snapshots, and generated reports
-- for personalized portfolio report generation.
--
-- This is a SEPARATE database from Phase 1's market_data.db
--
-- Version: 1.0.0
-- Created: 2026-02-01
-- =============================================================================

-- Portfolio master table
CREATE TABLE IF NOT EXISTS portfolios (
    portfolio_id TEXT PRIMARY KEY,
    portfolio_name TEXT NOT NULL,
    client_name TEXT,
    base_currency TEXT DEFAULT 'USD',
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    is_active INTEGER DEFAULT 1
);

-- Portfolio holdings (updated when new file is ingested)
CREATE TABLE IF NOT EXISTS portfolio_holdings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id TEXT NOT NULL,
    symbol TEXT NOT NULL,              -- Plain ticker from portfolio file (AAPL, EWZ)
    position_type TEXT NOT NULL,       -- 'LONG' or 'SHORT'
    
    -- From portfolio file (actual client data)
    quantity REAL NOT NULL,            -- Quantity from file (positive for LONG, negative for SHORT)
    market_value REAL,                 -- Market Value from file (USD)
    avg_price REAL,                    -- Average Price (cost basis) from file
    open_pnl REAL,                     -- Open Profit/Loss from file
    
    -- Resolved identifiers
    yf_ticker TEXT,                    -- Yahoo Finance ticker (may differ from symbol)
    security_name TEXT,                -- Full name from yfinance
    security_type TEXT,                -- 'EQUITY', 'ETF', 'MUTUALFUND', 'ADR', etc.
    
    -- yfinance metadata
    yf_sector TEXT,                    -- Raw yfinance sector (for stocks)
    yf_industry TEXT,                  -- Raw yfinance industry (for stocks)
    yf_category TEXT,                  -- ETF category (for ETFs)
    country TEXT,
    currency TEXT,
    
    -- Phase 1 taxonomy (mapped or looked up)
    tier1 TEXT,
    tier2 TEXT,
    tier3_tags TEXT,                   -- JSON array
    
    -- Classification tracking
    final1000_ticker TEXT,             -- If matched to Final 1000
    classification_source TEXT,        -- 'final1000', 'yfinance_mapped', 'haiku', 'failed'
    resolution_status TEXT DEFAULT 'pending',  -- 'resolved', 'failed', 'pending'
    resolution_error TEXT,             -- Error message if resolution failed
    
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(portfolio_id),
    UNIQUE (portfolio_id, symbol, position_type)
);

-- Daily portfolio snapshot
CREATE TABLE IF NOT EXISTS portfolio_daily (
    date TEXT NOT NULL,
    portfolio_id TEXT NOT NULL,
    holding_id INTEGER NOT NULL,       -- Reference to portfolio_holdings.id
    symbol TEXT NOT NULL,
    position_type TEXT NOT NULL,
    
    -- Position data
    quantity REAL,
    price REAL,                        -- Current price from yfinance
    market_value_usd REAL,             -- Current market value (qty × price or from file)
    weight REAL,                       -- Position weight (0-1), derived from market values
    
    -- Cost basis (from portfolio file)
    avg_price REAL,                    -- Cost basis per share
    cost_basis REAL,                   -- Total cost (qty × avg_price)
    
    -- P&L
    open_pnl REAL,                     -- Unrealized P&L from file or computed
    open_pnl_pct REAL,                 -- Unrealized P&L as % of cost basis
    daily_pnl REAL,                    -- Day's P&L (market_value change)
    
    -- Returns
    return_1d REAL,                    -- Daily return %
    return_ytd REAL,                   -- YTD return %
    contribution_1d REAL,              -- Contribution to portfolio return (basis points)
    
    fetch_status TEXT,                 -- 'success', 'stale', 'failed'
    
    PRIMARY KEY (date, portfolio_id, holding_id),
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(portfolio_id),
    FOREIGN KEY (holding_id) REFERENCES portfolio_holdings(id)
);

-- Portfolio aggregates by dimension
CREATE TABLE IF NOT EXISTS portfolio_aggregates (
    date TEXT NOT NULL,
    portfolio_id TEXT NOT NULL,
    dimension_type TEXT NOT NULL,      -- 'sector', 'country', 'tier1', 'tier2', 'tier3_tag'
    dimension_value TEXT NOT NULL,
    
    holding_count INTEGER,
    long_count INTEGER,
    short_count INTEGER,
    total_weight REAL,
    long_weight REAL,
    short_weight REAL,
    total_value_usd REAL,
    weighted_return_1d REAL,
    contribution_1d REAL,
    
    PRIMARY KEY (date, portfolio_id, dimension_type, dimension_value),
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(portfolio_id)
);

-- Portfolio-level daily summary
CREATE TABLE IF NOT EXISTS portfolio_summary (
    date TEXT NOT NULL,
    portfolio_id TEXT NOT NULL,
    
    -- Portfolio totals
    total_market_value REAL,
    total_long_value REAL,
    total_short_value REAL,
    net_exposure REAL,                 -- Long - Short
    gross_exposure REAL,               -- Long + |Short|
    holding_count INTEGER,
    long_count INTEGER,
    short_count INTEGER,
    
    -- Returns
    portfolio_return_1d REAL,          -- Portfolio return (weighted)
    portfolio_return_ytd REAL,
    
    -- P&L
    total_open_pnl REAL,
    daily_pnl REAL,
    
    -- Top contributors/detractors (JSON)
    top_contributors TEXT,             -- JSON array of {symbol, contribution, return}
    top_detractors TEXT,               -- JSON array of {symbol, contribution, return}
    
    PRIMARY KEY (date, portfolio_id),
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(portfolio_id)
);

-- Portfolio reports archive
CREATE TABLE IF NOT EXISTS portfolio_reports (
    report_id TEXT PRIMARY KEY,
    portfolio_id TEXT NOT NULL,
    report_date TEXT NOT NULL,
    report_type TEXT DEFAULT 'daily',
    generated_at TEXT NOT NULL,
    
    content_md TEXT,
    pdf_path TEXT,
    
    model_name TEXT,
    tokens_input INTEGER,
    tokens_output INTEGER,
    generation_time_ms INTEGER,
    
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(portfolio_id)
);

-- =============================================================================
-- INDEXES for performance
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_holdings_portfolio ON portfolio_holdings(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_holdings_symbol ON portfolio_holdings(symbol);
CREATE INDEX IF NOT EXISTS idx_holdings_tier1 ON portfolio_holdings(tier1);
CREATE INDEX IF NOT EXISTS idx_holdings_resolution ON portfolio_holdings(resolution_status);

CREATE INDEX IF NOT EXISTS idx_daily_date ON portfolio_daily(date);
CREATE INDEX IF NOT EXISTS idx_daily_portfolio ON portfolio_daily(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_daily_date_portfolio ON portfolio_daily(date, portfolio_id);

CREATE INDEX IF NOT EXISTS idx_aggregates_date ON portfolio_aggregates(date);
CREATE INDEX IF NOT EXISTS idx_aggregates_portfolio ON portfolio_aggregates(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_aggregates_dimension ON portfolio_aggregates(dimension_type);

CREATE INDEX IF NOT EXISTS idx_summary_date ON portfolio_summary(date);
CREATE INDEX IF NOT EXISTS idx_summary_portfolio ON portfolio_summary(portfolio_id);

CREATE INDEX IF NOT EXISTS idx_reports_date ON portfolio_reports(report_date);
CREATE INDEX IF NOT EXISTS idx_reports_portfolio ON portfolio_reports(portfolio_id);

-- =============================================================================
-- VIEWS for common queries
-- =============================================================================

-- Latest holdings for a portfolio
CREATE VIEW IF NOT EXISTS v_latest_holdings AS
SELECT h.*, p.portfolio_name, p.client_name
FROM portfolio_holdings h
JOIN portfolios p ON h.portfolio_id = p.portfolio_id
WHERE h.resolution_status = 'resolved';

-- Portfolio exposure summary
CREATE VIEW IF NOT EXISTS v_portfolio_exposure AS
SELECT 
    portfolio_id,
    tier1,
    COUNT(*) as holding_count,
    SUM(CASE WHEN position_type = 'LONG' THEN 1 ELSE 0 END) as long_count,
    SUM(CASE WHEN position_type = 'SHORT' THEN 1 ELSE 0 END) as short_count,
    SUM(market_value) as total_value,
    SUM(CASE WHEN position_type = 'LONG' THEN market_value ELSE 0 END) as long_value,
    SUM(CASE WHEN position_type = 'SHORT' THEN market_value ELSE 0 END) as short_value
FROM portfolio_holdings
WHERE resolution_status = 'resolved'
GROUP BY portfolio_id, tier1;

-- =============================================================================
-- END OF SCHEMA
-- =============================================================================
