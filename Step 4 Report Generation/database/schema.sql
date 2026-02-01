-- =============================================================================
-- MARKET DATA DATABASE SCHEMA
-- =============================================================================
-- 
-- Purpose: Store static asset data, daily prices, and generated reports
-- for the News from Data report generation pipeline.
--
-- Tables:
--   assets          - Static asset data with 3-tier taxonomy and betas
--   daily_prices    - Daily price snapshots (accumulates history)
--   intraday_prices - Intraday snapshots (kept 7 days)
--   category_stats  - Pre-computed category aggregates
--   reports         - Generated report archive
--
-- Version: 1.0.0
-- Created: 2026-01-30
-- =============================================================================

-- Static asset data (from Final 1000 Asset Master List)
-- Note: 'source' column kept internally but never shown in reports
CREATE TABLE IF NOT EXISTS assets (
    ticker TEXT PRIMARY KEY,
    name TEXT,
    description TEXT,
    
    -- 3-Tier Taxonomy
    tier1 TEXT,                    -- Equities, Fixed Income, Commodities, etc.
    tier2 TEXT,                    -- Thematic/Factor, Global Indices, etc.
    tier3_tags TEXT,               -- JSON array of tags
    
    -- Internal tracking (not shown in reports)
    source TEXT,                   -- 'ETF', 'Bloomberg', 'Goldman', 'Thematic ETF'
    
    -- 18 Beta exposures
    beta_spx REAL,                 -- S&P 500
    beta_russell2000 REAL,         -- Small cap
    beta_nasdaq100 REAL,           -- Tech/Growth
    beta_russell_value REAL,       -- Value
    beta_russell_growth REAL,      -- Growth
    beta_eafe REAL,                -- International developed
    beta_em REAL,                  -- Emerging markets
    beta_hy_credit REAL,           -- High yield credit
    beta_treasuries REAL,          -- Government bonds
    beta_tips REAL,                -- Inflation-linked
    beta_commodity REAL,           -- Broad commodities
    beta_agriculture REAL,         -- Agriculture
    beta_crypto REAL,              -- Cryptocurrency
    beta_reit_us REAL,             -- US REITs
    beta_reit_global REAL,         -- Global REITs
    
    -- Performance metrics (from classification)
    sharpe_1y REAL,
    sharpe_3y REAL,
    return_12m REAL,
    return_36m REAL,
    vol_12m REAL,
    correlation_spx REAL,
    
    -- Selection metadata
    selection_score REAL,
    quality_score REAL,
    thematic_rarity REAL,
    
    -- Timestamps
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

-- Daily price snapshots (accumulates history)
CREATE TABLE IF NOT EXISTS daily_prices (
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    
    -- Price data
    price REAL,
    price_open REAL,
    price_high REAL,
    price_low REAL,
    
    -- Returns (percentages)
    return_1d REAL,
    return_1w REAL,
    return_5d REAL,                -- 5-day return (new)
    return_1m REAL,
    return_ytd REAL,
    return_1y REAL,
    
    -- Technical indicators
    rsi_14 REAL,                   -- 14-day RSI (new)
    
    -- Volume and volatility
    volume INTEGER,
    volatility_30d REAL,
    volatility_60d REAL,
    volatility_240d REAL,          -- 240-day volatility (new)
    
    -- Derived metrics (calculated during refresh)
    z_score_1d REAL,               -- 1-day return / 60-day vol
    percentile_60d REAL,           -- Percentile rank vs last 60 days
    
    -- Beta-predicted return (sum of beta_i * factor_return_i)
    beta_predicted_return REAL,
    alpha_1d REAL,                 -- Actual - Beta predicted
    
    PRIMARY KEY (date, ticker),
    FOREIGN KEY (ticker) REFERENCES assets(ticker)
);

-- Intraday price snapshots (for flash reports, kept 7 days)
CREATE TABLE IF NOT EXISTS intraday_prices (
    timestamp TEXT NOT NULL,       -- ISO format: YYYY-MM-DDTHH:MM:SS
    ticker TEXT NOT NULL,
    
    price REAL,
    return_from_open REAL,         -- % change from day's open
    return_from_prev_close REAL,   -- % change from previous close
    
    PRIMARY KEY (timestamp, ticker),
    FOREIGN KEY (ticker) REFERENCES assets(ticker)
);

-- Pre-computed category aggregates (for fast report generation)
CREATE TABLE IF NOT EXISTS category_stats (
    date TEXT NOT NULL,
    category_type TEXT NOT NULL,   -- 'tier1', 'tier2', 'tier3_tag'
    category_value TEXT NOT NULL,
    
    -- Aggregate statistics
    count INTEGER,
    avg_return REAL,
    median_return REAL,
    std_return REAL,
    min_return REAL,
    max_return REAL,
    
    -- Best/worst performers (internal reference)
    best_ticker TEXT,
    best_return REAL,
    worst_ticker TEXT,
    worst_return REAL,
    
    -- Historical context
    percentile_60d REAL,           -- Where this day ranks in last 60 days
    streak_days INTEGER,           -- Consecutive days positive/negative
    streak_direction TEXT,         -- 'positive', 'negative', 'neutral'
    
    PRIMARY KEY (date, category_type, category_value)
);

-- Factor returns (for beta attribution)
CREATE TABLE IF NOT EXISTS factor_returns (
    date TEXT NOT NULL,
    factor_name TEXT NOT NULL,     -- 'SPX', 'Russell2000', 'Value', 'Growth', etc.
    
    return_1d REAL,
    return_1w REAL,
    return_1m REAL,
    return_ytd REAL,
    
    PRIMARY KEY (date, factor_name)
);

-- Rolling 60-day correlations (for factor analysis and regime detection)
CREATE TABLE IF NOT EXISTS asset_correlations (
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    
    -- Rolling 60-day correlations to each factor
    corr_spx REAL,
    corr_russell2000 REAL,
    corr_nasdaq100 REAL,
    corr_value REAL,
    corr_growth REAL,
    corr_eafe REAL,
    corr_em REAL,
    corr_hy_credit REAL,
    corr_treasuries REAL,
    corr_tips REAL,
    corr_commodities REAL,
    corr_agriculture REAL,
    corr_crypto REAL,
    corr_reit_us REAL,
    corr_reit_global REAL,
    
    -- Derived metrics
    r_squared_best REAL,           -- R-squared of best single factor
    best_factor TEXT,              -- Name of best correlated factor
    regime_change INTEGER DEFAULT 0, -- 1 if correlation regime changed (>0.3 shift in 5 days)
    
    PRIMARY KEY (date, ticker),
    FOREIGN KEY (ticker) REFERENCES assets(ticker)
);

-- Generated reports archive
CREATE TABLE IF NOT EXISTS reports (
    report_id TEXT PRIMARY KEY,
    report_type TEXT NOT NULL,     -- 'daily', 'flash'
    report_date TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    
    -- Content
    content_md TEXT,               -- Full markdown content
    content_summary TEXT,          -- Flash headlines / key points
    
    -- File references
    pdf_path TEXT,
    
    -- Metadata (JSON)
    metadata TEXT,                 -- Model used, token count, generation time, etc.
    
    -- LLM tracking
    model_name TEXT,
    prompt_version TEXT,
    tokens_input INTEGER,
    tokens_output INTEGER,
    generation_time_ms INTEGER
);

-- =============================================================================
-- INDEXES for performance
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_daily_prices_date ON daily_prices(date);
CREATE INDEX IF NOT EXISTS idx_daily_prices_ticker ON daily_prices(ticker);
CREATE INDEX IF NOT EXISTS idx_daily_prices_date_ticker ON daily_prices(date, ticker);

CREATE INDEX IF NOT EXISTS idx_intraday_timestamp ON intraday_prices(timestamp);
CREATE INDEX IF NOT EXISTS idx_intraday_ticker ON intraday_prices(ticker);

CREATE INDEX IF NOT EXISTS idx_category_stats_date ON category_stats(date);
CREATE INDEX IF NOT EXISTS idx_category_stats_type ON category_stats(category_type);

CREATE INDEX IF NOT EXISTS idx_reports_date ON reports(report_date);
CREATE INDEX IF NOT EXISTS idx_reports_type ON reports(report_type);

CREATE INDEX IF NOT EXISTS idx_assets_tier1 ON assets(tier1);
CREATE INDEX IF NOT EXISTS idx_assets_tier2 ON assets(tier2);
CREATE INDEX IF NOT EXISTS idx_assets_source ON assets(source);

CREATE INDEX IF NOT EXISTS idx_correlations_date ON asset_correlations(date);
CREATE INDEX IF NOT EXISTS idx_correlations_ticker ON asset_correlations(ticker);
CREATE INDEX IF NOT EXISTS idx_correlations_best_factor ON asset_correlations(best_factor);
CREATE INDEX IF NOT EXISTS idx_correlations_regime_change ON asset_correlations(regime_change);

-- =============================================================================
-- VIEWS for common queries
-- =============================================================================

-- Latest daily data for all assets
CREATE VIEW IF NOT EXISTS v_latest_daily AS
SELECT dp.*, a.name, a.tier1, a.tier2, a.tier3_tags
FROM daily_prices dp
JOIN assets a ON dp.ticker = a.ticker
WHERE dp.date = (SELECT MAX(date) FROM daily_prices);

-- Tier-1 summary for latest date
CREATE VIEW IF NOT EXISTS v_tier1_summary AS
SELECT 
    category_value as tier1,
    count,
    avg_return,
    median_return,
    std_return,
    percentile_60d,
    streak_days,
    streak_direction
FROM category_stats
WHERE category_type = 'tier1'
  AND date = (SELECT MAX(date) FROM category_stats WHERE category_type = 'tier1');

-- Tier-2 summary for latest date
CREATE VIEW IF NOT EXISTS v_tier2_summary AS
SELECT 
    category_value as tier2,
    count,
    avg_return,
    median_return,
    std_return,
    percentile_60d,
    streak_days,
    streak_direction
FROM category_stats
WHERE category_type = 'tier2'
  AND date = (SELECT MAX(date) FROM category_stats WHERE category_type = 'tier2')
ORDER BY avg_return DESC;

-- =============================================================================
-- END OF SCHEMA
-- =============================================================================
