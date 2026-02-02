# Phase 2: Personalized Portfolio Reports
## Product Requirements Document

**Version**: 1.2
**Date**: 2026-02-01
**Author**: Arjun Divecha
**Status**: Planning

---

## Executive Summary

Phase 2 is a **completely independent product** that generates personalized daily market reports through the lens of a client's actual portfolio. The system ingests client holdings (10-500 positions including LONG and SHORT positions, individual stocks and ETFs), enriches them with market data, classifies them, and generates reports that answer "What happened in markets that affects MY portfolio?"

### Relationship to Phase 1

| Aspect | Phase 1 | Phase 2 |
|--------|---------|---------|
| **Product** | Daily Market Wrap | Personalized Portfolio Reports |
| **Audience** | General market overview | Specific client portfolio |
| **Data Source** | Bloomberg (970 curated assets) | Yahoo Finance (client holdings) |
| **Independence** | Standalone | Standalone |
| **Database** | `Step 4 Report Generation/database/market_data.db` | `Phase 2 Portfolio Reports/database/portfolio.db` |
| **Can Run Without Other** | Yes | Yes |

**Phase 1 and Phase 2 are separate products that run independently.**

Phase 2 reuses the **taxonomy concepts** (Tier 1/2/3 classification) from Phase 1 for consistency, but does NOT depend on Phase 1's database, scripts, or runtime.

---

## Background

### Phase 1 Accomplishments
- Curated 970 assets (Final 1000 Asset Master List) across 7 Tier-1 categories
- Built 3-tier taxonomy classification system using Claude Haiku 4.5
- Trained fine-tuned Llama 3.1 8B model for cost-free classification
- Created daily market wrap report generation using Claude Opus 4.5
- Established SQLite database with assets, daily_prices, factor_returns, category_stats tables
- Implemented PDF generation via WeasyPrint/PrinceXML

### Phase 2 Motivation
Portfolio managers need reports that answer: "What happened in the markets **that affects MY portfolio**?" rather than generic market commentary. Phase 2 personalizes the daily wrap by:
1. Showing portfolio-level performance attribution
2. Highlighting holdings in sectors/regions with unusual moves
3. Connecting market themes to specific portfolio exposures
4. Identifying which holdings drove returns (positive and negative)

---

## Requirements

### Functional Requirements

#### FR-1: Portfolio Ingestion
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1.1 | Accept Excel/CSV files with columns: Symbol, Long Quantity | P0 |
| FR-1.2 | Support additional columns: Market Value, Average Price, Long Open Profit/Loss | P0 |
| FR-1.3 | Support plain tickers (AAPL, MSFT) without exchange suffixes | P0 |
| FR-1.4 | Handle both individual stocks AND ETFs (global equities, US, Europe, Asia, EM) | P0 |
| FR-1.5 | Support 10-500 holdings per portfolio | P0 |
| FR-1.6 | Store portfolios in database with audit trail | P1 |
| FR-1.7 | Use provided Market Value for weight calculation (more accurate than qty × price) | P0 |
| FR-1.8 | Store cost basis (Average Price) and P&L for reporting | P1 |
| FR-1.9 | **Handle SHORT positions** (negative quantity, negative market value) | P0 |
| FR-1.10 | **Handle same symbol appearing multiple times** (e.g., LONG and SHORT positions) | P0 |
| FR-1.11 | Handle micro-cap/penny stocks with very low prices (e.g., $0.0027) | P1 |
| FR-1.12 | Handle large position counts (e.g., 2,000,000 shares) | P0 |

#### FR-2: Ticker Resolution
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-2.1 | Resolve plain tickers to Yahoo Finance identifiers | P0 |
| FR-2.2 | Handle international tickers with exchange suffixes (.L, .DE, .T, .HK) | P0 |
| FR-2.3 | Check Final 1000 database for existing matches | P0 |
| FR-2.4 | Flag unresolved tickers for manual review (DO NOT skip silently) | P0 |
| FR-2.5 | Cache resolution results to avoid repeated API calls | P1 |

#### FR-3: Classification (Reusing Phase 1 Taxonomy)
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-3.1 | Use existing Tier 1/2/3 classification for holdings in Final 1000 | P0 |
| FR-3.2 | Map individual stocks via yfinance sector → Phase 1 taxonomy (deterministic) | P0 |
| FR-3.3 | Classify ETFs/funds not in Final 1000 using Claude Haiku (same prompt as Phase 1) | P0 |
| FR-3.4 | Store classification_source field (final1000, yfinance_mapped, haiku) | P0 |
| FR-3.5 | Use EXACT same taxonomy from `unified_asset_classifier.py` | P0 |

#### FR-4: Daily Price Enrichment
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-4.1 | Fetch daily prices from Yahoo Finance for all holdings | P0 |
| FR-4.2 | Compute daily returns (1-day, YTD) | P0 |
| FR-4.3 | Use provided Market Value for weight calculation (preferred) or compute from qty × price | P0 |
| FR-4.4 | Compute contribution to return (weight × return in basis points) | P0 |
| FR-4.5 | Track daily P&L using cost basis from Average Price | P1 |
| FR-4.6 | Compute unrealized P&L percentage (current value vs cost basis) | P1 |
| FR-4.7 | Handle missing prices gracefully (use previous day, mark as stale) | P1 |

#### FR-5: Portfolio Analytics
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-5.1 | Aggregate holdings by sector (using Phase 1 Tier-3 tags) | P0 |
| FR-5.2 | Aggregate holdings by region (using Phase 1 Tier-3 tags) | P0 |
| FR-5.3 | Aggregate holdings by Tier-1 category | P0 |
| FR-5.4 | Identify top 10 contributors (positive) and top 10 detractors (negative) | P0 |
| FR-5.5 | Identify holdings in sectors with unusual Phase 1 moves (z-score > 1.5) | P1 |
| FR-5.6 | Estimate factor exposures for holdings matching Final 1000 betas | P2 |

#### FR-6: Report Generation
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-6.1 | Generate integrated report rewriting market wrap through portfolio lens | P0 |
| FR-6.2 | Use Claude Opus 4.5 for report generation | P0 |
| FR-6.3 | Include portfolio performance summary at top (BLUF format) | P0 |
| FR-6.4 | Include sector contribution analysis with comparison to market | P0 |
| FR-6.5 | Include top contributors and detractors with context | P0 |
| FR-6.6 | Connect market themes to portfolio exposures | P0 |
| FR-6.7 | Output PDF only (same quality as Phase 1 reports) | P0 |
| FR-6.8 | Smart summarization for 100-500 holdings (not all holdings in prompt) | P0 |

#### FR-7: Pipeline Orchestration
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-7.1 | Single entry point script for portfolio report generation | P0 |
| FR-7.2 | Validate Phase 1 market data exists before generating | P0 |
| FR-7.3 | Support `--portfolio ID --date YYYY-MM-DD` flags | P0 |
| FR-7.4 | Support `--update-holdings file.xlsx` for portfolio updates | P0 |
| FR-7.5 | Support `--list` to show available portfolios | P1 |

### Non-Functional Requirements

#### NFR-1: Performance
| ID | Requirement | Target |
|----|-------------|--------|
| NFR-1.1 | Portfolio ingestion (500 holdings) | < 5 minutes |
| NFR-1.2 | Daily price fetch (500 holdings) | < 2 minutes |
| NFR-1.3 | Report generation | < 3 minutes |
| NFR-1.4 | Full pipeline execution | < 10 minutes |

#### NFR-2: Data Quality
| ID | Requirement | Target |
|----|-------------|--------|
| NFR-2.1 | Ticker resolution success rate | > 95% |
| NFR-2.2 | Price fetch success rate | > 98% |
| NFR-2.3 | Classification consistency with Phase 1 | 100% for Final 1000 matches |

#### NFR-3: Cost
| ID | Requirement | Target |
|----|-------------|--------|
| NFR-3.1 | Classification cost per portfolio (one-time) | < $1 |
| NFR-3.2 | Daily report generation cost | Same as Phase 1 (~$0.50) |
| NFR-3.3 | Price data cost | $0 (Yahoo Finance) |

---

## Technical Design

### Classification Strategy

**Decision Tree for Each Holding:**

```
For each holding in portfolio:
│
├─ Is ticker in Final 1000 database?
│   └─ YES → Use existing tier1/tier2/tier3 from assets table
│            classification_source = 'final1000' (FREE)
│
├─ Resolve ticker via yfinance
│   │
│   ├─ Is it an individual stock (quoteType = 'EQUITY')?
│   │   └─ YES → Use YFINANCE_SECTOR_MAP + YFINANCE_COUNTRY_MAP
│   │            tier1 = 'Equities', tier2 from sector, tier3 = [sector, region]
│   │            classification_source = 'yfinance_mapped' (FREE)
│   │
│   ├─ Is it an ETF/Fund (quoteType = 'ETF' or 'MUTUALFUND')?
│   │   └─ YES → Classify with Claude Haiku (same prompt as Phase 1)
│   │            classification_source = 'haiku' (~$0.002/asset)
│   │
│   └─ Is it a Bond/Preferred/Other?
│       └─ Use appropriate tier1 (Fixed Income, etc.) with Haiku
│
└─ Resolution failed?
    └─ Mark as 'failed', log error, continue (DO NOT skip silently)
```

### Taxonomy (Reused from Phase 1)

**TIER-1 (7 categories):**
1. Equities
2. Fixed Income
3. Commodities
4. Currencies (FX)
5. Multi-Asset / Thematic
6. Volatility / Risk Premia
7. Alternative / Synthetic

**TIER-2 (sub-categories by Tier-1):**
- Equities: Global Indices | Sector Indices | Country/Regional | Thematic/Factor | Real Estate / REITs
- Fixed Income: Sovereign Bonds | Corporate Credit | Credit Spreads | Yield Curves
- Commodities: Energy | Metals | Agriculture
- Currencies: Majors | EM FX
- Multi-Asset: Cross-Asset Indices | Inflation/Growth Themes
- Volatility: Vol Indices | Carry/Value Factors
- Alternative: Quant/Style Baskets | Custom/Proprietary

**TIER-3 TAGS (multi-select):**
- Region: US | Europe | Asia | EM | Global | China | Japan | India | Canada | APAC | Australia
- Sector/Theme: Tech | Energy | Financials | Healthcare | Industrials | Consumer | Defensive | ESG | Dividend | Growth | Value | Momentum | Quality | Infrastructure | Real Estate | Utilities
- Strategy: Active | Passive | Thematic | Quantitative | Factor-Based | Low Volatility | Defensive

### yfinance → Taxonomy Mapping

```python
YFINANCE_SECTOR_MAP = {
    # yfinance sector → (tier1, tier2, tier3_tags)
    'Technology': ('Equities', 'Sector Indices', ['Tech', 'Equity']),
    'Financial Services': ('Equities', 'Sector Indices', ['Financials', 'Equity']),
    'Healthcare': ('Equities', 'Sector Indices', ['Healthcare', 'Equity']),
    'Consumer Cyclical': ('Equities', 'Sector Indices', ['Consumer', 'Equity']),
    'Consumer Defensive': ('Equities', 'Sector Indices', ['Consumer', 'Defensive', 'Equity']),
    'Industrials': ('Equities', 'Sector Indices', ['Industrials', 'Equity']),
    'Energy': ('Equities', 'Sector Indices', ['Energy', 'Equity']),
    'Basic Materials': ('Equities', 'Sector Indices', ['Materials', 'Equity']),
    'Communication Services': ('Equities', 'Sector Indices', ['Tech', 'Equity']),
    'Utilities': ('Equities', 'Sector Indices', ['Utilities', 'Defensive', 'Equity']),
    'Real Estate': ('Equities', 'Real Estate / REITs', ['Real Estate', 'Equity']),
}

YFINANCE_COUNTRY_MAP = {
    # yfinance country → tier3 region tag
    'United States': 'US',
    'China': 'China',
    'Japan': 'Japan',
    'United Kingdom': 'Europe',
    'Germany': 'Europe',
    'France': 'Europe',
    'India': 'India',
    'Canada': 'Canada',
    'Australia': 'Australia',
    'Brazil': 'EM',
    'South Korea': 'Asia',
    'Taiwan': 'Asia',
    # Default: 'Global'
}
```

### Database Schema Extension

```sql
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

-- Portfolio holdings (updated monthly)
CREATE TABLE IF NOT EXISTS portfolio_holdings (
    portfolio_id TEXT NOT NULL,
    symbol TEXT NOT NULL,              -- Plain ticker from portfolio file (AAPL, EWZ)

    -- From portfolio file (actual client data)
    quantity REAL NOT NULL,            -- Long Quantity from file
    market_value REAL,                 -- Market Value from file (USD)
    avg_price REAL,                    -- Average Price (cost basis) from file
    open_pnl REAL,                     -- Long Open Profit/Loss from file

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
    classification_source TEXT,        -- 'final1000', 'yfinance_mapped', 'haiku'
    resolution_status TEXT DEFAULT 'pending',

    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),

    PRIMARY KEY (portfolio_id, symbol)
);

-- Daily portfolio snapshot
CREATE TABLE IF NOT EXISTS portfolio_daily (
    date TEXT NOT NULL,
    portfolio_id TEXT NOT NULL,
    symbol TEXT NOT NULL,

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

    PRIMARY KEY (date, portfolio_id, symbol)
);

-- Portfolio aggregates by dimension
CREATE TABLE IF NOT EXISTS portfolio_aggregates (
    date TEXT NOT NULL,
    portfolio_id TEXT NOT NULL,
    dimension_type TEXT NOT NULL,      -- 'sector', 'country', 'tier1', 'tier2'
    dimension_value TEXT NOT NULL,

    holding_count INTEGER,
    total_weight REAL,
    total_value_usd REAL,
    weighted_return_1d REAL,
    contribution_1d REAL,

    PRIMARY KEY (date, portfolio_id, dimension_type, dimension_value)
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
    generation_time_ms INTEGER
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_portfolio_holdings_ticker ON portfolio_holdings(ticker);
CREATE INDEX IF NOT EXISTS idx_portfolio_daily_date ON portfolio_daily(date);
CREATE INDEX IF NOT EXISTS idx_portfolio_daily_portfolio ON portfolio_daily(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_aggregates_date ON portfolio_aggregates(date);
```

### File Structure

Phase 2 is a **completely standalone product** with its own directory structure:

```
Phase 2 Portfolio Reports/
├── PRD.md                                # This document
├── README.md                             # Setup and usage guide
├── requirements.txt                      # Python dependencies
├── test_portfolio.xlsx                   # Test portfolio (24 holdings)
│
├── database/
│   ├── schema.sql                        # Database schema
│   └── portfolio.db                      # SQLite database (SEPARATE from Phase 1)
│
├── scripts/
│   ├── 01_ingest_portfolio.py            # Load Excel, resolve tickers, classify
│   ├── 02_fetch_daily_prices.py          # Yahoo Finance price fetching
│   ├── 03_compute_analytics.py           # Sector/region aggregations
│   ├── 04_generate_report.py             # Portfolio report generation
│   ├── run_pipeline.py                   # Main entry point
│   │
│   └── utils/
│       ├── __init__.py
│       ├── db.py                         # Database utilities
│       ├── llm.py                        # LLM utilities (Claude Opus 4.5)
│       ├── yfinance_utils.py             # Yahoo Finance helpers
│       ├── taxonomy.py                   # Tier 1/2/3 taxonomy definitions
│       └── classifier.py                 # Haiku classification for ETFs
│
├── prompts/
│   └── portfolio_daily_wrap.md           # Prompt template for report generation
│
└── outputs/
    └── {portfolio_id}/
        ├── portfolio_wrap_YYYY-MM-DD.md
        └── portfolio_wrap_YYYY-MM-DD.pdf
```

---

## Report Structure

### Portfolio Daily Wrap Sections

1. **Portfolio Performance Summary** (BLUF - start here)
   - Total return today ($ and %)
   - Total P&L (unrealized)
   - Key driver (sector/holding)
   - Biggest contributor/detractor

2. **Portfolio at a Glance**
   - Total market value
   - Daily return ($ and %)
   - YTD return
   - Total unrealized P&L
   - Holding count

3. **Regional/Country Exposure** (given ETF-heavy portfolio)
   - Table: Region/Country | Weight | Return | Contribution | vs Benchmark
   - Geographic concentration analysis

4. **Top Contributors & Detractors**
   - Top 5 contributors with return, contribution, and P&L
   - Top 5 detractors with return, contribution, and P&L
   - Context on why these names moved (connect to market themes)

5. **P&L Analysis**
   - Holdings with largest unrealized gains
   - Holdings with largest unrealized losses
   - Cost basis vs current value analysis

6. **Market Context Through Portfolio Lens**
   - Phase 1 market themes filtered by portfolio relevance
   - "You have X% exposure to [region/theme] via [holdings]"
   - EM exposure analysis (given heavy EM weight in sample portfolio)

7. **Unusual Patterns Affecting Portfolio**
   - Holdings in categories with z-score > 1.5
   - Country ETFs with unusual moves
   - Currency impacts on international holdings

8. **Areas of Attention**
   - Concentration risks (geographic, sector)
   - Holdings approaching significant P&L thresholds
   - Market events that could affect portfolio regions

### Smart Summarization Strategy

For 100-500 holdings, cannot send all to LLM. Strategy:

1. **Always include**: Sector-level aggregates (11 sectors max)
2. **Always include**: Top 10 contributors + Top 10 detractors
3. **Include if relevant**: Holdings in "unusual" Phase 1 categories
4. **Include if relevant**: Holdings in sectors with unusual moves
5. **Summarize rest**: Aggregate statistics only

---

## Implementation Plan

### Step 1: Database Schema Extension
- Add portfolio tables to `database/schema.sql`
- Run schema migration on `market_data.db`

### Step 2: Taxonomy Mapping Module
- Create `scripts/utils/taxonomy_mapping.py`
- Define `YFINANCE_SECTOR_MAP` and `YFINANCE_COUNTRY_MAP`
- Create `map_stock_to_taxonomy(yf_info)` function

### Step 3: Yahoo Finance Utilities
- Create `scripts/utils/yfinance_utils.py`
- `resolve_ticker(plain_ticker)` - resolve with exchange suffix handling
- `batch_fetch_info(tickers)` - get sector, country, security type
- `batch_fetch_prices(tickers, date)` - get daily prices

### Step 4: Portfolio Classifier
- Create `scripts/utils/portfolio_classifier.py`
- Copy TAXONOMY and SYSTEM_PROMPT from `unified_asset_classifier.py`
- `classify_etf_with_haiku(ticker, name, description)` - for ETFs not in Final 1000

### Step 5: Portfolio Ingestion Script
- Create `scripts/portfolio/01_ingest_portfolio.py`
- Load Excel/CSV
- Resolve tickers, classify based on decision tree
- Save to `portfolio_holdings` table

### Step 6: Daily Price Fetcher
- Create `scripts/portfolio/02_fetch_daily_prices.py`
- Fetch from Yahoo Finance
- Compute weights and contributions
- Save to `portfolio_daily` table

### Step 7: Analytics Computation
- Create `scripts/portfolio/03_compute_analytics.py`
- Aggregate by sector, region, tier1, tier2
- Save to `portfolio_aggregates` table

### Step 8: Portfolio Prompt Template
- Create `prompts/portfolio_daily_wrap.md`
- Design prompt with portfolio sections
- Include placeholders for data injection

### Step 9: Report Generator
- Create `scripts/portfolio/04_generate_report.py`
- Load Phase 1 market data
- Prepare portfolio context (smart summarization)
- Generate via Claude Opus 4.5
- Convert to PDF

### Step 10: Pipeline Orchestrator
- Create `scripts/portfolio/run_portfolio_pipeline.py`
- Single entry point with CLI flags
- Validation and error handling

---

## Error Handling (FAIL IS FAIL)

Per project guidelines - no silent fallbacks:

| Failure | Action |
|---------|--------|
| Ticker resolution fails | Log error, mark as 'failed', include warning in report |
| Price fetch fails | Try previous day (mark 'stale'), if none, flag 'no_data' |
| Classification fails | Default to 'Unknown', log for manual review |
| Phase 1 data missing | FAIL with clear error - do not generate partial report |

---

## Testing Plan

### Test 1: Portfolio Ingestion
```bash
python scripts/portfolio/01_ingest_portfolio.py --portfolio TEST --file test_holdings.xlsx
# Verify: portfolio_holdings table has all holdings with resolution_status='resolved'
```

### Test 2: Daily Prices
```bash
python scripts/portfolio/02_fetch_daily_prices.py --portfolio TEST --date 2026-01-30
# Verify: portfolio_daily table has prices, returns, contributions
```

### Test 3: Analytics
```bash
python scripts/portfolio/03_compute_analytics.py --portfolio TEST --date 2026-01-30
# Verify: portfolio_aggregates has sector breakdowns
```

### Test 4: Report Generation
```bash
python scripts/portfolio/04_generate_report.py --portfolio TEST --date 2026-01-30
# Verify: PDF generated, mentions specific holdings, sectors, contributions
```

### Test 5: Full Pipeline
```bash
python scripts/portfolio/run_portfolio_pipeline.py --portfolio TEST --date 2026-01-30
# Verify: End-to-end execution, PDF output
```

---

## Dependencies

### New Dependencies
```
yfinance>=0.2.0      # Yahoo Finance data
```

### Existing Dependencies (Already Available)
- anthropic - Claude API
- pandas - Data manipulation
- openpyxl - Excel I/O
- weasyprint - PDF generation
- sqlite3 - Database

---

## Critical Files to Reference

| File | Purpose | What to Reuse |
|------|---------|---------------|
| `Step 2/unified_asset_classifier.py` | Classification system | TAXONOMY, SYSTEM_PROMPT, classify_asset() |
| `Step 2/classify_etfs_full.py` | ETF classification | Prompt structure for ETFs |
| `Step 4/scripts/03_generate_daily_report.py` | Report generation | prepare_data_summary(), inject_data_into_prompt() |
| `Step 4/scripts/utils/llm.py` | LLM utilities | generate_anthropic(), model selection |
| `Step 4/scripts/utils/db.py` | Database utilities | Connection patterns |
| `Step 4/prompts/daily_wrap.md` | Prompt template | Structure, BLUF format |

---

## Success Criteria

1. **Portfolio ingestion** resolves >95% of tickers successfully
2. **Classification** uses same taxonomy as Phase 1 (100% consistent for Final 1000 matches)
3. **Report** mentions specific holdings, sectors, and contributions
4. **Report** connects market themes to portfolio exposures
5. **Full pipeline** completes in <10 minutes for 500 holdings
6. **PDF quality** matches Phase 1 reports

---

## Future Enhancements (Out of Scope for v1)

- Multi-portfolio batch generation
- Weekly/monthly summary reports
- Cost basis and P&L tracking
- Factor attribution analysis
- Benchmark comparison
- Email delivery
- Interactive web dashboard

---

## Appendix: Sample Portfolio File

### Actual Client Portfolio Format

```
| Symbol | Market Value | Average Price | Long Quantity | Long Open Profit/Loss |
|--------|--------------|---------------|---------------|----------------------|
| ASHR   | 515033.98    | 31.148351     | 15351.2365    | 36868.284575         |
| EDEN   | 36877.23     | 109.891667    | 300.0000      | 3909.730000          |
| EPHE   | 195206.42    | 25.977560     | 7525.3052     | -282.653112          |
| EPOL   | 554744.52    | 32.423503     | 14454.0000    | 86095.210000         |
| EWG    | 232794.00    | 41.337894     | 5400.0000     | 9569.370000          |
| EWI    | 1266636.16   | 50.898848     | 22550.0474    | 118864.722458        |
| EWJ    | 990179.44    | 79.690520     | 11543.2437    | 70292.354586         |
| EWP    | 987647.01    | 41.054836     | 17671.2652    | 262156.122028        |
| EWS    | 421389.17    | 27.252048     | 14646.8255    | 22233.179634         |
| EWZ    | 1886418.92   | 30.864623     | 49382.6942    | 362240.668438        |
| TUR    | 222746.53    | 32.386095     | 5357.0595     | 49252.294010         |
| VTV    | 1716253.46   | 170.940086    | 8581.2673     | 249370.890000        |
```

### Column Mapping

| Portfolio File Column | Database Field | Usage |
|----------------------|----------------|-------|
| Symbol | symbol | Ticker for resolution |
| Long Quantity | quantity | Position size |
| Market Value | market_value | Weight calculation (more accurate than qty × price) |
| Average Price | avg_price | Cost basis per share |
| Long Open Profit/Loss | open_pnl | Unrealized P&L for reporting |

### Required vs Optional Columns

**Required:**
- Symbol
- Long Quantity

**Optional but Recommended:**
- Market Value (used for accurate weight calculation)
- Average Price (enables cost basis and P&L tracking)
- Long Open Profit/Loss (enables P&L reporting)

### Portfolio Characteristics

This portfolio contains:
- **12 holdings** (~$9M total value)
- **Mix of country ETFs**: EWZ (Brazil), EWI (Italy), EWJ (Japan), EWP (Spain), etc.
- **Regional ETFs**: ASHR (China A-shares), VTV (US Value)
- **Could also contain individual stocks** in other portfolios
