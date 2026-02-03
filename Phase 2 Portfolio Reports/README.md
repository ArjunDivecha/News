# Phase 2 Portfolio Reports

Personalized daily portfolio wrap reports powered by Claude Opus 4.5.

## Overview

This system generates customized daily market reports tailored to YOUR specific portfolio holdings. Unlike generic market commentary, these reports focus on:

- **Your actual positions** and their performance
- **Your exposures** (regional, sector, theme)
- **Attribution analysis** - what helped and hurt
- **Risk assessment** - concentration, correlations
- **Actionable insights** - position sizing, profit-taking candidates

## Quick Start

```bash
cd scripts

# 1. Ingest a new portfolio
python run_pipeline.py --portfolio MYPORT --file ../sample_holdings.xlsx

# 2. Generate report for latest trading day
python run_pipeline.py --portfolio MYPORT

# 3. Generate report for specific date
python run_pipeline.py --portfolio MYPORT --date 2026-01-30

# 4. List available portfolios
python run_pipeline.py --list

# 5. Update portfolio holdings
python run_pipeline.py --portfolio MYPORT --update-holdings new_holdings.xlsx
```

## Input Format

The system accepts Excel (.xlsx) or CSV files with the following columns:

| Column | Required | Description |
|--------|----------|-------------|
| Symbol | Yes | Plain ticker (e.g., AAPL, EWZ) |
| Long Quantity / Quantity | Yes | Positive for LONG, negative for SHORT |
| Market Value | Recommended | Current market value in USD |
| Average Price | Optional | Cost basis per share |
| Long Open Profit/Loss | Optional | Unrealized P&L |

### Example Portfolio

```csv
Symbol,Long Quantity,Market Value,Average Price,Long Open Profit/Loss
AAPL,1000,175000,150.00,25000
EWZ,5000,185000,33.00,20000
IWF,-2000,-250000,120.00,-10000
```

## Pipeline Steps

The pipeline consists of 4 steps that run automatically:

### Step 1: Portfolio Ingestion (`01_ingest_portfolio.py`)
- Loads holdings from Excel/CSV
- Resolves tickers via Yahoo Finance
- Classifies assets using:
  1. Phase 1 Final 1000 database (free lookup)
  2. yfinance sector/country mapping (for stocks)
  3. Claude Haiku API (for ETFs/funds, ~$0.002/asset)
- Stores in `portfolio_holdings` table

### Step 2: Daily Price Fetch (`02_fetch_daily_prices.py`)
- Fetches current prices from Yahoo Finance
- Calculates daily and YTD returns
- Computes portfolio weights and contribution to return
- Stores in `portfolio_daily` and `portfolio_summary` tables

### Step 3: Analytics Computation (`03_compute_analytics.py`)
- Aggregates holdings by tier1, tier2, region, sector, tier3 tags
- Calculates weighted returns and contributions per dimension
- Stores in `portfolio_aggregates` table

### Step 4: Report Generation (`04_generate_report.py`)
- Loads portfolio data and Phase 1 market context
- Injects into prompt template
- Generates report with Claude Opus 4.5
- Converts to professional PDF using PrinceXML
- Outputs Markdown and PDF to `outputs/` directory

## Output

Reports are saved in:
```
outputs/{portfolio_id}/
├── portfolio_wrap_2026-01-30.md    # Markdown report
└── portfolio_wrap_2026-01-30.pdf   # Professional PDF (PrinceXML, PDF/X-1a:2003)
```

**PDF Quality:**
- Generated with PrinceXML for professional print quality
- PDF/X-1a:2003 profile with sRGB color management
- Proper typography, tables, and page layout
- Matches Phase 1 report styling

### Report Structure

1. **Executive Synthesis** - One-line summary and key takeaways
2. **Portfolio at a Glance** - Key metrics table
3. **Top Contributors & Detractors** - What helped and hurt
4. **Regional Exposure Analysis** - Geographic attribution
5. **Sector/Theme Exposure** - Sector breakdown
6. **Long vs Short Analysis** - Hedge effectiveness
7. **P&L Analysis** - Profit-taking and loss candidates
8. **Concentration & Risk Notes** - Risk flags
9. **Market Context** - Connection to broader market themes

## Configuration

### Environment Variables

Create a `.env` file in the project root (or use the existing one):

```
ANTHROPIC_API_KEY=your_key_here
```

### Database

Phase 2 uses a separate database from Phase 1:
- **Phase 2**: `database/portfolio.db`
- **Phase 1**: `../Step 4 Report Generation/database/market_data.db` (for market context)

To initialize the database:
```bash
sqlite3 database/portfolio.db < database/schema.sql
```

## Dependencies

### Required Software
- **PrinceXML** (for professional PDF generation): https://www.princexml.com/
- **Python 3.11+**

### Python Packages
```
pandas>=2.0
openpyxl
yfinance>=0.2.30
anthropic>=0.25
python-dotenv
jinja2
```

Install with:
```bash
pip install pandas openpyxl yfinance anthropic python-dotenv jinja2
```

## Cost Estimation

- **Claude Haiku** (classification): ~$0.002 per ETF/fund classification
- **Claude Opus 4.5** (report): ~$0.15-0.30 per report
- **Yahoo Finance**: Free

For a 50-holding portfolio:
- Initial ingestion: ~$0.10 (if 50% are ETFs needing Haiku)
- Daily report: ~$0.20
- Monthly cost: ~$4-6 for daily reports

## File Structure

```
Phase 2 Portfolio Reports/
├── README.md
├── PRD.md                          # Product requirements
├── database/
│   ├── schema.sql                  # Database schema
│   └── portfolio.db                # SQLite database
├── scripts/
│   ├── 01_ingest_portfolio.py      # Step 1
│   ├── 02_fetch_daily_prices.py    # Step 2
│   ├── 03_compute_analytics.py     # Step 3
│   ├── 04_generate_report.py       # Step 4
│   ├── run_pipeline.py             # Orchestrator
│   └── utils/
│       ├── __init__.py
│       ├── db.py                   # Database utilities
│       ├── llm.py                  # LLM utilities
│       ├── taxonomy.py             # Classification mappings
│       ├── yfinance_utils.py       # Yahoo Finance utilities
│       └── pdf_prince/             # PrinceXML PDF converter
│           ├── convert.py          # PDF generation
│           ├── charts.py           # Chart generation
│           └── templates/          # HTML/CSS templates
├── prompts/
│   └── portfolio_daily_wrap.md     # Report prompt template
├── outputs/                        # Generated reports
│   └── {portfolio_id}/
│       ├── portfolio_wrap_YYYY-MM-DD.md
│       └── portfolio_wrap_YYYY-MM-DD.pdf
└── test_portfolio.xlsx             # Sample portfolio file
```

## Troubleshooting

### Ticker Resolution Failures
- Check if the ticker symbol is correct (use Yahoo Finance format)
- Some OTC stocks or delisted securities may fail
- Failed holdings are logged but don't block the pipeline

### Price Fetch Issues
- Weekend/holiday dates are automatically adjusted to last trading day
- Some instruments (futures, exotic ETFs) may not have price data
- Missing prices use the market value from holdings file as fallback

### PDF Generation Errors
- Ensure PrinceXML is installed and the `prince` command is in your PATH
- Download from: https://www.princexml.com/download/
- Markdown version is always generated even if PDF fails
- Check PrinceXML installation: `prince --version`

## Future Enhancements

- [ ] Multi-day return analysis (5d, 1m, YTD)
- [ ] Historical performance charts
- [ ] Risk metrics (VaR, Sharpe, max drawdown)
- [ ] Benchmark comparison
- [ ] Email delivery
- [ ] Scheduled generation (cron/airflow)
