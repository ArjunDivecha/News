# Step 4 Report Generation - Implementation Plan

**Created:** 2026-01-30  
**Status:** In Progress (Stage 2 Complete)

---

## Overview

Build a personal report generation system that:
1. Pulls real-time Bloomberg data for 970 curated assets
2. Stores historical data in SQLite for pattern analysis
3. Uses multiple LLMs (GPT, Claude, Gemini) to generate daily market wrap reports
4. Produces 15-minute flash reports during market hours
5. Outputs professional PDF reports

**Key Principle:** All data stays on your Mac. Reports reference your proprietary 3-tier taxonomy (Tier-1/Tier-2/Tier-3), never individual securities or data sources.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    STATIC LAYER (Monthly Update)            │
│  Final 1000 Asset Master List                               │
│  - 970 assets with 3-tier taxonomy                         │
│  - 18 beta exposures per asset                             │
│  - Stored in SQLite assets table                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    DYNAMIC LAYER (Daily/Intraday)           │
│  Bloomberg Data (via Parallels)                             │
│  - Daily closing prices → daily_prices table               │
│  - Intraday snapshots → intraday_prices table              │
│  - Category aggregates → category_stats table              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    REPORT GENERATION                        │
│  - Daily wrap (after close) → 3 LLM models                 │
│  - Flash reports (every 15 min) → fast LLM                 │
│  - Historical pattern analysis enriches prompts            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT                                   │
│  - PDF reports in outputs/daily/ and outputs/flash/        │
│  - Markdown archive in database                            │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Stages

| Stage | Description | Status | Test |
|-------|-------------|--------|------|
| **1** | Folder structure + Database schema | ✅ Complete | 6/6 tests passed |
| **2** | Sync static data (Final 1000 → SQLite) | ✅ Complete | 970 assets synced |
| **3** | Prompt templates (daily + flash) | ✅ Complete | Files created |
| **4** | Mock data generator | ✅ Complete | 5/5 tests passed |
| **5** | Report generator (LLM integration) | ✅ Complete | 5/5 tests passed |
| **6** | PDF converter | ✅ Complete | PDF generated |
| **7** | Bloomberg fetcher (Parallels) | 🔲 Pending | - |
| **8** | End-to-end with real data | 🔲 Pending | - |
| **9** | Flash reports + real-time | 🔲 Pending | - |

---

## Database Schema

**Tables:**
- `assets` - 970 assets with taxonomy and betas
- `daily_prices` - Daily price snapshots (accumulates history)
- `intraday_prices` - Intraday snapshots (kept 7 days)
- `category_stats` - Pre-computed aggregates
- `factor_returns` - Factor returns for beta attribution
- `reports` - Generated report archive

**Views:**
- `v_latest_daily` - Latest daily data with asset info
- `v_tier1_summary` - Tier-1 aggregates for latest date
- `v_tier2_summary` - Tier-2 aggregates for latest date

---

## Report Philosophy

**What Reports Show:**
- Your proprietary taxonomy (Tier-1, Tier-2, Tier-3 tags)
- Category-level performance
- Historical patterns, streaks, extremes
- Beta attribution analysis

**What Reports DON'T Show:**
- Individual security names or tickers
- Data source attribution (Bloomberg, Goldman, etc.)
- Raw price data

**Core Motif:** UNUSUAL PATTERNS
- Flag anomalies, outliers, regime shifts
- Reference historical percentiles and streaks
- Challenge consensus when data compels

---

## File Structure

```
Step 4 Report Generation/
├── PLAN.md                     # This file
├── README.md                   # Usage guide
├── requirements.txt            # Dependencies
│
├── database/
│   ├── schema.sql              # ✅ Database schema
│   ├── market_data.db          # ✅ SQLite database (970 assets)
│   └── init_db.py              # ✅ Initialization script
│
├── prompts/
│   ├── daily_wrap.md           # ✅ Daily report prompt
│   └── flash_report.md         # ✅ Flash report prompt
│
├── scripts/
│   ├── 01_sync_static_data.py  # ✅ Sync Final 1000 → SQLite
│   ├── 02_refresh_bloomberg.py # 🔲 Pull Bloomberg data
│   ├── 03_generate_daily.py    # 🔲 Daily report generator
│   ├── 04_flash_report.py      # 🔲 Flash report generator
│   ├── 05_pattern_analysis.py  # 🔲 Historical patterns
│   ├── bloomberg_fetcher.py    # 🔲 Bloomberg script (for Parallels)
│   └── utils/
│       ├── db.py               # 🔲 Database utilities
│       ├── llm.py              # 🔲 LLM API wrappers
│       └── pdf.py              # 🔲 PDF generation
│
└── outputs/
    ├── daily/                  # Daily report PDFs
    └── flash/                  # Flash report PDFs
```

---

## Bloomberg Integration (Parallels)

Since Bloomberg only works in Parallels (Windows):

1. **On Windows (Parallels):** Run `bloomberg_fetcher.py`
   - Connects to Bloomberg DAPI
   - Pulls data for 970 tickers
   - Writes CSV to shared Dropbox folder

2. **On Mac:** Run `02_refresh_bloomberg.py`
   - Reads CSV from Dropbox
   - Loads into SQLite
   - Triggers report generation

---

## Daily Workflow

### After Market Close (16:30 ET)
```bash
# On Windows (Parallels)
python bloomberg_fetcher.py

# On Mac
cd "Step 4 Report Generation"
python scripts/02_refresh_bloomberg.py --date today
python scripts/03_generate_daily.py
```

### During Market Hours (Every 15 min)
```bash
# On Windows
python bloomberg_fetcher.py --intraday

# On Mac
python scripts/04_flash_report.py
```

---

## Next Steps

**Stage 4: Mock Data Generator**
- Create realistic fake Bloomberg data for testing
- Allows full pipeline testing without Bloomberg access

**Stage 5: Report Generator**
- LLM API integration (OpenAI, Anthropic, Google)
- Prompt injection with computed statistics
- Multi-model parallel generation

**Stage 6: PDF Converter**
- Markdown → PDF conversion
- Professional styling

---

## Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
openpyxl>=3.1.0
sqlite3 (built-in)
openai>=1.0.0
anthropic>=0.20.0
google-generativeai>=0.3.0
reportlab>=4.0.0
blpapi>=3.19.0 (Windows only)
```
