# AGENTS.md - Financial Analytics Pipeline

**Last Updated**: 2026-06-10 (rearchitecture + repo cleanup)  
**Project Root**: `/Users/arjundivecha/Dropbox/AAA Backup/A Working/News`

---

> ## IMPORTANT: Rearchitected 2026-06 - read this first
>
> The daily reporting chain (Phase 0 + Step 4 + Phase 2) described in parts
> of this document was **replaced by the unified system in `report/`**
> (single command: `python3 report/main.py`) and **moved to `archive/`**.
> See `report/README.md` and the "Learned Workspace Facts" section at the
> bottom of this file. Steps 1-3 (universe construction) are still current.

---

## Project Overview

This is a **multi-phase financial data pipeline** that transforms raw multi-asset market data (ETFs, Bloomberg Indices, Goldman Sachs baskets) into a curated dataset of ~1,000 diversified assets for news analysis and market monitoring.

### Data Sources
- **Bloomberg Indices** (~438 indices) - Market indices with performance metrics
- **ETF Master List** (~1,619 ETFs) - Exchange-traded funds with fund metadata
- **Goldman Sachs Baskets** (~2,667 baskets) - Thematic baskets via Marquee API
- **Thematic ETFs** (~231 ETFs) - Specialized thematic ETFs

### Target Output
- **Final 1000 Asset Master List.xlsx** - Optimized portfolio of ~1,000 diversified assets with strategic allocation across 7 Tier-1 categories

---

## Technology Stack

### Core Dependencies
```bash
pip install pandas numpy scipy scikit-learn openpyxl
pip install anthropic exa-py yfinance
pip install matplotlib seaborn
pip install gs-quant  # Goldman Sachs API
```

### Python Version
- Python 3.11+
- Optimized for high-memory systems (128GB RAM recommended)

### External APIs
- **Anthropic API** (Haiku 4.5) - Asset taxonomy classification (original method)
- **Tinker API** - Fine-tuned Llama 3.1 8B inference (new method, $0/run)
- **Goldman Sachs Marquee API** - Basket data retrieval
- **Exa API** - Web search for meme stock discovery
- **Yahoo Finance** - Market data validation

### Classification Methods
1. **Haiku Classification** (Original): ~$50-100/run, 30-60 min
2. **Fine-Tuned Llama** (New): $0/run, ~22 min for 970 assets
   - Model: `arjun-fund-classifier-v1-r7j1` via Tinker
   - ~80% Tier-1 agreement with Haiku

---

## Project Structure

```
News/
├── Step 1 Data Collection/              # Raw data acquisition
│   ├── Bloomberg_Indices_Cluster.py     # Filter Bloomberg indices (target: 500)
│   ├── ETF Cluster.py                   # Filter ETFs (target: 500)
│   ├── gs_basket_data.py                # Goldman Sachs API retrieval
│   ├── gs_basket_data_with_headings.py  # Enhanced GS data with headings
│   ├── README.md                        # Detailed usage instructions
│   └── Bloomberg_Indices_Cluster_documentation.md
│
├── Step 2 Data Processing - Final1000/  # Classification & integration
│   ├── unified_asset_classifier.py      # Test classifier (all sources)
│   ├── etf_classifier.py                # ETF classification with Haiku
│   ├── classify_bloomberg_full.py       # Bloomberg classification
│   ├── classify_etfs_full.py            # ETF classification runner
│   ├── classify_goldman_full.py         # Goldman classification
│   ├── classify_thematic_etfs_full.py   # Thematic ETF classification
│   ├── merge_classified_files.py        # Merge 4 sources into master
│   ├── final_selection_algorithm.py     # Select final 1000 assets
│   ├── README.md                        # Complete workflow guide
│   ├── CLAUDE.md                        # Architecture overview
│   └── PRD.md                           # Product requirements
│
├── Step 3 Data Analysis/                # Analytics & insights
│   ├── p0_deduplication_analysis.py     # Beta vector clustering
│   ├── p0_deduplication_analysis_fast.py
│   ├── p0_deduplication_analysis_full.py
│   ├── p1_category_analysis.py          # Performance by category
│   ├── p2_factor_decomposition.py       # Factor exposures
│   └── MemeFinder.py                    # Meme stock discovery tool
│
├── fine tuning/                         # ML model training
│   ├── scripts/                         # Training & inference scripts
│   │   ├── 01_prepare_data.py           # Data preparation
│   │   ├── 05_train_proper.py           # Production training
│   │   ├── classify_final_1000.py       # Classify Final 1000 list
│   │   ├── compare_models.py            # Haiku vs Fine-tuned comparison
│   │   └── predict_funds.py             # Batch prediction
│   ├── data/processed/                  # Training data JSONs
│   ├── README.md                        # Fine-tuning guide
│   └── INFERENCE.md                     # Inference documentation
│
├── Step 4 Report Generation/            # LLM-powered market reports
│   ├── database/                        # SQLite for historical data
│   │   ├── schema.sql                   # Database schema
│   │   ├── market_data.db               # 970 assets + daily prices
│   │   └── init_db.py                   # Initialization script
│   ├── prompts/                         # LLM prompt templates
│   │   ├── daily_wrap.md                # Daily report (1000-1500 words)
│   │   └── flash_report.md              # Flash report (200-300 words)
│   ├── scripts/                         # Pipeline scripts
│   │   ├── 01_sync_static_data.py       # Sync Final 1000 → SQLite
│   │   ├── 02_refresh_bloomberg.py      # Pull daily Bloomberg data
│   │   ├── 03_generate_daily.py         # Multi-model daily report
│   │   └── 04_flash_report.py           # 15-min flash reports
│   ├── outputs/                         # Generated PDF reports
│   │   ├── daily/                       # Daily wrap PDFs
│   │   └── flash/                       # Flash report PDFs
│   ├── PLAN.md                          # Implementation plan
│   └── README.md                        # Usage guide
│
└── .claude/
    └── settings.local.json              # Claude Code permissions
```

---

## Execution Sequence

### Complete Pipeline (45-80 minutes runtime)

```bash
# === STAGE 1: Data Collection ===
cd "Step 1 Data Collection"
python Bloomberg_Indices_Cluster.py      # Input: Bloomberg Indices.xlsx → Output: Filtered Bloomberg Indices.xlsx
python "ETF Cluster.py"                  # Input: ETF Master List.xlsx → Output: Filtered ETF List.xlsx
python gs_basket_data.py                 # API call → GSCB_FLAGSHIP_coverage_with_desc_ALL.xlsx

# === STAGE 2: Classification ===
cd "../Step 2 Data Processing - Final1000"
python classify_bloomberg_full.py        # Output: Filtered Bloomberg Indices Classified.xlsx
python classify_etfs_full.py             # Output: ETF Master List Classified.xlsx
python classify_goldman_full.py          # Output: GSCB_FLAGSHIP_coverage_Classified.xlsx
python classify_thematic_etfs_full.py    # Output: Thematic ETFs Classified.xlsx
python merge_classified_files.py         # Merge all → Master Asset List Classified.xlsx (~4,933 assets)

# === STAGE 3: Analysis ===
cd "../Step 3 Data Analysis"
python p1_category_analysis.py           # Output: P1_Tier1_Analysis.xlsx, P1_Tier2_Analysis.xlsx
python p0_deduplication_analysis.py      # Output: Dedup Report - Top 30 Groups.xlsx
python p2_factor_decomposition.py        # Output: P2_Enriched_Asset_Profiles.xlsx

# === FINAL SELECTION ===
cd "../Step 2 Data Processing - Final1000"
python final_selection_algorithm_v2.py   # Output: Final 1000 Asset Master List.xlsx (enhanced thematic diversity)

# === OPTIONAL: FINE-TUNED CLASSIFICATION ===
cd "../fine tuning"
source venv/bin/activate && source .env
python scripts/classify_final_1000.py    # Reclassify with fine-tuned Llama model
```

---

## Code Conventions

### Docstring Format
Every Python file follows this header convention:

```python
"""
=============================================================================
SCRIPT NAME - Brief Description
=============================================================================

INPUT FILES:
- /absolute/path/to/input.xlsx
  Description: What the file contains
  Required Format: Excel with specific columns
  Key Columns: Col1, Col2, Col3

OUTPUT FILES:
- /absolute/path/to/output.xlsx
  Description: What the output contains
  Format: Excel file structure
  Contents: Detailed description

VERSION HISTORY:
v1.0.0 (YYYY-MM-DD): Initial release
v1.1.0 (YYYY-MM-DD): Feature additions
v1.2.0 (YYYY-MM-DD): Documentation updates

PURPOSE:
One-line description of what this script does.
"""
```

### Hardcoded Paths
Scripts use **absolute paths** referencing the Dropbox location:
```python
input_file = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/News/ETF Master List.xlsx"
```

### Progress Logging
All scripts print detailed progress to console:
```python
print("="*80)
print("SCRIPT NAME")
print("="*80)
print("\n[1/4] Processing step...")
print(f"  Progress: {idx + 1}/{len(df)}")
```

---

## Key Configuration

### Taxonomy (7 Tier-1 Categories)
1. **Equities** - Stock indices, ETFs, equity-focused baskets
2. **Fixed Income** - Bonds, credit, yield-focused instruments
3. **Commodities** - Energy, metals, agriculture
4. **Currencies (FX)** - Currency pairs and FX instruments
5. **Multi-Asset / Thematic** - Cross-asset, thematic baskets
6. **Volatility / Risk Premia** - VIX, volatility indices
7. **Alternative / Synthetic** - Quant baskets, factor portfolios

### Strategic Allocation Targets (Final 1000)
- Equities: 520 (52%)
- Fixed Income: 170 (17%)
- Commodities: 110 (11%)
- Currencies (FX): 70 (7%)
- Multi-Asset / Thematic: 80 (8%)
- Volatility / Risk Premia: 50 (5%)
- Alternative / Synthetic: 20 (2%)

---

## Testing & Validation

### No Unit Tests
This project uses **manual execution** and **data validation** rather than automated tests:
- Scripts validate input file existence before processing
- Progress files saved every 50 items during classification (`*_PROGRESS.xlsx`)
- Summary statistics printed at completion

### Validation Checkpoints
1. **Input file existence** - Scripts exit gracefully if inputs missing
2. **Data quality checks** - Missing value counts reported
3. **Distribution reports** - Tier-1/Tier-2 category counts printed
4. **Progress saving** - Intermediate files for long-running processes

---

## Security Considerations

### API Credentials

All API keys are now stored in the `.env` file at the project root (which is in `.gitignore` and should never be committed):

```bash
# .env
EXA_API_KEY=your_exa_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

**Scripts load keys automatically using `python-dotenv`:**

1. **Exa API** (`MemeFinder.py`)
   ```python
   from dotenv import load_dotenv
   load_dotenv()
   exa = Exa(api_key=os.getenv("EXA_API_KEY"))
   ```

2. **Anthropic API** (`unified_asset_classifier.py`, `etf_classifier.py`, etc.)
   ```python
   from dotenv import load_dotenv
   load_dotenv()
   client = Anthropic()  # Uses ANTHROPIC_API_KEY from .env
   ```

3. **Goldman Sachs API** (`gs_basket_data.py`, `gs_basket_data_with_headings.py`)
   ```python
   from dotenv import load_dotenv
   load_dotenv()
   client_id = os.getenv('GS_CLIENT_ID')
   client_secret = os.getenv('GS_CLIENT_SECRET')
   GsSession.use(client_id=client_id, client_secret=client_secret, ...)
   ```

**Setup:**
1. The `.env` file is at project root with placeholder values
2. Replace with actual API keys as needed
3. Never commit `.env` to version control (it's in `.gitignore`)

### Data Privacy
- Raw Bloomberg data retained locally; do not expose proprietary time-series
- Strip price columns before any web publication
- Dropbox path suggests personal/single-user environment

---

## Common Issues & Recovery

### Resume Interrupted Classification
Classification scripts save progress every 50 items:
```python
# If classify_etfs_full.py is interrupted, check for:
# "ETF Master List Classified PROGRESS.xlsx"
# Resume by re-running the script (it will overwrite with complete data)
```

### Memory Issues
- Close unnecessary applications (128GB recommended)
- Reduce batch sizes if needed
- Scripts process data in batches of 200 for API calls

### API Rate Limits
- Classification includes automatic rate limiting
- Exponential backoff implemented for retries
- Runtime: 30-60 minutes for full classification

---

## CRITICAL: Step 4 Report Generation Lessons Learned

### YTD Data in LLM Reports

**Problem (2026-01-31):** Reports were missing YTD columns in tables despite the user requesting YTD context. The LLM was not including YTD because:
1. The **data passed to the LLM** didn't include YTD - the `get_tier1_stats()` and `get_tier2_stats()` functions only queried `category_stats` table which has no YTD
2. Even if prompted, LLMs cannot invent data they weren't given

**Solution:** Modified `get_tier1_stats()` and `get_tier2_stats()` in `03_generate_daily_report.py` to:
1. Query `daily_prices` table joined with `assets` to get `AVG(return_ytd)` grouped by tier1/tier2
2. Include YTD column in the markdown tables passed to the LLM
3. The prompt template (`daily_wrap_structured.md`) explicitly specifies YTD as a required table column

**Key Principle:** 
> **If you want the LLM to include specific data in its output, you MUST pass that data in the input. LLMs cannot hallucinate real market data.**

### Two-Stage Workflow (PC + Mac)

**Problem:** Bloomberg API (`blpapi`) only works on Windows with Bloomberg Terminal.

**Solution:**
1. **PC Side (Windows/Parallels):** Run `bloomberg_daily.py` and `bloomberg_backfill.py` to fetch data into `market_data.db`
2. **Mac Side:** Run `run_pipeline.py` which calls correlation computation and report generation

**Critical Files:**
- `bloomberg_backfill.py` - Fetches historical data (90 days). Only fetches `CHG_PCT_1D`, NOT `CHG_PCT_YTD`
- `bloomberg_daily.py` - Fetches daily data. Fetches BOTH `CHG_PCT_1D` and `CHG_PCT_YTD`

**Implication:** Historical YTD data must be calculated or will only be available from daily runs going forward.

### Git Recovery Protocol

**Problem:** Destructive `git reset --hard` operations lost files and database state.

**Solution - Never Do:**
- `git reset --hard` to commits before current work
- Deleting files without checking their dependencies

**Solution - Always Do:**
1. Use `git checkout <commit-hash> -- <file-path>` to selectively restore specific files
2. Verify database schema/data after any git operation
3. Check for missing tables: `asset_correlations`, etc.

### Database Schema Critical Tables

The `market_data.db` requires these tables:
```sql
-- Core tables
assets              -- Static asset data from Final 1000
daily_prices        -- Daily returns (return_1d, return_ytd)
factor_returns      -- 18 factor returns per day
category_stats      -- Tier1/Tier2 aggregated stats per day

-- Derived tables  
asset_correlations  -- 60-day rolling correlations (computed by 04_compute_correlations.py)
```

If `asset_correlations` is missing, run:
```bash
python scripts/04_compute_correlations.py --backfill
```

---

## Claude Code Permissions

The `.claude/settings.local.json` file contains permissions for this project:
- File system access to `/Users/macbook2024/**` and Dropbox paths
- Bash command execution for Python scripts
- Anthropic API key access for classification tasks

---

## Dependencies Reference

| Library | Purpose | Install |
|---------|---------|---------|
| pandas | Excel manipulation | `pip install pandas openpyxl` |
| numpy | Numerical computation | `pip install numpy` |
| scikit-learn | Clustering algorithms | `pip install scikit-learn` |
| scipy | Distance calculations | `pip install scipy` |
| anthropic | Haiku 4.5 LLM | `pip install anthropic` |
| exa-py | Web search API | `pip install exa-py` |
| yfinance | Yahoo Finance data | `pip install yfinance` |
| gs-quant | Goldman Sachs API | `pip install gs-quant` |
| matplotlib | Visualization | `pip install matplotlib seaborn` |

---

## File Naming Conventions

- Input files: `*.xlsx` (Excel format)
- Output files: `* Classified.xlsx`, `* Classified PROGRESS.xlsx`
- Analysis outputs: `P0_*.xlsx`, `P1_*.xlsx`, `P2_*.xlsx`
- JSON data: `search_based_meme_stocks.json`
- Documentation: `README.md`, `CLAUDE.md`, `PRD.md`

---

## Contact & Support

For technical issues:
1. Check individual script documentation headers
2. Review console output for specific error messages
3. Verify API credentials and network connectivity
4. Ensure sufficient system resources (memory, disk space)

---

## Learned User Preferences

- Verify pipeline changes by actually running the end-to-end reports and comparing old vs new generated output ("the proof of the pudding") — never claim success without produced artifacts.
- Reports must be visually polished, institutional-grade PDFs; sparse or ugly report output is unacceptable and will be rejected.
- The daily report targets a hedge-fund IC audience: highlight unusual patterns, reference history, be willing to challenge consensus, and stay mainly informational (actionable only in unusual circumstances).
- Never name Goldman Sachs or Bloomberg in generated reports — refer only to categories and themes.
- Daily reports run after market close; on weekends/holidays use the last trading day's data rather than failing.
- Once given a green light, build and test all phases autonomously without prompting, but leave a readable status report of what was done.
- Check broker connectivity at the start of every report run and prompt the user in the loop when the Schwab token is stale or IBKR/TWS is not running.
- Do not enforce any maximum file-length rule — the user explicitly deleted the old 500-line-limit rule.

## Learned Workspace Facts

- The `rearchitect` branch contains the new unified daily report system: a single command `python3 report/main.py` (~2.5 min run) replaces the legacy Phase 0 → Step 4 → Phase 2 chain; outputs land in `outputs/unified/`.
- The daily universe is now ETF-only (~808 Yahoo Finance tickers in `data/universe.xlsx` with taxonomy and 15 factor ETFs) — no Bloomberg terminal dependency for daily reports.
- A single SQLite database `data/report.db` replaces `market_data.db` + `portfolio.db` in the new system.
- IBKR access requires the `.venv-ibkr312` Python 3.12 venv; `ib_insync.reqAccountUpdates` hangs forever on this multi-account setup — use one-shot `reqPositions()` + `accountSummary()` instead.
- Schwab tokens live in `~/.schwabdev/tokens.db`: access tokens auto-refresh (~30 min) but the refresh token expires every 7 days and requires interactive browser re-auth; credentials are stored in `.env` as `SCHWAB_APP_KEY`/`SCHWAB_APP_SECRET`.
- Holdings are pulled live from Schwab (7 accounts) and IBKR (3 accounts, including short positions) on every report run after preflight checks; VIX futures positions are tagged `VIX.FUT` so they are never priced as stocks.
- PrinceXML is the chosen PDF engine (HTML/CSS → print-quality PDF) for report generation.
- The system is built for personal use only (no redistribution), which avoids Bloomberg data licensing constraints.

---

*This AGENTS.md file is intended for AI coding agents. For human documentation, see README.md files in each directory.*
