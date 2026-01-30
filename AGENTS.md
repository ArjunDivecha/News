# AGENTS.md - Financial Analytics Pipeline

**Last Updated**: 2026-01-30  
**Project Root**: `/Users/arjundivecha/Dropbox/AAA Backup/A Working/News`

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

*This AGENTS.md file is intended for AI coding agents. For human documentation, see README.md files in each directory.*
