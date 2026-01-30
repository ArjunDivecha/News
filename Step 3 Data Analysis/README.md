# Step 3: Data Analysis

Performance analytics, deduplication, and factor decomposition for the classified asset master list.

## Overview

This stage analyzes the ~4,933 classified assets to:
1. Identify duplicate/proxy assets via correlation clustering
2. Analyze performance by category and source
3. Decompose factor exposures for portfolio construction

## Scripts

### p0_deduplication_analysis.py
**Purpose**: Identify highly correlated assets that are essentially duplicates/proxies.

**Input**: `Master Asset List Classified.xlsx` (from Step 2)  
**Output**: 
- `Dedup Report - Top 30 Groups.xlsx` — Top proxy groups for review
- `Dedup Report - Full Dataset.xlsx` — Complete clustering results

**Method**: 
- Uses 30-dimensional beta vectors (market betas, sector betas, country betas)
- Hierarchical clustering with correlation threshold > 0.9
- Groups similar assets and recommends centroids for retention

### p1_category_analysis.py
**Purpose**: Analyze performance metrics across categories and sources.

**Input**: `Master Asset List Classified.xlsx`  
**Outputs**:
- `P1_Tier1_Analysis.xlsx` — Performance by Tier-1 category
- `P1_Tier2_Analysis.xlsx` — Performance by Tier-2 subcategory
- `P1_Source_Analysis.xlsx` — Distribution by data source

**Metrics**: Average returns, Sharpe ratios, volatility, coverage gaps

### p2_factor_decomposition.py
**Purpose**: Extract factor exposures for ML-ready feature matrix.

**Input**: `Master Asset List Classified.xlsx`  
**Output**: `P2_Enriched_Asset_Profiles.xlsx` (4,934 assets × 19 analytical columns)

**Factors**:
- Market beta (overall equity exposure)
- Size tilt (large/mid/small cap)
- Geographic exposure (US, Europe, Asia, EM)
- Sector concentrations
- Volatility characteristics

### MemeFinder.py
**Purpose**: Discover trending meme stocks using web search and sentiment analysis.

**Input**: None (web-based data retrieval via Exa API)  
**Output**: `search_based_meme_stocks.json`

**Method**: 
- Searches for recent Reddit/Twitter mentions of stock symbols
- Validates via Yahoo Finance for tradability
- Outputs ranked list with sentiment scores

## Data Files

| File | Description |
|------|-------------|
| `Betas from Bloomberg.xlsx` | Reference beta data for factor analysis |
| `Dedup Report - Top 30 Groups.xlsx` | Top proxy groups |
| `Dedup Report - Full Dataset.xlsx` | Complete clustering results |
| `P1_Tier1_Analysis.xlsx` | Tier-1 category performance |
| `P1_Tier2_Analysis.xlsx` | Tier-2 subcategory performance |
| `P1_Source_Analysis.xlsx` | Source distribution analysis |
| `P2_Enriched_Asset_Profiles.xlsx` | Factor-enriched profiles |
| `search_based_meme_stocks.json` | Meme stock discoveries |

## Usage

```bash
cd "Step 3 Data Analysis"

# Run analysis scripts (assumes Step 2 complete)
python p1_category_analysis.py    # Category performance
python p0_deduplication_analysis.py  # Proxy detection
python p2_factor_decomposition.py  # Factor exposures

# Optional: Meme stock discovery
python MemeFinder.py
```

## Variants

- `p0_deduplication_analysis_fast.py` — Quick version with reduced iterations
- `p0_deduplication_analysis_full.py` — Full version with all assets

## Dependencies

```bash
pip install pandas numpy scipy scikit-learn openpyxl
pip install exa-py yfinance  # For MemeFinder
```

## Notes

- Analysis scripts read from Step 2 outputs
- Results feed into final selection algorithm
- Factor profiles can be used for ML-based portfolio optimization

---

**Last Updated**: 2026-01-30
