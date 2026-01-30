# Financial Asset Classification Pipeline

A multi-phase data pipeline that transforms raw multi-asset market data (ETFs, Bloomberg Indices, Goldman Sachs baskets) into a curated dataset of ~1,000 diversified assets for news analysis and market monitoring.

## Overview

| Stage | Purpose | Output |
|-------|---------|--------|
| **Step 1: Data Collection** | Raw data acquisition | Filtered datasets from 3 sources |
| **Step 2: Data Processing** | Classification & selection | Final 1000 Asset Master List |
| **Step 3: Data Analysis** | Performance analytics | Factor profiles & deduplication |
| **Fine Tuning** | ML model training | Fine-tuned Llama classifier |

## Quick Start

```bash
# Install dependencies
pip install pandas numpy scipy scikit-learn openpyxl anthropic

# Run full pipeline (45-80 minutes)
cd "Step 1 Data Collection"
python Bloomberg_Indices_Cluster.py
python "ETF Cluster.py"
python gs_basket_data.py

cd "../Step 2 Data Processing - Final1000"
python classify_bloomberg_full.py
python classify_etfs_full.py
python classify_goldman_full.py
python merge_classified_files.py
python final_selection_algorithm_v2.py
```

## Data Sources

| Source | Raw Count | After Filtering | Description |
|--------|-----------|-----------------|-------------|
| Bloomberg Indices | ~1,000+ | 438 | Market indices with performance metrics |
| ETF Master List | 1,619 | 500 | Exchange-traded funds with metadata |
| Goldman Sachs Baskets | 2,667 | 2,667 | Thematic baskets via Marquee API |
| Thematic ETFs | 231 | 231 | Specialized thematic ETFs |
| **Total** | **~5,500** | **~4,933** | Merged into master list |

## Final Output

**Final 1000 Asset Master List.xlsx** — ~970 diversified assets with:

| Tier-1 Category | Target Allocation |
|-----------------|-------------------|
| Equities | 52% (~520) |
| Fixed Income | 17% (~170) |
| Commodities | 11% (~110) |
| Currencies (FX) | 7% (~70) |
| Multi-Asset / Thematic | 8% (~80) |
| Volatility / Risk Premia | 5% (~50) |
| Alternative / Synthetic | 2% (~20) |

## Project Structure

```
News/
├── Step 1 Data Collection/          # Raw data acquisition
├── Step 2 Data Processing - Final1000/  # Classification & selection
├── Step 3 Data Analysis/            # Performance analytics
├── fine tuning/                     # ML model training
├── AGENTS.md                        # AI agent instructions
└── README.md                        # This file
```

## Classification Methods

### 1. Haiku Classification (Original)
- Uses Anthropic Claude Haiku 4.5 for taxonomy assignment
- Cost: ~$50-100 per full run
- Runtime: 30-60 minutes

### 2. Fine-Tuned Llama (New)
- Fine-tuned Llama 3.1 8B via Tinker
- Cost: $0 per inference run
- Runtime: ~22 minutes for 970 assets
- Location: `fine tuning/`

## Selection Algorithm

**Version 2** (`final_selection_algorithm_v2.py`) includes:
- Proxy Sharpe calculation for assets missing performance data
- Source quotas (15% Goldman, 5% Bloomberg minimum per category)
- Thematic rarity scoring for diversity
- Improved Goldman representation: 13% → 23%

## Requirements

- **Python**: 3.11+
- **Memory**: 8GB+ (128GB recommended)
- **APIs**: Anthropic, Goldman Sachs Marquee (optional), Exa (optional)

## Documentation

| File | Purpose |
|------|---------|
| `AGENTS.md` | AI coding agent instructions |
| `Step 1/README.md` | Data collection details |
| `Step 2/README.md` | Classification workflow |
| `Step 2/CLAUDE.md` | Architecture overview |
| `Step 2/PRD.md` | Product requirements |
| `fine tuning/README.md` | ML training guide |
| `fine tuning/INFERENCE.md` | Model inference guide |

---

**Last Updated**: 2026-01-30  
**Version**: 2.0.0
