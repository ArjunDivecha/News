# ETF Classifier - Inference Utilities

Batch and single-fund classification using the fine-tuned Llama-3.1-8B model.

## Model Performance

| Metric | Accuracy |
|--------|----------|
| Tier-1 | **95.7%** |
| Tier-2 | **85.6%** |
| Tier-3 (Exact Match) | **84.6%** |

## Prerequisites

- Python 3.11+ with virtualenv
- Tinker API access with valid `TINKER_API_KEY`
- Fine-tuned model checkpoint available

## Quick Start

### 0. Setup

```bash
cd "/Users/arjundivecha/Dropbox/AAA Backup/A Working/News/fine tuning"
source venv/bin/activate

# Set your Tinker API key
export TINKER_API_KEY=your_key_here
```

### 1. Single Fund Classification

```bash
# Basic usage
python scripts/predict_single.py --name "Vanguard S&P 500 ETF" --ticker "VOO"

# With more context
python scripts/predict_single.py \
  --name "iShares MSCI Emerging Markets ETF" \
  --ticker "EEM" \
  --asset-class "Equity" \
  --geo "Emerging Markets"

# Demo mode (no API call)
python scripts/predict_single.py --name "Test Fund" --demo
```

### 2. Batch Classification (Excel)

```bash
# Basic usage
python scripts/predict_funds.py input.xlsx output.xlsx

# With options
python scripts/predict_funds.py input.xlsx output.xlsx --batch-size 50

# Demo mode (simulated predictions)
python scripts/predict_funds.py input.xlsx output.xlsx --demo
```

## Input Format

Your Excel file should have these columns (same as ETF Master List):

| Column | Required | Description |
|--------|----------|-------------|
| `Name` | ✅ Yes | Fund name |
| `Ticker` | No | Ticker symbol |
| `Bloomberg` | No | Bloomberg ticker |
| `FUND_ASSET_CLASS_FOCUS` | No | Asset class (Equity, Fixed Income, etc.) |
| `FUND_GEO_FOCUS` | No | Geographic focus |
| `FUND_OBJECTIVE_LONG` | No | Fund objective description |
| `FUND_STRATEGY` | No | Strategy description |
| `STYLE_ANALYSIS_REGION_FOCUS` | No | Region focus for style analysis |

## Output Format

The output Excel will have all original columns plus:

| New Column | Description |
|------------|-------------|
| `Predicted_Tier1` | One of: Equities, Fixed Income, Commodities, Currencies, Multi-Asset, Volatility, Alternative |
| `Predicted_Tier2` | Sub-category (e.g., "Global Indices", "Sector Indices", "Sovereign Bonds") |
| `Predicted_Tier3` | Comma-separated tags (Asset Class, Region, Sector, Strategy, etc.) |

## Examples

### Example 1: Single Fund

```bash
python scripts/predict_single.py \
  --name "ARK Innovation ETF" \
  --ticker "ARKK" \
  --asset-class "Equity" \
  --objective "Long-term growth"
```

**Expected Output:**
```
Tier 1: Equities
Tier 2: Thematic/Factor
Tier 3: Equity, US, Tech, Growth, Thematic
```

### Example 2: Batch Processing

```bash
# Create sample input
python -c "
import pandas as pd
df = pd.DataFrame([
    {'Name': 'SPDR S&P 500 ETF', 'Ticker': 'SPY US', 'FUND_ASSET_CLASS_FOCUS': 'Equity'},
    {'Name': 'iShares 20+ Year Treasury Bond ETF', 'Ticker': 'TLT US', 'FUND_ASSET_CLASS_FOCUS': 'Fixed Income'},
    {'Name': 'Invesco QQQ Trust', 'Ticker': 'QQQ US', 'FUND_ASSET_CLASS_FOCUS': 'Equity'},
])
df.to_excel('sample_input.xlsx', index=False)
"

# Run classification
python scripts/predict_funds.py sample_input.xlsx classified_output.xlsx
```

## Model Path

Default model (fine-tuned on 4,208 ETF samples):
```
tinker://3a0f33d5-93be-5c30-b756-c183909f1804:train:0/sampler_weights/final
```

To use a different checkpoint:
```bash
python scripts/predict_funds.py input.xlsx output.xlsx \
  --model-path "tinker://.../checkpoints/step_1000"
```

## Troubleshooting

### Import Error
```bash
# Make sure you're in the right directory
cd "/Users/arjundivecha/Dropbox/AAA Backup/A Working/News/fine tuning"
source venv/bin/activate
```

### Tinker Connection Error
- Ensure Tinker service is running
- Check model path is correct
- Verify checkpoint exists

### Parse Errors
If the model returns non-JSON responses, the script will:
1. Mark Tier-1 as "Unknown" or best-effort extracted
2. Continue processing other funds
3. Report total parse errors in summary

## Performance Notes

- Inference speed: ~1.4-1.6s per fund
- Batch of 500 funds: ~12-15 minutes
- Memory: Uses ~2GB GPU memory for 8B model with LoRA

## Taxonomy Reference

### Tier-1 Categories
1. **Equities** - Stock indices, equity ETFs, equity-focused baskets
2. **Fixed Income** - Bonds, credit, yield-focused instruments
3. **Commodities** - Energy, metals, agriculture
4. **Currencies (FX)** - Currency pairs and FX instruments
5. **Multi-Asset / Thematic** - Cross-asset, thematic baskets
6. **Volatility / Risk Premia** - VIX, volatility indices, carry strategies
7. **Alternative / Synthetic** - Quant baskets, factor portfolios

### Tier-2 Examples
- Equities: Global Indices, Sector Indices, Country/Regional, Thematic/Factor
- Fixed Income: Sovereign Bonds, Corporate Credit, Credit Spreads
- Commodities: Energy, Metals, Agriculture
- Currencies: Majors, EM FX, Broad Currency
