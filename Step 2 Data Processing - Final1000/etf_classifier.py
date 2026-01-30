"""
=============================================================================
ETF CLASSIFIER WITH HAIKU 4.5 - Large-Scale ETF Taxonomy Classification
=============================================================================

INPUT FILES:
- /Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/News/ETF Master List.xlsx
  Description: Complete ETF database with 1,619 ETFs and comprehensive fund metadata
  Required Format: Excel file with complete ETF information including descriptions and classifications
  Key Columns: Ticker, Name, CIE_DES, FUND_ASSET_CLASS_FOCUS, FUND_GEO_FOCUS, FUND_OBJECTIVE_LONG, FUND_STRATEGY

OUTPUT FILES:
- /Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/News/ETF Master List Classified.xlsx
  Description: ETF database with added taxonomy classifications (Tier-1, Tier-2, Tier-3 tags)
  Format: Excel file preserving all original columns plus new classification columns
  Contents: Original ETF data + category_tier1, category_tier2, category_tags columns

- /Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/News/ETF Master List Classified PROGRESS.xlsx
  Description: Intermediate checkpoint file saved every 50 ETFs during processing
  Format: Excel file with partial classifications for recovery/resume capability
  Contents: Same structure as final output but with in-progress classifications

VERSION HISTORY:
v1.0.0 (2025-10-16): Initial release with basic ETF classification
v1.1.0 (2025-10-17): Added batch processing and progress saving
v1.2.0 (2025-11-06): Enhanced documentation and error handling

DEPENDENCIES:
- pandas
- anthropic (for Haiku API)
- ANTHROPIC_API_KEY environment variable

USAGE:
python etf_classifier.py

TAXONOMY:
Tier-1 (7 categories):
  1. Equities
  2. Fixed Income
  3. Commodities
  4. Currencies (FX)
  5. Multi-Asset / Thematic
  6. Volatility / Risk Premia
  7. Alternative / Synthetic

Tier-2 (examples):
  Equities: Global Indices, Sector Indices, Country/Regional, Thematic/Factor
  Fixed Income: Sovereign Bonds, Corporate Credit, Credit Spreads, Yield Curves
  Commodities: Energy, Metals, Agriculture
  etc.

Tier-3 Tags (all applicable):
  - Asset Class: Equity | Credit | FX | Commodity | Multi-Asset
  - Region: US | Europe | Asia | EM | Global
  - Sector/Theme: Tech | Energy | Financials | Healthcare | Industrials | Consumer | Defensive | ESG | Momentum | Value | Growth | Dividend
  - Strategy: Active | Passive | Equal-Weight | Thematic | Quantitative
  - Duration (bonds): Short (<2Y) | Medium (2-10Y) | Long (>10Y)

NOTES:
- Processes ETFs in batches with rate limiting
- Saves intermediate results every 50 ETFs
- Uses JSON parsing for structured output

=============================================================================
"""

import pandas as pd
import json
import os
import time
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Anthropic client
client = Anthropic()

# Taxonomy for the prompt
TAXONOMY = """
TIER-1 CATEGORIES (top-level asset class):
1. Equities - Stock indices, ETFs, equity-focused baskets
2. Fixed Income - Bonds, credit, yield-focused instruments
3. Commodities - Energy, metals, agriculture
4. Currencies (FX) - Currency pairs and FX instruments
5. Multi-Asset / Thematic - Cross-asset, thematic baskets, macro themes
6. Volatility / Risk Premia - VIX, volatility indices, carry strategies
7. Alternative / Synthetic - Quantitative baskets, factor portfolios, proprietary constructs

TIER-2 CATEGORIES (by sub-class/structure):
Equities: Global Indices | Sector Indices | Country/Regional | Thematic/Factor
Fixed Income: Sovereign Bonds | Corporate Credit | Credit Spreads | Yield Curves
Commodities: Energy | Metals | Agriculture
Currencies: Majors (EUR/USD, GBP/USD, JPY/USD) | EM FX
Multi-Asset: Cross-Asset Indices | Inflation/Growth Themes
Volatility: Vol Indices | Carry/Value Factors
Alternative: Quant/Style Baskets | Custom/Proprietary

TIER-3 TAGS (all applicable - multi-select):
Asset Class: Equity | Credit | FX | Commodity | Multi-Asset
Region: US | Europe | Asia | EM | Global
Sector/Theme: Tech | Energy | Financials | Healthcare | Industrials | Consumer | Defensive | ESG | Dividend | Growth | Value | Momentum | Quality | Infrastructure | Real Estate | Utilities
Strategy: Active | Passive | Equal-Weight | Thematic | Quantitative | Options-Based | Dividend | Factor-Based
Duration (bonds only): Short (<2Y) | Medium (2-10Y) | Long (>10Y)
Special: Inverse | Long-Short | Covered-Call (NOTE: Most of these should already be removed)
"""

SYSTEM_PROMPT = f"""You are an expert ETF classification specialist. Your job is to classify ETFs into a structured taxonomy.

{TAXONOMY}

For each ETF, return a JSON object with exactly these fields:
{{
  "ticker": "ETF ticker",
  "tier1": "One of: Equities, Fixed Income, Commodities, Currencies (FX), Multi-Asset / Thematic, Volatility / Risk Premia, Alternative / Synthetic",
  "tier2": "Sub-category appropriate for the tier1 (see examples above)",
  "tier3_tags": ["tag1", "tag2", "tag3"] - list of all applicable tags from the taxonomy
}}

Be precise and use ONLY the categories and tags from the taxonomy above. If unsure, choose the closest match."""

def classify_etf(ticker: str, name: str, description: str, asset_class: str, geo_focus: str, objective: str, strategy: str) -> dict:
    """Classify a single ETF using Haiku 4.5."""

    user_message = f"""Classify this ETF:

Ticker: {ticker}
Name: {name}
Description: {description}
Existing Asset Class: {asset_class}
Geographic Focus: {geo_focus}
Objective: {objective}
Strategy: {strategy}

Respond with ONLY valid JSON, no other text."""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )

        # Parse the JSON response
        response_text = response.content[0].text.strip()

        # Try to extract JSON if wrapped in markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        result = json.loads(response_text)
        return result

    except json.JSONDecodeError as e:
        print(f"  ❌ JSON parse error for {ticker}: {e}")
        print(f"     Response: {response_text[:100]}")
        return None
    except Exception as e:
        print(f"  ❌ API error for {ticker}: {e}")
        return None

def classify_etfs(input_file: str, output_file: str, start_row: int = 0):
    """Classify all ETFs and save results."""

    print("Loading ETF data...")
    df = pd.read_excel(input_file)
    print(f"Total ETFs to classify: {len(df)}\n")

    # Add columns for classification
    df['category_tier1'] = None
    df['category_tier2'] = None
    df['category_tags'] = None

    # Classify each ETF
    for idx in range(start_row, len(df)):
        row = df.iloc[idx]
        ticker = row['Ticker']

        # Show progress
        if (idx + 1) % 10 == 0:
            print(f"Progress: {idx + 1}/{len(df)}")

        # Get classification
        result = classify_etf(
            ticker=ticker,
            name=row['Name'],
            description=row['CIE_DES'],
            asset_class=row['FUND_ASSET_CLASS_FOCUS'],
            geo_focus=row['FUND_GEO_FOCUS'],
            objective=row['FUND_OBJECTIVE_LONG'],
            strategy=row['FUND_STRATEGY']
        )

        if result:
            df.at[idx, 'category_tier1'] = result.get('tier1', 'Unknown')
            df.at[idx, 'category_tier2'] = result.get('tier2', 'Unknown')
            tags = result.get('tier3_tags', [])
            df.at[idx, 'category_tags'] = ', '.join(tags) if isinstance(tags, list) else tags
        else:
            df.at[idx, 'category_tier1'] = 'Unclassified'
            df.at[idx, 'category_tier2'] = 'Unclassified'
            df.at[idx, 'category_tags'] = ''

        # Save progress every 50 ETFs
        if (idx + 1) % 50 == 0:
            print(f"  Saving progress...")
            df.to_excel(output_file, index=False)

    # Final save
    df.to_excel(output_file, index=False)
    print(f"\n✓ Classification complete!")
    print(f"✓ Saved to: {output_file}")

    # Print summary
    print("\n" + "="*80)
    print("CLASSIFICATION SUMMARY:")
    print("="*80)
    print("\nTier-1 Distribution:")
    print(df['category_tier1'].value_counts())

    print("\nTop 10 Tier-2 Categories:")
    print(df['category_tier2'].value_counts().head(10))

    print("\nSample Classifications:")
    for idx in range(min(5, len(df))):
        row = df.iloc[idx]
        print(f"\n{row['Ticker']} ({row['Name'][:40]}):")
        print(f"  Tier-1: {row['category_tier1']}")
        print(f"  Tier-2: {row['category_tier2']}")
        print(f"  Tags: {row['category_tags'][:80]}...")

if __name__ == "__main__":
    input_file = "ETF Master List.xlsx"
    output_file = "ETF Master List Classified.xlsx"

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found")
        exit(1)

    classify_etfs(input_file, output_file)
