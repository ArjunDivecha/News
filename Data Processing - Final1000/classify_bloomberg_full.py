"""
=============================================================================
CLASSIFY BLOOMBERG INDICES FULL - Large-Scale Index Taxonomy Classification
=============================================================================

INPUT FILES:
- /Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/News/Filtered Bloomberg Indices.xlsx
  Description: Filtered Bloomberg indices database with 438 entries and comprehensive metadata
  Required Format: Excel file with index identifiers, descriptions, and security classifications
  Key Columns: Ticker, Index Name, LONG_COMP_NAME, SECURITY_TYP, REGION_OR_COUNTRY

OUTPUT FILES:
- /Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/News/Filtered Bloomberg Indices Classified.xlsx
  Description: Bloomberg indices with added taxonomy classifications (Tier-1, Tier-2, Tier-3 tags)
  Format: Excel file preserving all original columns plus new classification columns
  Contents: Original index data + category_tier1, category_tier2, category_tags columns

- /Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/News/Filtered Bloomberg Indices Classified PROGRESS.xlsx
  Description: Intermediate checkpoint file saved every 50 indices during processing
  Format: Excel file with partial classifications for recovery/resume capability
  Contents: Same structure as final output but with in-progress classifications

VERSION HISTORY:
v1.0.0 (2025-10-16): Initial release with basic index classification
v1.1.0 (2025-10-17): Added batch processing and progress saving
v1.2.0 (2025-11-06): Enhanced documentation and error handling

TAXONOMY:
TIER-1: Equities | Fixed Income | Commodities | Currencies (FX) | Multi-Asset / Thematic | Volatility / Risk Premia | Alternative / Synthetic

TIER-2 EXAMPLES:
Equities: Global Indices | Sector Indices | Country/Regional | Thematic/Factor | Real Estate / REITs
Fixed Income: Sovereign Bonds | Corporate Credit | Credit Spreads | Yield Curves
Commodities: Energy | Metals | Agriculture
Currencies: Majors | EM FX
Multi-Asset: Cross-Asset Indices | Inflation/Growth Themes
Volatility: Vol Indices | Carry/Value Factors
Alternative: Quant/Style Baskets | Custom/Proprietary

TIER-3 TAGS:
Asset Class: Equity | Credit | FX | Commodity | Multi-Asset
Region: US | Europe | Asia | EM | Global | China | Japan | India | Canada | UAE | APAC | Australia
Sector/Theme: Tech | Energy | Financials | Healthcare | Industrials | Consumer | Defensive | ESG | Dividend | Growth | Value | Momentum | Quality | Infrastructure | Real Estate | Utilities
Strategy: Active | Passive | Equal-Weight | Thematic | Quantitative | Options-Based | Dividend | Factor-Based | Low Volatility | Defensive | Domestic
Special: Stimulus | Going Global | Long/Short
Duration: Short (<2Y) | Medium (2-10Y) | Long (>10Y)
"""
import pandas as pd
import json
import os
from anthropic import Anthropic

client = Anthropic()

TAXONOMY = """
TIER-1: Equities | Fixed Income | Commodities | Currencies (FX) | Multi-Asset / Thematic | Volatility / Risk Premia | Alternative / Synthetic

TIER-2 EXAMPLES:
Equities: Global Indices | Sector Indices | Country/Regional | Thematic/Factor | Real Estate / REITs
Fixed Income: Sovereign Bonds | Corporate Credit | Credit Spreads | Yield Curves
Commodities: Energy | Metals | Agriculture
Currencies: Majors | EM FX
Multi-Asset: Cross-Asset Indices | Inflation/Growth Themes
Volatility: Vol Indices | Carry/Value Factors
Alternative: Quant/Style Baskets | Custom/Proprietary

TIER-3 TAGS:
Asset Class: Equity | Credit | FX | Commodity | Multi-Asset
Region: US | Europe | Asia | EM | Global | China | Japan | India | Canada | UAE | APAC | Australia
Sector/Theme: Tech | Energy | Financials | Healthcare | Industrials | Consumer | Defensive | ESG | Dividend | Growth | Value | Momentum | Quality | Infrastructure | Real Estate | Utilities
Strategy: Active | Passive | Equal-Weight | Thematic | Quantitative | Options-Based | Dividend | Factor-Based | Low Volatility | Defensive | Domestic
Special: Stimulus | Going Global | Long/Short
Duration: Short (<2Y) | Medium (2-10Y) | Long (>10Y)
"""

SYSTEM_PROMPT = f"""Classify Bloomberg indices into this taxonomy:
{TAXONOMY}

Return ONLY valid JSON: {{"identifier":"X", "tier1":"Y", "tier2":"Z", "tier3_tags":["a","b","c"]}}"""

def classify_index(ticker, index_name, description, security_type, region):
    msg = f"""Classify Bloomberg Index:
Ticker: {ticker}
Index Name: {index_name}
Description: {description}
Security Type: {security_type}
Region: {region}

Return ONLY JSON."""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": msg}]
        )
        text = response.content[0].text.strip()
        if "```" in text:
            text = text.split("```")[1].split("```")[0].strip() if "```json" not in text else text.split("```json")[1].split("```")[0].strip()
        return json.loads(text)
    except Exception as e:
        print(f"Error {ticker}: {str(e)[:50]}")
        return None

print("="*80)
print("CLASSIFYING 438 BLOOMBERG INDICES - PRESERVING ALL ORIGINAL DATA")
print("="*80)

# Input file validation
input_file = "Filtered Bloomberg Indices.xlsx"
if not os.path.exists(input_file):
    print(f"Error: Input file {input_file} not found")
    exit(1)

df = pd.read_excel(input_file)
print(f"\nOriginal columns: {len(df.columns)}")
print(f"Original rows: {len(df)}\n")

# Add new classification columns
df['category_tier1'] = None
df['category_tier2'] = None
df['category_tags'] = None

for idx in range(len(df)):
    row = df.iloc[idx]
    if (idx + 1) % 50 == 0:
        print(f"Progress: {idx + 1}/438 - Saving checkpoint...")
        df.to_excel("Filtered Bloomberg Indices Classified PROGRESS.xlsx", index=False)

    result = classify_index(
        row['Ticker'],
        row['Index Name'],
        row['LONG_COMP_NAME'],
        row['SECURITY_TYP'],
        row['REGION_OR_COUNTRY']
    )

    if result:
        df.at[idx, 'category_tier1'] = result.get('tier1', 'Unknown')
        df.at[idx, 'category_tier2'] = result.get('tier2', 'Unknown')
        tags = result.get('tier3_tags', [])
        df.at[idx, 'category_tags'] = ', '.join(tags) if isinstance(tags, list) else str(tags)

df.to_excel("Filtered Bloomberg Indices Classified.xlsx", index=False)
print(f"\n✓ BLOOMBERG CLASSIFICATION COMPLETE")
print(f"  File: Filtered Bloomberg Indices Classified.xlsx")
print(f"  Rows: {len(df)} (all original data preserved)")
print(f"  New columns: category_tier1, category_tier2, category_tags")
print(f"  Total columns: {len(df.columns)}")
print(f"\nTier-1 Distribution:")
print(df['category_tier1'].value_counts().to_dict())
