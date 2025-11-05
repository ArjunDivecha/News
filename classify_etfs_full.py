"""Classify all 1619 ETFs - PRESERVE ALL ORIGINAL COLUMNS"""
import pandas as pd
import json
import os
from anthropic import Anthropic

client = Anthropic()

TAXONOMY = """
TIER-1 CATEGORIES:
1. Equities | 2. Fixed Income | 3. Commodities | 4. Currencies (FX) | 5. Multi-Asset / Thematic | 6. Volatility / Risk Premia | 7. Alternative / Synthetic

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

SYSTEM_PROMPT = f"""Classify assets into this taxonomy:
{TAXONOMY}

Return ONLY valid JSON: {{"identifier":"X", "tier1":"Y", "tier2":"Z", "tier3_tags":["a","b","c"]}}"""

def classify_etf(ticker, name, description, metadata):
    msg = f"""Classify ETF:
Ticker: {ticker}
Name: {name}
Description: {description}
Asset Class: {metadata.get('asset_class')}
Region: {metadata.get('region')}
Objective: {metadata.get('objective')}
Strategy: {metadata.get('strategy')}

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
print("CLASSIFYING 1619 ETFs - PRESERVING ALL ORIGINAL DATA")
print("="*80)

df = pd.read_excel("ETF Master List.xlsx")
print(f"\nOriginal columns: {len(df.columns)}")
print(f"Original rows: {len(df)}\n")

# Add new classification columns
df['category_tier1'] = None
df['category_tier2'] = None
df['category_tags'] = None

for idx in range(len(df)):
    row = df.iloc[idx]
    if (idx + 1) % 50 == 0:
        print(f"Progress: {idx + 1}/1619 - Saving checkpoint...")
        df.to_excel("ETF Master List Classified PROGRESS.xlsx", index=False)

    result = classify_etf(
        row['Ticker'],
        row['Name'],
        row['CIE_DES'],
        {'asset_class': row['FUND_ASSET_CLASS_FOCUS'], 'region': row['FUND_GEO_FOCUS'],
         'objective': row['FUND_OBJECTIVE_LONG'], 'strategy': row['FUND_STRATEGY']}
    )

    if result:
        df.at[idx, 'category_tier1'] = result.get('tier1', 'Unknown')
        df.at[idx, 'category_tier2'] = result.get('tier2', 'Unknown')
        tags = result.get('tier3_tags', [])
        df.at[idx, 'category_tags'] = ', '.join(tags) if isinstance(tags, list) else str(tags)

df.to_excel("ETF Master List Classified.xlsx", index=False)
print(f"\n✓ ETF CLASSIFICATION COMPLETE")
print(f"  File: ETF Master List Classified.xlsx")
print(f"  Rows: {len(df)} (all original data preserved)")
print(f"  New columns: category_tier1, category_tier2, category_tags")
print(f"  Total columns: {len(df.columns)}")
print(f"\nTier-1 Distribution:")
print(df['category_tier1'].value_counts().to_dict())
