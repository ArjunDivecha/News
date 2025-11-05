"""
=============================================================================
UNIFIED ASSET CLASSIFIER - ETFs, Bloomberg Indices, Goldman Baskets
=============================================================================

PURPOSE:
Classify assets from 3 sources using unified Haiku 4.5 LLM taxonomy.
Handles different data structures and missing fields gracefully.

SOURCES:
1. ETFs (1619) - CIE_DES descriptions
2. Bloomberg Indices (438) - LONG_COMP_NAME descriptions
3. Goldman Baskets (2667) - description field (1807 with content)

UPDATED TAXONOMY:

TIER-1 CATEGORIES (7):
1. Equities
2. Fixed Income
3. Commodities
4. Currencies (FX)
5. Multi-Asset / Thematic
6. Volatility / Risk Premia
7. Alternative / Synthetic

TIER-2 CATEGORIES (updated):
Equities: Global Indices | Sector Indices | Country/Regional | Thematic/Factor | Real Estate / REITs
Fixed Income: Sovereign Bonds | Corporate Credit | Credit Spreads | Yield Curves
Commodities: Energy | Metals | Agriculture
Currencies: Majors | EM FX
Multi-Asset: Cross-Asset Indices | Inflation/Growth Themes
Volatility: Vol Indices | Carry/Value Factors
Alternative: Quant/Style Baskets | Custom/Proprietary

TIER-3 TAGS (updated with expanded regions & themes):
Asset Class: Equity | Credit | FX | Commodity | Multi-Asset
Region: US | Europe | Asia | EM | Global | China | Japan | India | Canada | UAE | APAC | Australia
Sector/Theme: Tech | Energy | Financials | Healthcare | Industrials | Consumer | Defensive | ESG | Dividend | Growth | Value | Momentum | Quality | Infrastructure | Real Estate | Utilities
Strategy: Active | Passive | Equal-Weight | Thematic | Quantitative | Options-Based | Dividend | Factor-Based | Low Volatility | Defensive | Domestic
Special Themes: Stimulus | Going Global | Long/Short

=============================================================================
"""

import pandas as pd
import json
import os
from anthropic import Anthropic

client = Anthropic()

TAXONOMY = """
TIER-1 CATEGORIES (top-level asset class):
1. Equities - Stock indices, ETFs, equity-focused baskets, REITs
2. Fixed Income - Bonds, credit, yield-focused instruments
3. Commodities - Energy, metals, agriculture
4. Currencies (FX) - Currency pairs and FX instruments
5. Multi-Asset / Thematic - Cross-asset, thematic baskets, macro themes
6. Volatility / Risk Premia - VIX, volatility indices, carry strategies
7. Alternative / Synthetic - Quantitative baskets, factor portfolios, proprietary constructs

TIER-2 CATEGORIES (by sub-class/structure):
Equities: Global Indices | Sector Indices | Country/Regional | Thematic/Factor | Real Estate / REITs
Fixed Income: Sovereign Bonds | Corporate Credit | Credit Spreads | Yield Curves
Commodities: Energy | Metals | Agriculture
Currencies: Majors (EUR/USD, GBP/USD, JPY/USD) | EM FX
Multi-Asset: Cross-Asset Indices | Inflation/Growth Themes
Volatility: Vol Indices | Carry/Value Factors
Alternative: Quant/Style Baskets | Custom/Proprietary

TIER-3 TAGS (all applicable - multi-select):
Asset Class: Equity | Credit | FX | Commodity | Multi-Asset
Region: US | Europe | Asia | EM | Global | China | Japan | India | Canada | UAE | APAC | Australia
Sector/Theme: Tech | Energy | Financials | Healthcare | Industrials | Consumer | Defensive | ESG | Dividend | Growth | Value | Momentum | Quality | Infrastructure | Real Estate | Utilities
Strategy: Active | Passive | Equal-Weight | Thematic | Quantitative | Options-Based | Dividend | Factor-Based | Low Volatility | Defensive | Domestic
Special Themes: Stimulus | Going Global | Long/Short
Duration (bonds only): Short (<2Y) | Medium (2-10Y) | Long (>10Y)
"""

SYSTEM_PROMPT = f"""You are an expert asset classification specialist. Your job is to classify financial assets (ETFs, indices, baskets) into a unified taxonomy.

{TAXONOMY}

For each asset, return a JSON object with exactly these fields:
{{
  "identifier": "Ticker or name",
  "tier1": "One of: Equities, Fixed Income, Commodities, Currencies (FX), Multi-Asset / Thematic, Volatility / Risk Premia, Alternative / Synthetic",
  "tier2": "Sub-category appropriate for the tier1 (see examples above)",
  "tier3_tags": ["tag1", "tag2", "tag3"] - list of ALL applicable tags from the taxonomy
}}

Be precise and use ONLY the categories and tags from the taxonomy above. If unsure, choose the closest match.
For assets with no description, use the name/identifier and any available metadata to make best guess."""

def classify_asset(identifier: str, name: str, description: str, source: str, metadata: dict = None) -> dict:
    """Classify a single asset using Haiku 4.5.

    Args:
        identifier: Ticker or code
        name: Full asset name
        description: Long description (or None/empty)
        source: "ETF" | "Bloomberg" | "Goldman"
        metadata: Optional dict with security_type, region, etc.
    """

    # Build metadata string
    metadata_str = ""
    if metadata:
        if metadata.get('security_type'):
            metadata_str += f"Security Type: {metadata['security_type']}\n"
        if metadata.get('region'):
            metadata_str += f"Region: {metadata['region']}\n"
        if metadata.get('asset_class'):
            metadata_str += f"Asset Class: {metadata['asset_class']}\n"
        if metadata.get('objective'):
            metadata_str += f"Objective: {metadata['objective']}\n"
        if metadata.get('strategy'):
            metadata_str += f"Strategy: {metadata['strategy']}\n"

    user_message = f"""Classify this asset from {source}:

Identifier: {identifier}
Name: {name}
{metadata_str}
Description: {description if description else '(No description available)'}

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

        response_text = response.content[0].text.strip()

        # Extract JSON if wrapped in markdown
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        result = json.loads(response_text)
        return result

    except json.JSONDecodeError as e:
        print(f"  ❌ JSON parse error for {identifier}: {e}")
        return None
    except Exception as e:
        print(f"  ❌ API error for {identifier}: {e}")
        return None

def test_classifier():
    """Test classifier with mixed assets from all 3 sources."""

    print("="*80)
    print("UNIFIED ASSET CLASSIFIER - TEST RUN")
    print("="*80)

    results = []

    # Test ETFs (first 15)
    print("\n[1/3] Testing ETFs (15 samples)...")
    etf_df = pd.read_excel("/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/News/ETF Master List.xlsx")

    for idx in range(min(15, len(etf_df))):
        row = etf_df.iloc[idx]
        print(f"  [{idx+1}/15] {row['Ticker']}...", end=" ", flush=True)

        result = classify_asset(
            identifier=row['Ticker'],
            name=row['Name'],
            description=row['CIE_DES'],
            source="ETF",
            metadata={
                'asset_class': row['FUND_ASSET_CLASS_FOCUS'],
                'region': row['FUND_GEO_FOCUS'],
                'objective': row['FUND_OBJECTIVE_LONG'],
                'strategy': row['FUND_STRATEGY']
            }
        )

        if result:
            results.append({**result, 'source': 'ETF'})
            print(f"✓ {result.get('tier1')}")
        else:
            print("✗ Error")

    # Test Bloomberg (first 15)
    print("\n[2/3] Testing Bloomberg Indices (15 samples)...")
    bloomberg_df = pd.read_excel("/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/News/Filtered Bloomberg Indices.xlsx")

    for idx in range(min(15, len(bloomberg_df))):
        row = bloomberg_df.iloc[idx]
        print(f"  [{idx+1}/15] {row['Ticker']}...", end=" ", flush=True)

        result = classify_asset(
            identifier=row['Ticker'],
            name=row['Index Name'],
            description=row['LONG_COMP_NAME'],
            source="Bloomberg",
            metadata={
                'security_type': row['SECURITY_TYP'],
                'region': row['REGION_OR_COUNTRY']
            }
        )

        if result:
            results.append({**result, 'source': 'Bloomberg'})
            print(f"✓ {result.get('tier1')}")
        else:
            print("✗ Error")

    # Test Goldman (first 20)
    print("\n[3/3] Testing Goldman Baskets (20 samples)...")
    goldman_df = pd.read_excel("/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/News/Data Collection/GSCB_FLAGSHIP_coverage_with_desc_ALL.xlsx")

    for idx in range(min(20, len(goldman_df))):
        row = goldman_df.iloc[idx]
        ticker = row['Index Name']
        desc = row['description'] if pd.notna(row['description']) else ""

        print(f"  [{idx+1}/20] {ticker[:20]}...", end=" ", flush=True)

        result = classify_asset(
            identifier=ticker,
            name=ticker,
            description=str(desc),
            source="Goldman",
            metadata={
                'region': row['REGION_OR_COUNTRY']
            }
        )

        if result:
            results.append({**result, 'source': 'Goldman'})
            print(f"✓ {result.get('tier1')}")
        else:
            print("✗ Error")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_excel("Unified Classifier Test Results.xlsx", index=False)

    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"\nTotal classified: {len(results)} assets")

    print("\nTier-1 Distribution:")
    print(results_df['tier1'].value_counts())

    print("\nBy Source:")
    print(results_df['source'].value_counts())

    print("\nTier-1 by Source:")
    cross = pd.crosstab(results_df['source'], results_df['tier1'])
    print(cross)

    print("\nSample Classifications:")
    for source in ['ETF', 'Bloomberg', 'Goldman']:
        sample = results_df[results_df['source'] == source].head(1)
        if len(sample) > 0:
            row = sample.iloc[0]
            print(f"\n{source}:")
            print(f"  ID: {row['identifier']}")
            print(f"  Tier-1: {row['tier1']}")
            print(f"  Tier-2: {row['tier2']}")
            print(f"  Tags: {str(row['tier3_tags'])[:80]}...")

    print(f"\n✓ Full results saved to: Unified Classifier Test Results.xlsx")

if __name__ == "__main__":
    test_classifier()
