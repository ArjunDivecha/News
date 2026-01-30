#!/usr/bin/env python3
"""
=============================================================================
SINGLE FUND CLASSIFIER - QUICK INFERENCE
=============================================================================

Quickly classify a single fund without creating an Excel file.

USAGE:
    python predict_single.py --name "Vanguard S&P 500 ETF" --ticker "VOO"
    python predict_single.py --name "iShares MSCI Emerging Markets" --asset-class "Equity" --geo "Emerging Markets"

VERSION HISTORY:
v1.0.0 (2026-01-29): Initial release

PURPOSE:
Quick single-fund classification for testing or ad-hoc use.
=============================================================================
"""

import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import tinker
from tinker import types


def build_llama3_prompt(system_prompt: str, user_prompt: str) -> str:
    """Build Llama-3 chat format prompt manually."""
    return (
        f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


DEFAULT_MODEL_PATH = "tinker://3a0f33d5-93be-5c30-b756-c183909f1804:train:0/sampler_weights/final"

SYSTEM_PROMPT = """You are an expert financial asset classification specialist. Classify the given asset into a structured taxonomy.

TIER-1 CATEGORIES (pick exactly one):
- Equities: Stock indices, equity ETFs, equity-focused baskets, equity indices
- Fixed Income: Bonds, credit, yield-focused instruments, fixed income ETFs
- Commodities: Energy, metals, agriculture, commodity indices
- Currencies (FX): Currency pairs and FX instruments
- Multi-Asset / Thematic: Cross-asset, thematic baskets, macro themes, multi-asset indices
- Volatility / Risk Premia: VIX, volatility indices, carry strategies, risk premia
- Alternative / Synthetic: Quantitative baskets, factor portfolios, proprietary constructs, custom indices

TIER-2 CATEGORIES (examples by Tier-1):
- Equities: Global Indices | Sector Indices | Country/Regional | Thematic/Factor | Real Estate / REITs
- Fixed Income: Sovereign Bonds | Corporate Credit | Credit Spreads | Yield Curves | Broad Fixed Income
- Commodities: Energy | Metals | Agriculture | Broad Commodities
- Currencies: Majors (EUR/USD, GBP/USD, USD/JPY) | EM FX | Broad Currency
- Multi-Asset: Cross-Asset Indices | Inflation/Growth Themes | Thematic Baskets
- Volatility: Vol Indices | Carry/Value Factors | Risk Premia Strategies
- Alternative: Quant/Style Baskets | Custom/Proprietary | Factor-Based

TIER-3 TAGS (select all that apply from):
- Asset Class: Equity | Credit | FX | Commodity | Multi-Asset | Volatility | Alternative
- Region: US | Europe | Asia | EM | Global | China | Japan | India | Canada | UAE | APAC | Australia | Developed
- Sector/Theme: Tech | Energy | Financials | Healthcare | Industrials | Consumer | Defensive | ESG | Dividend | Growth | Value | Momentum | Quality | Infrastructure | Real Estate | Utilities | Materials
- Strategy: Active | Passive | Equal-Weight | Thematic | Quantitative | Options-Based | Dividend | Factor-Based | Low Volatility | Defensive | Domestic | International | Diversified
- Duration/Credit: Investment Grade | High Yield | Short Duration | Long Duration | IG Credit | HY Credit

RESPOND ONLY with a JSON object in this exact format:
{
  "tier_1": "<one of the 7 Tier-1 categories>",
  "tier_2": "<descriptive sub-category>",
  "tier_3": ["tag1", "tag2", "tag3"]
}

No explanation, no markdown formatting, just the JSON."""


def classify(name: str, ticker: str = "", asset_class: str = "", 
             geo_focus: str = "", objective: str = "", strategy: str = "",
             model_path: str = DEFAULT_MODEL_PATH) -> dict:
    """Classify a single fund."""
    
    # Build prompt
    prompt_parts = [f"Asset Name: {name}"]
    if ticker:
        prompt_parts.append(f"Ticker: {ticker}")
    if asset_class:
        prompt_parts.append(f"Asset Class Focus: {asset_class}")
    if geo_focus:
        prompt_parts.append(f"Geographic Focus: {geo_focus}")
    if objective:
        prompt_parts.append(f"Objective: {objective}")
    if strategy:
        prompt_parts.append(f"Strategy: {strategy}")
    
    user_prompt = "\n".join(prompt_parts)
    
    # Initialize model
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(model_path=model_path)
    
    # Build Llama-3 formatted prompt
    rendered = build_llama3_prompt(SYSTEM_PROMPT, user_prompt)
    
    # Sample
    params = types.SamplingParams(max_tokens=256, temperature=0.0)
    future = sampling_client.sample(prompt=rendered, sampling_params=params, num_samples=1)
    result = future.result()
    
    # Decode
    response_text = ''.join([chr(t) for t in result.sequences[0].tokens])
    
    # Parse
    try:
        text = response_text.strip()
        if text.startswith('```json'):
            text = text[7:]
        elif text.startswith('```'):
            text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
        return json.loads(text.strip())
    except:
        return {"raw_response": response_text}


def main():
    parser = argparse.ArgumentParser(description='Classify a single fund')
    parser.add_argument('--name', required=True, help='Fund name')
    parser.add_argument('--ticker', default='', help='Ticker symbol')
    parser.add_argument('--asset-class', default='', help='Asset class focus')
    parser.add_argument('--geo', default='', help='Geographic focus')
    parser.add_argument('--objective', default='', help='Fund objective')
    parser.add_argument('--strategy', default='', help='Fund strategy')
    parser.add_argument('--model-path', default=DEFAULT_MODEL_PATH, help='Model path')
    parser.add_argument('--demo', action='store_true', help='Demo mode (no API call)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("SINGLE FUND CLASSIFIER")
    print("="*60)
    print(f"\nInput:")
    print(f"  Name: {args.name}")
    if args.ticker:
        print(f"  Ticker: {args.ticker}")
    if args.asset_class:
        print(f"  Asset Class: {args.asset_class}")
    if args.geo:
        print(f"  Geo Focus: {args.geo}")
    
    if args.demo:
        print("\n🎮 DEMO MODE - No API call")
        print("\n" + "="*60)
        print("EXPECTED RESULT")
        print("="*60)
        print(f"\nTier 1: Equities")
        print(f"Tier 2: Global Indices")
        print(f"Tier 3: Equity, US, Large Cap, Passive")
        return
    
    # Check for TINKER_API_KEY
    if not os.environ.get('TINKER_API_KEY'):
        print("\n❌ Error: TINKER_API_KEY environment variable not set")
        print("   Set it with: export TINKER_API_KEY=your_key_here")
        print("   Or run in demo mode: --demo")
        sys.exit(1)
    
    print(f"\nClassifying...")
    
    try:
        result = classify(
            name=args.name,
            ticker=args.ticker,
            asset_class=args.asset_class,
            geo_focus=args.geo,
            objective=args.objective,
            strategy=args.strategy,
            model_path=args.model_path
        )
    except Exception as e:
        print(f"\n❌ Classification failed: {e}")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("RESULT")
    print("="*60)
    print(f"\nTier 1: {result.get('tier_1', 'N/A')}")
    print(f"Tier 2: {result.get('tier_2', 'N/A')}")
    tier_3 = result.get('tier_3', [])
    if isinstance(tier_3, list):
        print(f"Tier 3: {', '.join(tier_3)}")
    else:
        print(f"Tier 3: {tier_3}")
    
    # Print full JSON
    print(f"\nFull Response:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
