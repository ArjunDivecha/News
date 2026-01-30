#!/usr/bin/env python3
"""
=============================================================================
FUND CLASSIFIER - BATCH INFERENCE UTILITY
=============================================================================

INPUT FILES:
- Excel file with fund data (same format as ETF Master List.xlsx)
  Required columns: Name, Ticker, Bloomberg, FUND_ASSET_CLASS_FOCUS, 
  FUND_GEO_FOCUS, FUND_OBJECTIVE_LONG, FUND_STRATEGY, STYLE_ANALYSIS_REGION_FOCUS

OUTPUT FILES:
- Excel file with predicted classifications
  Added columns: Predicted_Tier1, Predicted_Tier2, Predicted_Tier3, 
                 Confidence_Score (if available)

USAGE:
    python predict_funds.py input.xlsx output.xlsx
    python predict_funds.py input.xlsx output.xlsx --model-path <path>

VERSION HISTORY:
v1.0.0 (2026-01-29): Initial release

PURPOSE:
Batch classify funds using the fine-tuned Llama-3.1-8B ETF classifier.
=============================================================================
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

# Add parent directory to path for imports
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


# Model path (fine-tuned ETF classifier)
DEFAULT_MODEL_PATH = "tinker://3a0f33d5-93be-5c30-b756-c183909f1804:train:0/sampler_weights/final"

# System prompt (same as training)
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


def build_prompt(row: pd.Series) -> str:
    """Build the classification prompt from a dataframe row."""
    
    # Extract fields with fallbacks for missing data
    name = str(row.get('Name', '')).strip()
    ticker = str(row.get('Ticker', '')).strip()
    bloomberg = str(row.get('Bloomberg', '')).strip()
    asset_class = str(row.get('FUND_ASSET_CLASS_FOCUS', '')).strip()
    geo_focus = str(row.get('FUND_GEO_FOCUS', '')).strip()
    objective = str(row.get('FUND_OBJECTIVE_LONG', '')).strip()
    strategy = str(row.get('FUND_STRATEGY', '')).strip()
    style_region = str(row.get('STYLE_ANALYSIS_REGION_FOCUS', '')).strip()
    
    # Build prompt
    prompt_parts = [f"Asset Name: {name}"]
    
    if ticker and ticker != 'nan':
        prompt_parts.append(f"Ticker: {ticker}")
    if bloomberg and bloomberg != 'nan':
        prompt_parts.append(f"Bloomberg: {bloomberg}")
    if asset_class and asset_class != 'nan':
        prompt_parts.append(f"Asset Class Focus: {asset_class}")
    if geo_focus and geo_focus != 'nan':
        prompt_parts.append(f"Geographic Focus: {geo_focus}")
    if objective and objective != 'nan':
        prompt_parts.append(f"Objective: {objective}")
    if strategy and strategy != 'nan':
        prompt_parts.append(f"Strategy: {strategy}")
    if style_region and style_region != 'nan':
        prompt_parts.append(f"Style Region: {style_region}")
    
    return "\n".join(prompt_parts)


def parse_response(response_text: str) -> dict:
    """Parse the model response into structured classification."""
    
    try:
        # Clean up the response
        text = response_text.strip()
        
        # Remove markdown code blocks if present
        if text.startswith('```json'):
            text = text[7:]
        elif text.startswith('```'):
            text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
        
        text = text.strip()
        
        # Parse JSON
        result = json.loads(text)
        
        return {
            'tier_1': result.get('tier_1', 'Unknown'),
            'tier_2': result.get('tier_2', 'Unknown'),
            'tier_3': result.get('tier_3', [])
        }
    except json.JSONDecodeError:
        # Fallback: try to extract using simple heuristics
        text = response_text.lower()
        
        # Try to find tier_1
        tier_1 = 'Unknown'
        tier_1_options = ['equities', 'fixed income', 'commodities', 'currencies', 
                         'multi-asset', 'volatility', 'alternative']
        for option in tier_1_options:
            if option in text:
                tier_1 = option.title()
                break
        
        return {
            'tier_1': tier_1,
            'tier_2': 'Unknown',
            'tier_3': [],
            'parse_error': True
        }


def classify_fund(sampling_client, prompt: str, max_tokens: int = 256) -> dict:
    """Classify a single fund using the fine-tuned model."""
    
    # Build Llama-3 formatted prompt
    rendered = build_llama3_prompt(SYSTEM_PROMPT, prompt)
    
    # Sample
    params = types.SamplingParams(max_tokens=max_tokens, temperature=0.0)
    future = sampling_client.sample(prompt=rendered, sampling_params=params, num_samples=1)
    result = future.result()
    
    # Decode response
    response_text = ''.join([chr(t) for t in result.sequences[0].tokens])
    
    # Parse
    return parse_response(response_text)


def validate_input(df: pd.DataFrame) -> tuple[bool, list]:
    """Validate that the input dataframe has required columns."""
    
    required = ['Name']
    recommended = ['Ticker', 'Bloomberg', 'FUND_ASSET_CLASS_FOCUS', 
                   'FUND_GEO_FOCUS', 'FUND_OBJECTIVE_LONG', 'FUND_STRATEGY']
    
    missing_required = [col for col in required if col not in df.columns]
    missing_recommended = [col for col in recommended if col not in df.columns]
    
    if missing_required:
        return False, [f"Missing required columns: {missing_required}"]
    
    warnings = []
    if missing_recommended:
        warnings.append(f"Missing recommended columns: {missing_recommended}")
    
    return True, warnings


def main():
    parser = argparse.ArgumentParser(
        description='Classify funds using fine-tuned ETF classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_funds.py input.xlsx output.xlsx
  python predict_funds.py funds.xlsx classified.xlsx --model-path <custom_path>
  python predict_funds.py input.xlsx output.xlsx --batch-size 50
        """
    )
    
    parser.add_argument('input', help='Input Excel file with fund data')
    parser.add_argument('output', help='Output Excel file for predictions')
    parser.add_argument('--model-path', default=DEFAULT_MODEL_PATH,
                       help='Tinker model path (default: fine-tuned ETF classifier)')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Progress reporting interval (default: 100)')
    parser.add_argument('--max-tokens', type=int, default=256,
                       help='Max tokens for generation (default: 256)')
    parser.add_argument('--sheet-name', default=0,
                       help='Sheet name or index to read (default: 0)')
    parser.add_argument('--demo', action='store_true',
                       help='Demo mode - simulate without API calls')
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {args.input}")
        sys.exit(1)
    
    print("="*70)
    print("FUND CLASSIFIER - BATCH INFERENCE")
    print("="*70)
    
    # Load input data
    print(f"\n📥 Loading input: {args.input}")
    try:
        df = pd.read_excel(args.input, sheet_name=args.sheet_name)
        print(f"   Loaded {len(df)} rows")
    except Exception as e:
        print(f"❌ Error loading input: {e}")
        sys.exit(1)
    
    # Validate columns
    valid, warnings = validate_input(df)
    if not valid:
        print(f"❌ Validation failed: {warnings}")
        sys.exit(1)
    
    if warnings:
        print(f"⚠️  Warnings: {warnings}")
    
    # Check for demo mode or API key
    if args.demo:
        print(f"\n🎮 DEMO MODE - Simulating predictions (no API calls)")
        sampling_client = None
    else:
        if not os.environ.get('TINKER_API_KEY'):
            print("\n❌ Error: TINKER_API_KEY environment variable not set")
            print("   Set it with: export TINKER_API_KEY=your_key_here")
            print("   Or run in demo mode: --demo")
            sys.exit(1)
        
        # Initialize model
        print(f"\n🔌 Connecting to Tinker...")
        print(f"   Model: {args.model_path}")
        
        try:
            service_client = tinker.ServiceClient()
            sampling_client = service_client.create_sampling_client(
                model_path=args.model_path
            )
            print("   ✅ Connected")
        except Exception as e:
            print(f"❌ Error connecting to Tinker: {e}")
            sys.exit(1)
    
    # Run inference (or simulation)
    if args.demo:
        print(f"\n🎮 Simulating predictions for {len(df)} funds...")
    else:
        print(f"\n🚀 Running inference on {len(df)} funds...")
    
    predictions = []
    parse_errors = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Classifying"):
        try:
            if args.demo:
                # Demo mode: simulate predictions based on asset class
                asset_class = str(row.get('FUND_ASSET_CLASS_FOCUS', '')).lower()
                if 'equity' in asset_class or 'equities' in asset_class:
                    result = {'tier_1': 'Equities', 'tier_2': 'Global Indices', 'tier_3': ['Equity', 'Passive']}
                elif 'fixed income' in asset_class or 'bond' in asset_class:
                    result = {'tier_1': 'Fixed Income', 'tier_2': 'Sovereign Bonds', 'tier_3': ['Credit', 'Government']}
                elif 'commodity' in asset_class:
                    result = {'tier_1': 'Commodities', 'tier_2': 'Broad Commodities', 'tier_3': ['Commodity', 'Diversified']}
                else:
                    result = {'tier_1': 'Equities', 'tier_2': 'Global Indices', 'tier_3': ['Equity']}
            else:
                # Build prompt
                prompt = build_prompt(row)
                
                # Classify
                result = classify_fund(sampling_client, prompt, args.max_tokens)
                
                # Track parse errors
                if result.get('parse_error'):
                    parse_errors += 1
            
            predictions.append({
                'Predicted_Tier1': result['tier_1'],
                'Predicted_Tier2': result['tier_2'],
                'Predicted_Tier3': ', '.join(result['tier_3']) if result['tier_3'] else ''
            })
            
        except Exception as e:
            print(f"\n⚠️  Error on row {idx}: {e}")
            predictions.append({
                'Predicted_Tier1': 'ERROR',
                'Predicted_Tier2': 'ERROR',
                'Predicted_Tier3': ''
            })
    
    # Add predictions to dataframe
    pred_df = pd.DataFrame(predictions)
    output_df = pd.concat([df, pred_df], axis=1)
    
    # Save output
    print(f"\n💾 Saving results to: {args.output}")
    try:
        output_df.to_excel(args.output, index=False)
        print("   ✅ Saved")
    except Exception as e:
        print(f"❌ Error saving output: {e}")
        sys.exit(1)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total funds processed: {len(df)}")
    print(f"Parse errors: {parse_errors}")
    print(f"\nTier-1 Distribution:")
    print(pred_df['Predicted_Tier1'].value_counts().to_string())
    print(f"\n✅ Results saved to: {args.output}")


if __name__ == "__main__":
    main()
