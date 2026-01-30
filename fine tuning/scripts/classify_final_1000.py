#!/usr/bin/env python3
"""
=============================================================================
CLASSIFY FINAL 1000 - Fine-tuned Model vs Haiku Comparison
=============================================================================

INPUT FILES:
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/News/Step 2 Data Processing - Final1000/Final 1000 Asset Master List.xlsx

OUTPUT FILES:
- outputs/final_1000_classified.xlsx (with fine-tuned classifications)
- outputs/final_1000_comparison_report.xlsx (detailed comparison)

USAGE:
    python classify_final_1000.py
    python classify_final_1000.py --demo  # Test without API
    python classify_final_1000.py --limit 50  # Test with first 50

VERSION HISTORY:
v1.0.0 (2026-01-30): Initial release

PURPOSE:
Reclassify the Final 1000 Asset Master List using the fine-tuned Llama model
and compare with existing Haiku classifications.
=============================================================================
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Tinker (lazy import for demo mode compatibility)
try:
    import tinker
    from tinker import types
    TINKER_AVAILABLE = True
except ImportError:
    TINKER_AVAILABLE = False

# Default paths
INPUT_FILE = "../Step 2 Data Processing - Final1000/Final 1000 Asset Master List.xlsx"
OUTPUT_DIR = "outputs/final_1000"
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


def build_llama3_prompt(system_prompt: str, user_prompt: str) -> str:
    """Build Llama-3 chat format prompt."""
    return (
        f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def build_prompt_from_row(row: pd.Series) -> str:
    """Build classification prompt from Final 1000 list row."""

    prompt_parts = []

    # Name (required)
    name = str(row.get('Name', '')).strip()
    if name and name != 'nan':
        prompt_parts.append(f"Asset Name: {name}")

    # Bloomberg Ticker
    ticker = str(row.get('Bloomberg_Ticker', '')).strip()
    if ticker and ticker != 'nan':
        prompt_parts.append(f"Bloomberg Ticker: {ticker}")

    # Description (use Long_Description for context)
    description = str(row.get('Long_Description', '')).strip()
    if description and description != 'nan' and len(description) > 5:
        # Truncate long descriptions
        if len(description) > 500:
            description = description[:500] + "..."
        prompt_parts.append(f"Description: {description}")

    # Source (ETF, Goldman, Bloomberg) - useful context
    source = str(row.get('source', '')).strip()
    if source and source != 'nan':
        prompt_parts.append(f"Source: {source}")

    return "\n".join(prompt_parts)


def parse_response(response_text: str) -> dict:
    """Parse the model response into structured classification."""

    try:
        text = response_text.strip()

        # Remove markdown code blocks if present
        if text.startswith('```json'):
            text = text[7:]
        elif text.startswith('```'):
            text = text[3:]
        if text.endswith('```'):
            text = text[:-3]

        text = text.strip()
        result = json.loads(text)

        return {
            'tier_1': result.get('tier_1', 'Unknown'),
            'tier_2': result.get('tier_2', 'Unknown'),
            'tier_3': result.get('tier_3', []),
            'parse_error': False
        }
    except json.JSONDecodeError:
        # Fallback: try to extract tier_1 from text
        text = response_text.lower()

        tier_1 = 'Unknown'
        tier_1_options = {
            'equities': 'Equities',
            'fixed income': 'Fixed Income',
            'commodities': 'Commodities',
            'currencies': 'Currencies (FX)',
            'multi-asset': 'Multi-Asset / Thematic',
            'volatility': 'Volatility / Risk Premia',
            'alternative': 'Alternative / Synthetic'
        }

        for key, value in tier_1_options.items():
            if key in text:
                tier_1 = value
                break

        return {
            'tier_1': tier_1,
            'tier_2': 'Unknown',
            'tier_3': [],
            'parse_error': True,
            'raw_response': response_text[:200]
        }


def normalize_tier1(tier1: str) -> str:
    """Normalize Tier-1 category names for comparison."""
    if pd.isna(tier1):
        return 'Unknown'

    tier1 = str(tier1).strip()

    # Normalize common variations
    mappings = {
        'equities': 'Equities',
        'equity': 'Equities',
        'fixed income': 'Fixed Income',
        'fixed_income': 'Fixed Income',
        'commodities': 'Commodities',
        'commodity': 'Commodities',
        'currencies': 'Currencies (FX)',
        'currencies (fx)': 'Currencies (FX)',
        'fx': 'Currencies (FX)',
        'multi-asset': 'Multi-Asset / Thematic',
        'multi-asset / thematic': 'Multi-Asset / Thematic',
        'thematic': 'Multi-Asset / Thematic',
        'volatility': 'Volatility / Risk Premia',
        'volatility / risk premia': 'Volatility / Risk Premia',
        'alternative': 'Alternative / Synthetic',
        'alternative / synthetic': 'Alternative / Synthetic',
    }

    tier1_lower = tier1.lower()
    return mappings.get(tier1_lower, tier1)


def compute_agreement(haiku: str, finetuned: str) -> str:
    """Compute agreement level between Haiku and fine-tuned classifications."""

    h = normalize_tier1(haiku)
    f = normalize_tier1(finetuned)

    if h == f:
        return 'Exact Match'

    # Check for partial matches
    h_lower = h.lower()
    f_lower = f.lower()

    tier1_groups = {
        'equities': ['equities', 'equity'],
        'fixed_income': ['fixed income', 'bonds', 'credit'],
        'commodities': ['commodities', 'commodity'],
        'currencies': ['currencies', 'fx', 'currency'],
        'multi_asset': ['multi-asset', 'thematic'],
        'volatility': ['volatility', 'risk premia', 'vix'],
        'alternative': ['alternative', 'synthetic', 'quant']
    }

    for group, keywords in tier1_groups.items():
        h_in_group = any(kw in h_lower for kw in keywords)
        f_in_group = any(kw in f_lower for kw in keywords)
        if h_in_group and f_in_group:
            return 'Partial Match'

    return 'Mismatch'


def simulate_prediction(row: pd.Series) -> dict:
    """Simulate prediction for demo mode based on existing classification."""

    # Use existing classification with small random variations for demo
    tier1 = str(row.get('category_tier1', 'Equities'))
    tier2 = str(row.get('category_tier2', 'Global Indices'))

    # Simulate 90% agreement with existing classifications
    import random
    if random.random() < 0.90:
        return {'tier_1': tier1, 'tier_2': tier2, 'tier_3': ['Simulated']}
    else:
        # Random different classification
        alternatives = ['Equities', 'Fixed Income', 'Commodities', 'Multi-Asset / Thematic']
        return {'tier_1': random.choice(alternatives), 'tier_2': 'Simulated', 'tier_3': ['Demo']}


def generate_comparison_report(df: pd.DataFrame, output_dir: str):
    """Generate detailed comparison report between Haiku and fine-tuned model."""

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("GENERATING COMPARISON REPORT")
    print("=" * 70)

    # Compute agreement statistics
    df['haiku_tier1_norm'] = df['category_tier1'].apply(normalize_tier1)
    df['finetuned_tier1_norm'] = df['finetuned_tier1'].apply(normalize_tier1)

    tier1_match = (df['haiku_tier1_norm'] == df['finetuned_tier1_norm']).sum()
    tier1_match_pct = tier1_match / len(df) * 100

    tier2_match = (df['category_tier2'] == df['finetuned_tier2']).sum()
    tier2_match_pct = tier2_match / len(df) * 100

    # Agreement breakdown
    df['agreement'] = df.apply(
        lambda x: compute_agreement(x['category_tier1'], x['finetuned_tier1']),
        axis=1
    )

    agreement_counts = df['agreement'].value_counts()

    # Stats summary
    stats = {
        'Total Assets': len(df),
        'Tier-1 Exact Match': f"{tier1_match} ({tier1_match_pct:.1f}%)",
        'Tier-2 Exact Match': f"{tier2_match} ({tier2_match_pct:.1f}%)",
        'Partial Matches': agreement_counts.get('Partial Match', 0),
        'Mismatches': agreement_counts.get('Mismatch', 0),
        'Parse Errors': (df['finetuned_tier1'] == 'Parse Error').sum()
    }

    print(f"\n   Tier-1 Agreement: {stats['Tier-1 Exact Match']}")
    print(f"   Tier-2 Agreement: {stats['Tier-2 Exact Match']}")
    print(f"   Mismatches: {stats['Mismatches']}")

    # Source-level analysis
    print("\n📊 Agreement by Source:")
    for source in df['source'].unique():
        src_df = df[df['source'] == source]
        src_match = (src_df['haiku_tier1_norm'] == src_df['finetuned_tier1_norm']).sum()
        src_pct = src_match / len(src_df) * 100 if len(src_df) > 0 else 0
        print(f"   {source}: {src_match}/{len(src_df)} ({src_pct:.1f}%)")

    # Tier-1 distribution comparison
    haiku_dist = df['haiku_tier1_norm'].value_counts().to_frame('Haiku')
    finetuned_dist = df['finetuned_tier1_norm'].value_counts().to_frame('FineTuned')
    tier1_comparison = haiku_dist.join(finetuned_dist, how='outer').fillna(0).astype(int)
    tier1_comparison['Difference'] = tier1_comparison['FineTuned'] - tier1_comparison['Haiku']

    # Disagreements analysis
    disagreements = df[df['agreement'] == 'Mismatch'].copy()

    # Save Excel report
    report_path = os.path.join(output_dir, 'classification_comparison_report.xlsx')
    with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
        # Sheet 1: Full comparison
        comparison_cols = [
            'Bloomberg_Ticker', 'Name', 'source',
            'category_tier1', 'category_tier2', 'category_tags',
            'finetuned_tier1', 'finetuned_tier2', 'finetuned_tier3',
            'agreement'
        ]
        available_cols = [c for c in comparison_cols if c in df.columns]
        df[available_cols].to_excel(writer, sheet_name='Full_Comparison', index=False)

        # Sheet 2: Disagreements only
        if len(disagreements) > 0:
            disagreements[available_cols].to_excel(writer, sheet_name='Disagreements', index=False)

        # Sheet 3: Summary statistics
        stats_df = pd.DataFrame([stats]).T.reset_index()
        stats_df.columns = ['Metric', 'Value']
        stats_df.to_excel(writer, sheet_name='Summary', index=False)

        # Sheet 4: Tier-1 distribution
        tier1_comparison.to_excel(writer, sheet_name='Tier1_Distribution')

        # Sheet 5: By source
        source_agreement = []
        for source in df['source'].unique():
            src_df = df[df['source'] == source]
            src_match = (src_df['haiku_tier1_norm'] == src_df['finetuned_tier1_norm']).sum()
            source_agreement.append({
                'Source': source,
                'Total': len(src_df),
                'Tier1_Match': src_match,
                'Match_Rate': f"{src_match/len(src_df)*100:.1f}%"
            })
        pd.DataFrame(source_agreement).to_excel(writer, sheet_name='By_Source', index=False)

    print(f"\n   Saved: {report_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description='Classify Final 1000 list with fine-tuned model')
    parser.add_argument('--input', default=INPUT_FILE, help='Input Excel file')
    parser.add_argument('--output-dir', default=OUTPUT_DIR, help='Output directory')
    parser.add_argument('--model-path', default=DEFAULT_MODEL_PATH, help='Tinker model path')
    parser.add_argument('--limit', type=int, default=None, help='Limit to first N assets (for testing)')
    parser.add_argument('--demo', action='store_true', help='Demo mode - simulate without API calls')

    args = parser.parse_args()

    print("=" * 70)
    print("FINAL 1000 CLASSIFICATION: Fine-Tuned Llama vs Haiku")
    print("=" * 70)

    # Load input
    print(f"\n📥 Loading input: {args.input}")
    try:
        df = pd.read_excel(args.input)
        print(f"   Loaded {len(df)} assets")
    except Exception as e:
        print(f"❌ Error loading input: {e}")
        sys.exit(1)

    # Limit for testing
    if args.limit:
        df = df.head(args.limit)
        print(f"   Limited to first {args.limit} assets")

    # Initialize model
    sampling_client = None
    tokenizer = None

    if args.demo:
        print(f"\n🎮 DEMO MODE - Simulating predictions")
    else:
        if not os.environ.get('TINKER_API_KEY'):
            print("\n❌ Error: TINKER_API_KEY not set")
            print("   Set with: export TINKER_API_KEY=your_key")
            print("   Or use --demo for simulation")
            sys.exit(1)

        # Load tokenizer
        print(f"\n🔌 Loading tokenizer...")
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("thinkingmachineslabinc/meta-llama-3-tokenizer")
            print("   ✅ Tokenizer loaded")
        except Exception as e:
            print(f"❌ Error loading tokenizer: {e}")
            sys.exit(1)

        print(f"\n🔌 Connecting to Tinker...")
        print(f"   Model: {args.model_path}")

        try:
            if not TINKER_AVAILABLE:
                raise ImportError("Tinker library not available")

            service_client = tinker.ServiceClient()
            sampling_client = service_client.create_sampling_client(model_path=args.model_path)
            print("   ✅ Connected")
        except Exception as e:
            print(f"❌ Error connecting to Tinker: {e}")
            sys.exit(1)

    # Run inference
    print(f"\n🚀 Classifying {len(df)} assets...")
    if not args.demo:
        print(f"   Estimated time: ~{len(df) * 1.5 / 60:.0f} minutes")

    predictions = []
    parse_errors = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Classifying"):
        try:
            if args.demo:
                result = simulate_prediction(row)
            else:
                # Build prompt
                prompt = build_prompt_from_row(row)
                rendered = build_llama3_prompt(SYSTEM_PROMPT, prompt)

                # Tokenize
                tokens = tokenizer.encode(rendered, add_special_tokens=False)
                prompt_input = types.ModelInput.from_ints(tokens)

                # Sample
                params = types.SamplingParams(max_tokens=256, temperature=0.0)
                future = sampling_client.sample(prompt=prompt_input, sampling_params=params, num_samples=1)
                response = future.result()

                # Decode and parse
                output_tokens = response.sequences[0].tokens
                response_text = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
                result = parse_response(response_text)

                if result.get('parse_error'):
                    parse_errors += 1

            predictions.append({
                'finetuned_tier1': result['tier_1'],
                'finetuned_tier2': result['tier_2'],
                'finetuned_tier3': ', '.join(result['tier_3']) if result['tier_3'] else ''
            })

        except Exception as e:
            print(f"\n⚠️  Error on row {idx}: {e}")
            predictions.append({
                'finetuned_tier1': 'ERROR',
                'finetuned_tier2': 'ERROR',
                'finetuned_tier3': ''
            })

    # Merge predictions
    print("\n📊 Merging predictions...")
    pred_df = pd.DataFrame(predictions)
    df = pd.concat([df, pred_df], axis=1)

    # Save classified output
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'Final_1000_FineTuned_Classified.xlsx')
    df.to_excel(output_path, index=False)
    print(f"   ✅ Saved: {output_path}")

    # Generate comparison report
    stats = generate_comparison_report(df, args.output_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total assets: {len(df)}")
    print(f"Parse errors: {parse_errors}")
    print(f"\nFine-tuned Tier-1 Distribution:")
    print(df['finetuned_tier1'].value_counts().to_string())

    print(f"\n✅ Classification complete!")
    print(f"   Output: {output_path}")
    print(f"   Report: {args.output_dir}/classification_comparison_report.xlsx")


if __name__ == "__main__":
    main()
