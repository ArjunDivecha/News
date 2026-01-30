#!/usr/bin/env python3
"""
=============================================================================
MODEL COMPARISON: Fine-Tuned Llama vs Haiku
=============================================================================

Compares classifications between Haiku (baseline) and fine-tuned Llama-3.1-8B.

INPUT FILES:
- ETF Master List Classified.xlsx (with Haiku classifications)

OUTPUT FILES:
- ETF_Master_List_FineTuned.xlsx (with Llama classifications)
- model_comparison_report.xlsx (detailed comparison)

USAGE:
    python compare_models.py
    python compare_models.py --limit 100  # Test with first 100

VERSION HISTORY:
v1.0.0 (2026-01-29): Initial release

PURPOSE:
Generate comprehensive comparison between Haiku and fine-tuned model.
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
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

import tinker
from tinker import types


DEFAULT_MODEL_PATH = "tinker://3a0f33d5-93be-5c30-b756-c183909f1804:train:0/sampler_weights/final"
INPUT_FILE = "../Step 2 Data Processing - Final1000/ETF Master List Classified.xlsx"
OUTPUT_DIR = "outputs/comparison"

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


def build_prompt_llama3(system_msg: str, user_msg: str) -> str:
    """Build Llama-3 chat format prompt."""
    return (
        f"<|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def build_fund_prompt(row: pd.Series) -> str:
    """Build the classification prompt from a dataframe row."""
    name = str(row.get('Name', '')).strip()
    ticker = str(row.get('Ticker', '')).strip()
    bloomberg = str(row.get('Bloomberg', '')).strip()
    asset_class = str(row.get('FUND_ASSET_CLASS_FOCUS', '')).strip()
    geo_focus = str(row.get('FUND_GEO_FOCUS', '')).strip()
    objective = str(row.get('FUND_OBJECTIVE_LONG', '')).strip()
    strategy = str(row.get('FUND_STRATEGY', '')).strip()
    style_region = str(row.get('STYLE_ANALYSIS_REGION_FOCUS', '')).strip()
    
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
        text = response_text.strip()
        
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
        return {
            'tier_1': 'Parse Error',
            'tier_2': 'Parse Error',
            'tier_3': [],
            'parse_error': True,
            'raw_response': response_text[:200]
        }


def classify_fund(sampling_client, tokenizer, prompt: str, max_tokens: int = 256) -> dict:
    """Classify a single fund using the fine-tuned model."""
    
    rendered = build_prompt_llama3(SYSTEM_PROMPT, prompt)
    
    # Tokenize
    tokens = tokenizer.encode(rendered, add_special_tokens=False)
    prompt_input = types.ModelInput.from_ints(tokens)
    
    params = types.SamplingParams(max_tokens=max_tokens, temperature=0.0)
    future = sampling_client.sample(prompt=prompt_input, sampling_params=params, num_samples=1)
    result = future.result()
    
    output_tokens = result.sequences[0].tokens
    response_text = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
    
    return parse_response(response_text)


def normalize_category(cat: str) -> str:
    """Normalize category names for comparison."""
    if pd.isna(cat):
        return 'Unknown'
    return str(cat).strip()


def compute_agreement(haiku: str, llama: str) -> str:
    """Compute agreement level between two classifications."""
    h = normalize_category(haiku)
    l = normalize_category(llama)
    
    if h == l:
        return 'Exact Match'
    
    # Check for partial matches (e.g., both are Equities variants)
    h_lower = h.lower()
    l_lower = l.lower()
    
    # Tier-1 partial matches
    tier1_groups = {
        'equities': ['equities', 'equity'],
        'fixed_income': ['fixed income', 'bonds', 'credit', 'treasury'],
        'commodities': ['commodities', 'commodity'],
        'currencies': ['currencies', 'fx', 'currency'],
        'multi_asset': ['multi-asset', 'multi asset', 'thematic'],
        'volatility': ['volatility', 'risk premia', 'vix'],
        'alternative': ['alternative', 'synthetic', 'quant']
    }
    
    for group, keywords in tier1_groups.items():
        h_in_group = any(kw in h_lower for kw in keywords)
        l_in_group = any(kw in l_lower for kw in keywords)
        if h_in_group and l_in_group:
            return 'Partial Match'
    
    return 'Mismatch'


def generate_comparison_report(df: pd.DataFrame, output_dir: str):
    """Generate detailed comparison report."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("GENERATING COMPARISON REPORT")
    print("="*70)
    
    # 1. Overall Agreement Statistics
    print("\n📊 Computing agreement statistics...")
    
    tier1_agreement = (df['category_tier1'] == df['llama_tier1']).sum()
    tier2_agreement = (df['category_tier2'] == df['llama_tier2']).sum()
    
    tier1_agreement_pct = tier1_agreement / len(df) * 100
    tier2_agreement_pct = tier2_agreement / len(df) * 100
    
    # Compute partial matches for Tier-1
    partial_matches = df[df['agreement'] == 'Partial Match']
    
    stats = {
        'Total ETFs': len(df),
        'Tier-1 Exact Match': f"{tier1_agreement} ({tier1_agreement_pct:.1f}%)",
        'Tier-2 Exact Match': f"{tier2_agreement} ({tier2_agreement_pct:.1f}%)",
        'Tier-1 Partial Match': f"{len(partial_matches)} ({len(partial_matches)/len(df)*100:.1f}%)",
        'Parse Errors': (df['llama_tier1'] == 'Parse Error').sum()
    }
    
    print(f"\n   Tier-1 Exact Agreement: {stats['Tier-1 Exact Match']}")
    print(f"   Tier-2 Exact Agreement: {stats['Tier-2 Exact Match']}")
    
    # 2. Tier-1 Distribution Comparison
    print("\n📊 Comparing Tier-1 distributions...")
    
    haiku_dist = df['category_tier1'].value_counts().to_frame('Haiku_Count')
    llama_dist = df['llama_tier1'].value_counts().to_frame('Llama_Count')
    
    tier1_comparison = haiku_dist.join(llama_dist, how='outer').fillna(0).astype(int)
    tier1_comparison['Difference'] = tier1_comparison['Llama_Count'] - tier1_comparison['Haiku_Count']
    tier1_comparison['Pct_Change'] = (tier1_comparison['Difference'] / tier1_comparison['Haiku_Count'] * 100).round(1)
    
    # 3. Disagreement Analysis
    print("\n📊 Analyzing disagreements...")
    
    disagreements = df[df['agreement'] == 'Mismatch'].copy()
    
    # Create confusion matrix for Tier-1
    confusion = pd.crosstab(
        disagreements['category_tier1'], 
        disagreements['llama_tier1'],
        margins=True
    )
    
    # 4. Save detailed comparison file
    print("\n💾 Saving comparison files...")
    
    # Main comparison file
    comparison_cols = [
        'Name', 'Ticker', 'Bloomberg', 'FUND_ASSET_CLASS_FOCUS',
        'category_tier1', 'category_tier2', 'category_tags',
        'llama_tier1', 'llama_tier2', 'llama_tier3',
        'agreement', 'tier1_match', 'tier2_match'
    ]
    
    available_cols = [c for c in comparison_cols if c in df.columns]
    comparison_df = df[available_cols].copy()
    
    output_path = os.path.join(output_dir, 'ETF_Master_List_Comparison.xlsx')
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: Full comparison
        comparison_df.to_excel(writer, sheet_name='Full_Comparison', index=False)
        
        # Sheet 2: Only disagreements
        disagreements.to_excel(writer, sheet_name='Disagreements', index=False)
        
        # Sheet 3: Summary stats
        stats_df = pd.DataFrame([stats]).T.reset_index()
        stats_df.columns = ['Metric', 'Value']
        stats_df.to_excel(writer, sheet_name='Summary_Stats', index=False)
        
        # Sheet 4: Tier-1 distribution comparison
        tier1_comparison.to_excel(writer, sheet_name='Tier1_Distribution')
        
        # Sheet 5: Confusion matrix
        if len(confusion) > 0:
            confusion.to_excel(writer, sheet_name='Confusion_Matrix')
    
    print(f"   ✅ Saved: {output_path}")
    
    # 5. Generate text report
    report_path = os.path.join(output_dir, 'comparison_report.txt')
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MODEL COMPARISON REPORT\n")
        f.write("Fine-Tuned Llama-3.1-8B vs Haiku\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 70 + "\n")
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")
        
        f.write("\n\nTIER-1 DISTRIBUTION COMPARISON\n")
        f.write("-" * 70 + "\n")
        f.write(tier1_comparison.to_string())
        
        f.write("\n\nTOP DISAGREEMENTS (Haiku -> Llama)\n")
        f.write("-" * 70 + "\n")
        
        if len(disagreements) > 0:
            diag_pairs = disagreements.groupby(['category_tier1', 'llama_tier1']).size().sort_values(ascending=False).head(20)
            for (h_cat, l_cat), count in diag_pairs.items():
                f.write(f"{h_cat} -> {l_cat}: {count} ETFs\n")
        
        f.write("\n\nSAMPLE DISAGREEMENTS\n")
        f.write("-" * 70 + "\n")
        
        sample_disagreements = disagreements.head(10)
        for _, row in sample_disagreements.iterrows():
            f.write(f"\nName: {row['Name']}\n")
            f.write(f"Ticker: {row['Ticker']}\n")
            f.write(f"Haiku: {row['category_tier1']} / {row['category_tier2']}\n")
            f.write(f"Llama: {row['llama_tier1']} / {row['llama_tier2']}\n")
    
    print(f"   ✅ Saved: {report_path}")
    
    # Print summary to console
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"\nTotal ETFs Processed: {len(df)}")
    print(f"Tier-1 Exact Agreement: {tier1_agreement_pct:.1f}%")
    print(f"Tier-2 Exact Agreement: {tier2_agreement_pct:.1f}%")
    print(f"\nFiles saved to: {output_dir}/")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Compare Haiku vs Fine-Tuned Llama')
    parser.add_argument('--limit', type=int, default=None, help='Limit to first N ETFs (for testing)')
    parser.add_argument('--model-path', default=DEFAULT_MODEL_PATH, help='Model path')
    parser.add_argument('--output-dir', default=OUTPUT_DIR, help='Output directory')
    
    args = parser.parse_args()
    
    print("="*70)
    print("MODEL COMPARISON: Haiku vs Fine-Tuned Llama-3.1-8B")
    print("="*70)
    
    # Check API key
    if not os.environ.get('TINKER_API_KEY'):
        print("\n❌ Error: TINKER_API_KEY environment variable not set")
        print("   Set it with: export TINKER_API_KEY=your_key_here")
        sys.exit(1)
    
    # Load data
    print(f"\n📥 Loading input: {INPUT_FILE}")
    try:
        df = pd.read_excel(INPUT_FILE)
        print(f"   Loaded {len(df)} ETFs")
    except Exception as e:
        print(f"❌ Error loading input: {e}")
        sys.exit(1)
    
    # Limit if specified
    if args.limit:
        df = df.head(args.limit)
        print(f"   Limited to first {args.limit} ETFs")
    
    # Initialize model and tokenizer
    print(f"\n🔌 Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("thinkingmachineslabinc/meta-llama-3-tokenizer")
        print("   ✅ Loaded")
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        sys.exit(1)
    
    print(f"\n🔌 Connecting to Tinker...")
    print(f"   Model: {args.model_path}")
    
    try:
        service_client = tinker.ServiceClient()
        sampling_client = service_client.create_sampling_client(model_path=args.model_path)
        print("   ✅ Connected")
    except Exception as e:
        print(f"❌ Error connecting to Tinker: {e}")
        sys.exit(1)
    
    # Run inference
    print(f"\n🚀 Running inference on {len(df)} ETFs...")
    print(f"   Estimated time: ~{len(df) * 1.5 / 60:.0f} minutes")
    
    predictions = []
    parse_errors = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Classifying"):
        try:
            prompt = build_fund_prompt(row)
            result = classify_fund(sampling_client, tokenizer, prompt)
            
            if result.get('parse_error'):
                parse_errors += 1
            
            predictions.append({
                'llama_tier1': result['tier_1'],
                'llama_tier2': result['tier_2'],
                'llama_tier3': ', '.join(result['tier_3']) if result['tier_3'] else ''
            })
            
        except Exception as e:
            print(f"\n⚠️  Error on row {idx}: {e}")
            predictions.append({
                'llama_tier1': 'ERROR',
                'llama_tier2': 'ERROR',
                'llama_tier3': ''
            })
    
    # Add predictions to dataframe
    print("\n📊 Merging predictions...")
    pred_df = pd.DataFrame(predictions)
    df = pd.concat([df, pred_df], axis=1)
    
    # Compute agreement metrics
    print("\n📊 Computing agreement metrics...")
    df['tier1_match'] = df['category_tier1'] == df['llama_tier1']
    df['tier2_match'] = df['category_tier2'] == df['llama_tier2']
    df['agreement'] = df.apply(lambda x: compute_agreement(x['category_tier1'], x['llama_tier1']), axis=1)
    
    # Save fine-tuned classifications
    os.makedirs(args.output_dir, exist_ok=True)
    finetuned_path = os.path.join(args.output_dir, 'ETF_Master_List_FineTuned.xlsx')
    df.to_excel(finetuned_path, index=False)
    print(f"   ✅ Saved: {finetuned_path}")
    
    # Generate comparison report
    stats = generate_comparison_report(df, args.output_dir)
    
    print("\n" + "="*70)
    print("✅ COMPARISON COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
