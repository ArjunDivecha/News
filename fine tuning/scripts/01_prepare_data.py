#!/usr/bin/env python3
"""
================================================================================
DATA PREPARATION FOR FINE-TUNING
Extract classified assets from Excel files and format for Tinker
================================================================================

This script:
1. Loads 4 classified Excel files from the parent directories
2. Extracts asset information and classifications
3. Formats into chat completion format for fine-tuning
4. Creates stratified train/val/test splits
5. Outputs JSONL files ready for Tinker

Input Files:
- ../Step 2 Data Processing - Final1000/ETF Master List Classified.xlsx
- ../Step 2 Data Processing - Final1000/Filtered Bloomberg Indices Classified.xlsx
- ../Step 1 Data Collection/GSCB_FLAGSHIP_coverage_Classified.xlsx
- ../Step 2 Data Processing - Final1000/Thematic ETFs Classified.xlsx

Output Files:
- data/processed/train.jsonl (85% of data)
- data/processed/val.jsonl (10% of data)
- data/processed/test.jsonl (5% of data)
- data/processed/statistics.json (distribution analysis)
- data/processed/sample_examples.json (for inspection)

Usage:
    python scripts/01_prepare_data.py

Author: Claude (AI Assistant)
Date: 2026-01-29
================================================================================
"""

import pandas as pd
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
import sys

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Source file paths (relative to fine tuning/ directory)
DATA_SOURCES = {
    'etf': {
        'path': '../Step 2 Data Processing - Final1000/ETF Master List Classified.xlsx',
        'ticker_col': 'Ticker',
        'name_col': 'Name',
        'desc_col': 'CIE_DES',
        'tier1_col': 'category_tier1',
        'tier2_col': 'category_tier2',
        'tags_col': 'category_tags',
        'optional_cols': ['FUND_ASSET_CLASS_FOCUS', 'FUND_GEO_FOCUS', 'FUND_OBJECTIVE_LONG', 'FUND_STRATEGY']
    },
    'bloomberg': {
        'path': '../Step 2 Data Processing - Final1000/Filtered Bloomberg Indices Classified.xlsx',
        'ticker_col': 'Ticker',
        'name_col': 'Index Name',
        'desc_col': 'LONG_COMP_NAME',
        'tier1_col': 'category_tier1',
        'tier2_col': 'category_tier2',
        'tags_col': 'category_tags',
        'optional_cols': ['SECURITY_TYP', 'REGION_OR_COUNTRY']
    },
    'goldman': {
        'path': '../Step 1 Data Collection/GSCB_FLAGSHIP_coverage_Classified.xlsx',
        'ticker_col': 'Bloomberg',
        'name_col': 'Index Name',
        'desc_col': 'description',
        'tier1_col': 'category_tier1',
        'tier2_col': 'category_tier2',
        'tags_col': 'category_tags',
        'optional_cols': ['REGION_OR_COUNTRY']
    },
    'thematic': {
        'path': '../Step 2 Data Processing - Final1000/Thematic ETFs Classified.xlsx',
        'ticker_col': 'Ticker',
        'name_col': 'Name',
        'desc_col': 'CIE_DES',
        'tier1_col': 'category_tier1',
        'tier2_col': 'category_tier2',
        'tags_col': 'category_tags',
        'optional_cols': ['FUND_ASSET_CLASS_FOCUS', 'FUND_GEO_FOCUS', 'FUND_OBJECTIVE_LONG', 'FUND_STRATEGY']
    }
}

# Output directory
OUTPUT_DIR = Path('data/processed')

# Random seed for reproducibility
RANDOM_SEED = 42

# Split ratios
TRAIN_RATIO = 0.85
VAL_RATIO = 0.10
TEST_RATIO = 0.05

# System prompt for classification
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
- Strategy: Active | Passive | Equal-Weight | Thematic | Quantitative | Options-Based | Dividend | Factor-Based | Low Volatility | Defensive | Domestic | Leveraged | Inverse
- Duration (bonds only): Short (<2Y) | Medium (2-10Y) | Long (>10Y) | Unspecified
- Special: Inverse | Long-Short | Covered-Call | Leveraged | Custom | Proprietary

Respond with ONLY valid JSON in this exact format:
{"ticker": "TICKER", "tier1": "Category", "tier2": "Sub-category", "tier3_tags": ["tag1", "tag2", "tag3"]}"""


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def clean_text(text: Any) -> str:
    """Clean and normalize text fields"""
    if pd.isna(text) or text is None:
        return ""
    text = str(text).strip()
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters that might break JSON
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    return text


def parse_tags(tag_string: Any) -> List[str]:
    """Parse comma-separated tags into list"""
    if pd.isna(tag_string) or tag_string is None:
        return []
    tags = [t.strip() for t in str(tag_string).split(',') if t.strip()]
    return tags


def is_valid_classification(tier1: Any, tier2: Any) -> bool:
    """Check if classification is valid (not null, not error, not unable to classify)"""
    if pd.isna(tier1) or pd.isna(tier2):
        return False
    tier1_str = str(tier1).strip()
    tier2_str = str(tier2).strip()
    
    # Check for empty strings
    if not tier1_str or not tier2_str:
        return False
    
    invalid_markers = [
        'unable to classify', 'unable', 'unknown', 'unclassified', 
        'error', 'parse error', 'parseerror'
    ]
    
    tier1_lower = tier1_str.lower()
    tier2_lower = tier2_str.lower()
    
    # Check for NaN-like strings
    if tier1_lower in ('nan', 'none', 'null', 'n/a', 'na'):
        return False
    if tier2_lower in ('nan', 'none', 'null', 'n/a', 'na'):
        return False
    
    # Check for invalid markers
    for marker in invalid_markers:
        if marker and marker in tier1_lower:
            return False
        if marker and marker in tier2_lower:
            return False
    
    return True


def create_training_example(row: pd.Series, source: str, config: Dict) -> Optional[Dict]:
    """
    Convert a classified row to chat completion format for fine-tuning
    
    Args:
        row: DataFrame row with asset data
        source: Source name (etf, bloomberg, goldman, thematic)
        config: Configuration dict for this source
    
    Returns:
        Training example dict or None if invalid
    """
    # Extract required fields
    ticker = row.get(config['ticker_col'])
    name = row.get(config['name_col'])
    description = row.get(config['desc_col'])
    tier1 = row.get(config['tier1_col'])
    tier2 = row.get(config['tier2_col'])
    tags = row.get(config['tags_col'])
    
    # Validate required fields
    if not is_valid_classification(tier1, tier2):
        return None
    
    if pd.isna(ticker) or pd.isna(name):
        return None
    
    # Build user prompt
    fields = []
    fields.append(f"Ticker: {clean_text(ticker)}")
    fields.append(f"Name: {clean_text(name)}")
    
    if pd.notna(description) and description:
        fields.append(f"Description: {clean_text(description)}")
    
    # Add optional fields if available
    field_mapping = {
        'FUND_ASSET_CLASS_FOCUS': 'Existing Asset Class',
        'FUND_GEO_FOCUS': 'Geographic Focus',
        'FUND_OBJECTIVE_LONG': 'Objective',
        'FUND_STRATEGY': 'Strategy',
        'SECURITY_TYP': 'Security Type',
        'REGION_OR_COUNTRY': 'Region'
    }
    
    for col, label in field_mapping.items():
        if col in row and pd.notna(row[col]) and row[col]:
            value = clean_text(row[col])
            if value and value.lower() not in ['nan', 'none', 'null', '']:
                fields.append(f"{label}: {value}")
    
    user_content = "Classify this asset:\n" + "\n".join(fields)
    
    # Build assistant response
    tier3_list = parse_tags(tags)
    
    assistant_content = json.dumps({
        "ticker": clean_text(ticker),
        "tier1": clean_text(tier1),
        "tier2": clean_text(tier2),
        "tier3_tags": tier3_list
    }, ensure_ascii=False)
    
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ],
        "metadata": {
            "source": source,
            "original_tier1": clean_text(tier1),
            "original_tier2": clean_text(tier2)
        }
    }


def load_source_data(source_name: str, config: Dict) -> tuple:
    """
    Load and process a single data source
    
    Returns:
        (examples, stats) tuple
    """
    print(f"\n📁 Loading {source_name}...")
    print(f"   Path: {config['path']}")
    
    try:
        df = pd.read_excel(config['path'])
        initial_count = len(df)
        
        print(f"   Total rows: {initial_count}")
        
        # Check for required columns
        required_cols = [config['tier1_col'], config['tier2_col']]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            print(f"   ⚠️  Missing columns: {missing_cols}")
            print(f"   Available: {list(df.columns)}")
            return [], {'error': f'Missing columns: {missing_cols}'}
        
        # Filter valid classifications
        valid_mask = df.apply(
            lambda row: is_valid_classification(
                row.get(config['tier1_col']),
                row.get(config['tier2_col'])
            ),
            axis=1
        )
        df_valid = df[valid_mask].copy()
        valid_count = len(df_valid)
        
        print(f"   Valid classifications: {valid_count}")
        
        # Show tier-1 distribution
        tier1_dist = df_valid[config['tier1_col']].value_counts().to_dict()
        print(f"   Tier-1 distribution:")
        for tier, count in sorted(tier1_dist.items(), key=lambda x: -x[1]):
            print(f"      {tier}: {count}")
        
        # Create examples
        examples = []
        for _, row in df_valid.iterrows():
            example = create_training_example(row, source_name, config)
            if example:
                examples.append(example)
        
        print(f"   Created {len(examples)} training examples")
        
        stats = {
            'source': source_name,
            'total_rows': initial_count,
            'valid_classifications': valid_count,
            'training_examples': len(examples),
            'tier1_distribution': tier1_dist
        }
        
        return examples, stats
        
    except FileNotFoundError:
        print(f"   ❌ File not found: {config['path']}")
        return [], {'error': 'File not found'}
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return [], {'error': str(e)}


def stratified_split(examples: List[Dict], 
                     train_ratio: float = 0.85,
                     val_ratio: float = 0.10,
                     test_ratio: float = 0.05,
                     seed: int = 42) -> Dict[str, List[Dict]]:
    """
    Split examples into train/val/test with stratification by Tier-1
    
    Returns:
        Dict with 'train', 'val', 'test' keys
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "Ratios must sum to 1"
    
    random.seed(seed)
    
    # Group by Tier-1 category
    by_category = defaultdict(list)
    for ex in examples:
        tier1 = ex['metadata']['original_tier1']
        by_category[tier1].append(ex)
    
    train_examples, val_examples, test_examples = [], [], []
    
    print(f"\n📊 Stratified split by Tier-1 category:")
    print(f"   {'Category':<30} {'Total':<8} {'Train':<8} {'Val':<8} {'Test':<8}")
    print("   " + "-" * 62)
    
    for tier1, category_examples in sorted(by_category.items()):
        random.shuffle(category_examples)
        n = len(category_examples)
        
        n_train = max(1, int(n * train_ratio)) if n >= 3 else n
        n_val = max(1, int(n * val_ratio)) if n >= 3 else 0
        n_test = n - n_train - n_val
        
        # Adjust if we took too many
        if n_train + n_val > n:
            n_val = n - n_train
        
        train_examples.extend(category_examples[:n_train])
        val_examples.extend(category_examples[n_train:n_train + n_val])
        test_examples.extend(category_examples[n_train + n_val:])
        
        print(f"   {tier1:<30} {n:<8} {n_train:<8} {n_val:<8} {n_test:<8}")
    
    # Shuffle each split
    random.shuffle(train_examples)
    random.shuffle(val_examples)
    random.shuffle(test_examples)
    
    return {
        'train': train_examples,
        'val': val_examples,
        'test': test_examples
    }


def save_datasets(splits: Dict[str, List[Dict]], output_dir: Path):
    """Save datasets in JSONL format"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving datasets to {output_dir}:")
    
    for split_name, examples in splits.items():
        jsonl_path = output_dir / f"{split_name}.jsonl"
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for ex in examples:
                # Strip metadata for training
                train_ex = {k: v for k, v in ex.items() if k != 'metadata'}
                f.write(json.dumps(train_ex, ensure_ascii=False) + '\n')
        print(f"   {split_name}.jsonl: {len(examples)} examples")
    
    # Save sample examples for inspection
    sample_path = output_dir / "sample_examples.json"
    samples = {
        'train': splits['train'][:3],
        'val': splits['val'][:2],
        'test': splits['test'][:2]
    }
    with open(sample_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"   sample_examples.json: saved for inspection")


def calculate_statistics(splits: Dict[str, List[Dict]], source_stats: List[Dict]) -> Dict:
    """Calculate comprehensive statistics"""
    stats = {
        'total_examples': sum(len(v) for v in splits.values()),
        'splits': {
            'train': len(splits['train']),
            'val': len(splits['val']),
            'test': len(splits['test'])
        },
        'split_ratios': {
            'train': len(splits['train']) / sum(len(v) for v in splits.values()),
            'val': len(splits['val']) / sum(len(v) for v in splits.values()),
            'test': len(splits['test']) / sum(len(v) for v in splits.values())
        },
        'tier1_distribution': {},
        'source_breakdown': source_stats
    }
    
    # Tier-1 distribution per split
    for split_name, examples in splits.items():
        tier1_counts = defaultdict(int)
        for ex in examples:
            tier1 = ex['metadata']['original_tier1']
            tier1_counts[tier1] += 1
        stats['tier1_distribution'][split_name] = dict(tier1_counts)
    
    return stats


def print_final_report(stats: Dict):
    """Print final summary report"""
    print("\n" + "="*70)
    print("✅ DATA PREPARATION COMPLETE")
    print("="*70)
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"   Total examples: {stats['total_examples']}")
    print(f"   Training set:   {stats['splits']['train']} ({stats['split_ratios']['train']:.1%})")
    print(f"   Validation set: {stats['splits']['val']} ({stats['split_ratios']['val']:.1%})")
    print(f"   Test set:       {stats['splits']['test']} ({stats['split_ratios']['test']:.1%})")
    
    print(f"\n📁 SOURCE BREAKDOWN:")
    for src in stats['source_breakdown']:
        if 'error' not in src:
            print(f"   {src['source']:<15}: {src['training_examples']:>5} examples")
    
    print(f"\n📈 TIER-1 DISTRIBUTION (Training Set):")
    train_dist = stats['tier1_distribution']['train']
    for tier, count in sorted(train_dist.items(), key=lambda x: -x[1]):
        pct = count / stats['splits']['train'] * 100
        print(f"   {tier:<35}: {count:>4} ({pct:>5.1f}%)")
    
    print(f"\n💾 OUTPUT FILES:")
    print(f"   data/processed/train.jsonl")
    print(f"   data/processed/val.jsonl")
    print(f"   data/processed/test.jsonl")
    print(f"   data/processed/statistics.json")
    print(f"   data/processed/sample_examples.json")
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"   1. Inspect sample_examples.json to verify format")
    print(f"   2. Run: export TINKER_API_KEY='your-key'")
    print(f"   3. Run: python scripts/02_train_tinker.py")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("="*70)
    print("🚀 DATA PREPARATION FOR FINE-TUNING")
    print("="*70)
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Split ratios: Train {TRAIN_RATIO:.0%}, Val {VAL_RATIO:.0%}, Test {TEST_RATIO:.0%}")
    
    # Load all sources
    all_examples = []
    source_stats = []
    
    for source_name, config in DATA_SOURCES.items():
        examples, stats = load_source_data(source_name, config)
        all_examples.extend(examples)
        source_stats.append(stats)
    
    if not all_examples:
        print("\n❌ No valid examples found. Check file paths and data.")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"📊 TOTAL: {len(all_examples)} valid training examples")
    print(f"{'='*70}")
    
    # Create stratified splits
    print(f"\n✂️  Creating stratified splits...")
    splits = stratified_split(
        all_examples,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=RANDOM_SEED
    )
    
    # Save datasets
    save_datasets(splits, OUTPUT_DIR)
    
    # Calculate statistics
    stats = calculate_statistics(splits, source_stats)
    
    # Save statistics
    stats_path = OUTPUT_DIR / "statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print report
    print_final_report(stats)
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
