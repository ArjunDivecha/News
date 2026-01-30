"""
=============================================================================
FINAL SELECTION ALGORITHM V2 - Enhanced Thematic Diversity
=============================================================================

INPUT FILES:
- Final Master.xlsx: Complete master dataset with 4,933 classified assets

OUTPUT FILES:
- Final 1000 Asset Master List.xlsx: Optimized portfolio with thematic diversity

VERSION HISTORY:
v1.0.0 (2025-10-16): Initial release with basic selection algorithm
v1.1.0 (2025-10-17): Added strategic allocation targets and quality scoring
v1.2.0 (2025-11-06): Enhanced documentation and optimization metrics
v2.0.0 (2026-01-30): Added proxy Sharpe calculation and source quotas for thematic diversity

KEY CHANGES IN V2:
1. PROXY SHARPE: Computes Sharpe from 36M Return + 12M Volatility for assets with missing data
2. SOURCE QUOTAS: Ensures minimum representation from Goldman (15%) and Bloomberg (5%)
3. THEMATIC DIVERSITY: Prioritizes unique themes within each source
4. DATA PARITY: Levels playing field between ETFs (which have data) and proprietary indices (which don't)

ALLOCATION TARGETS:
- Equities: 520 assets (52%)
- Fixed Income: 170 assets (17%)
- Commodities: 110 assets (11%)
- Currencies (FX): 70 assets (7%)
- Multi-Asset / Thematic: 80 assets (8%)
- Volatility / Risk Premia: 50 assets (5%)
- Alternative / Synthetic: 20 assets (2%)

SOURCE QUOTAS (per Tier-1 category):
- Goldman: minimum 15% of each category's allocation
- Bloomberg: minimum 5% of each category's allocation
- Remaining: best scores regardless of source
=============================================================================
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("FINAL SELECTION ALGORITHM V2: Enhanced Thematic Diversity")
print("=" * 80)

# =============================================================================
# STEP 1: LOAD AND PREPARE DATA
# =============================================================================
print("\n[1/5] Loading data...")

df = pd.read_excel("Final Master.xlsx")
print(f"  Loaded {len(df)} assets")

# Convert numeric columns
numeric_cols = ['Daily 1 Year Beta to SPX', 'Russell 2000 Index', 'MSCI EAFE Index',
                '1 Year Sharpe', '12 Month Return', '12 month volatility',
                '36 Month Return', '30 day volatility', 'Correlation with SPX']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# =============================================================================
# STEP 2: COMPUTE PROXY SHARPE FOR MISSING DATA
# =============================================================================
print("\n[2/5] Computing proxy Sharpe ratios...")

# Actual Sharpe where available
df['actual_sharpe'] = df['1 Year Sharpe']

# Proxy Sharpe: (Annualized 36M Return - Risk Free Rate) / 12M Volatility
# Annualized 1Y Return ≈ 36 Month Return / 3
# Risk Free Rate ≈ 4% (current T-bill rate)
RISK_FREE_RATE = 4.0

df['annualized_return'] = df['36 Month Return'] / 3  # 3-year to 1-year
df['proxy_sharpe'] = (df['annualized_return'] - RISK_FREE_RATE) / df['12 month volatility']

# Use actual Sharpe where available, otherwise use proxy
df['effective_sharpe'] = df['actual_sharpe'].fillna(df['proxy_sharpe'])

# Report on data coverage
for source in df['source'].unique():
    src_df = df[df['source'] == source]
    actual = src_df['actual_sharpe'].notna().sum()
    proxy = src_df['proxy_sharpe'].notna().sum()
    effective = src_df['effective_sharpe'].notna().sum()
    print(f"  {source}: {actual} actual, {proxy} proxy → {effective} effective ({effective/len(src_df)*100:.1f}%)")

# =============================================================================
# STEP 3: COMPUTE SELECTION SCORES
# =============================================================================
print("\n[3/5] Computing selection scores...")

# Quality score from effective Sharpe (handles negative Sharpe correctly)
sharpe_min = df['effective_sharpe'].min()
sharpe_max = df['effective_sharpe'].max()
sharpe_range = sharpe_max - sharpe_min if sharpe_max != sharpe_min else 1
df['quality_score'] = (df['effective_sharpe'].fillna(0) - sharpe_min) / sharpe_range

# Uniqueness score (inverse of category popularity)
# But calculate WITHIN each source to avoid penalizing Goldman's larger catalog
df['tier2_popularity'] = df.groupby('source')['category_tier2'].transform('count')
df['uniqueness_score'] = 1 / (1 + np.log1p(df['tier2_popularity']))  # Log scale for fairness

# Data completeness (less important now that we have proxy Sharpe)
data_cols = ['effective_sharpe', '12 month volatility', 'Correlation with SPX', 'Daily 1 Year Beta to SPX']
df['data_quality'] = df[data_cols].notna().sum(axis=1) / len(data_cols)

# Diversification bonus (lower correlation to SPX is better)
df['diversification'] = 1 - df['Correlation with SPX'].abs().fillna(0.5)

# Thematic diversity bonus: reward unique Tier-2 categories
tier2_global_counts = df['category_tier2'].value_counts()
df['thematic_rarity'] = 1 / (1 + np.log1p(df['category_tier2'].map(tier2_global_counts)))

# Composite selection score
df['selection_score'] = (
    0.35 * df['quality_score'] +           # Reduced from 40% to level playing field
    0.25 * df['uniqueness_score'] +         # Within-source uniqueness
    0.15 * df['data_quality'] +             # Reduced importance
    0.10 * df['diversification'] +          # SPX correlation
    0.15 * df['thematic_rarity']            # NEW: Global thematic rarity
)

print(f"  Selection scores: min={df['selection_score'].min():.3f}, max={df['selection_score'].max():.3f}")

# =============================================================================
# STEP 4: APPLY STRATEGIC ALLOCATION WITH SOURCE QUOTAS
# =============================================================================
print("\n[4/5] Applying strategic allocation with source quotas...")

# Allocation targets by Tier-1
targets = {
    'Equities': 520,
    'Fixed Income': 170,
    'Commodities': 110,
    'Currencies (FX)': 70,
    'Multi-Asset / Thematic': 80,
    'Volatility / Risk Premia': 50,
    'Alternative / Synthetic': 20,
}

# Source quotas (minimum percentage from each source)
SOURCE_QUOTAS = {
    'Goldman': 0.15,      # At least 15% from Goldman
    'Bloomberg': 0.05,    # At least 5% from Bloomberg
}

selected = []

for tier1, target in targets.items():
    tier1_assets = df[df['category_tier1'] == tier1].copy()

    if len(tier1_assets) == 0:
        print(f"  {tier1}: No assets found")
        continue

    tier1_selected = []
    remaining_slots = target

    # PHASE 1: Fill source quotas first
    for source, min_pct in SOURCE_QUOTAS.items():
        source_assets = tier1_assets[tier1_assets['source'] == source].copy()
        if len(source_assets) == 0:
            continue

        min_count = int(target * min_pct)

        # Sort by selection score and take top N
        source_assets = source_assets.sort_values('selection_score', ascending=False)
        quota_select = min(len(source_assets), min_count)

        if quota_select > 0:
            tier1_selected.append(source_assets.head(quota_select))
            remaining_slots -= quota_select

    # PHASE 2: Fill remaining slots with best scores regardless of source
    if remaining_slots > 0 and len(tier1_selected) > 0:
        already_selected = pd.concat(tier1_selected)['Bloomberg_Ticker'].tolist()
        remaining_assets = tier1_assets[~tier1_assets['Bloomberg_Ticker'].isin(already_selected)]
    else:
        remaining_assets = tier1_assets

    if len(remaining_assets) > 0 and remaining_slots > 0:
        remaining_assets = remaining_assets.sort_values('selection_score', ascending=False)
        tier1_selected.append(remaining_assets.head(remaining_slots))

    # Combine selections for this tier
    if tier1_selected:
        combined = pd.concat(tier1_selected, ignore_index=True)
        selected.append(combined)

        # Report source breakdown
        source_counts = combined['source'].value_counts().to_dict()
        breakdown = ", ".join([f"{s}: {c}" for s, c in sorted(source_counts.items())])
        print(f"  {tier1}: {len(combined)}/{target} selected ({breakdown})")

# Combine all selections
final_df = pd.concat(selected, ignore_index=True)
print(f"\nTotal selected: {len(final_df)} assets")

# =============================================================================
# STEP 5: GENERATE FINAL OUTPUT
# =============================================================================
print("\n[5/5] Generating final master list...")

# Format Bloomberg_Ticker based on source type
def format_bloomberg_ticker(row):
    ticker = row['Bloomberg_Ticker']
    source = row['source']

    if pd.isna(ticker) or ticker == '':
        return ticker

    if source == 'ETF':
        return f"{ticker} Equity"
    elif source == 'Bloomberg':
        return f"{ticker} Index"
    elif source == 'Thematic':
        return f"{ticker} Equity"
    else:  # Goldman Sachs
        return ticker

final_df['Bloomberg_Ticker'] = final_df.apply(format_bloomberg_ticker, axis=1)

# Build final output
final_output = pd.DataFrame({
    'Bloomberg_Ticker': final_df['Bloomberg_Ticker'],
    'Name': final_df['Name'],
    'Long_Description': final_df['Long_Description'],
    'category_tier1': final_df['category_tier1'],
    'category_tier2': final_df['category_tier2'],
    'category_tags': final_df['category_tags'],
    'source': final_df['source'],

    # Key metrics (use effective Sharpe which includes proxy)
    'Sharpe_1Y': final_df['effective_sharpe'],
    'Sharpe_Type': final_df['actual_sharpe'].apply(lambda x: 'actual' if pd.notna(x) else 'proxy'),
    'Return_12M': final_df['12 Month Return'],
    'Return_36M': final_df['36 Month Return'],
    'Vol_12M': final_df['12 month volatility'],
    'Beta_SPX': final_df['Daily 1 Year Beta to SPX'],
    'Correlation_SPX': final_df['Correlation with SPX'],

    # Scores
    'Selection_Score': final_df['selection_score'],
    'Quality_Score': final_df['quality_score'],
    'Thematic_Rarity': final_df['thematic_rarity'],
})

# Sort by Tier-1, then by selection score
final_output = final_output.sort_values(['category_tier1', 'Selection_Score'], ascending=[True, False])

# Save
final_output.to_excel("Final 1000 Asset Master List.xlsx", index=False)

# =============================================================================
# REPORTING
# =============================================================================
print(f"\n{'='*80}")
print("RESULTS SUMMARY")
print(f"{'='*80}")

print(f"\n✓ Final selection complete")
print(f"  File: Final 1000 Asset Master List.xlsx")
print(f"  Total assets: {len(final_output)}")

print(f"\nDistribution by Tier-1:")
print(final_output['category_tier1'].value_counts().sort_index())

print(f"\nSource distribution:")
source_dist = final_output['source'].value_counts()
for source, count in source_dist.items():
    pct = count / len(final_output) * 100
    print(f"  {source}: {count} ({pct:.1f}%)")

print(f"\nSharpe data source:")
print(final_output['Sharpe_Type'].value_counts())

print(f"\nQuality metrics:")
print(f"  Avg Effective Sharpe: {final_output['Sharpe_1Y'].mean():.3f}")
print(f"  Avg 36M Return: {final_output['Return_36M'].mean():.2f}%")
print(f"  Avg 12M Vol: {final_output['Vol_12M'].mean():.2f}%")

# Goldman thematic analysis
goldman_selected = final_output[final_output['source'] == 'Goldman']
if len(goldman_selected) > 0:
    print(f"\nGoldman thematic breakdown:")
    print(goldman_selected['category_tier2'].value_counts().head(10))

print(f"\n{'='*80}")
print("✓ FINAL 1000-ASSET MASTER LIST V2 READY")
print(f"{'='*80}")
