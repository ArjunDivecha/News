"""
FINAL SELECTION ALGORITHM
- Reduce 4933 assets to ~1000
- Maximize diversity + quality + coverage
- Apply strategic allocation targets
"""

import pandas as pd
import numpy as np

print("="*80)
print("FINAL SELECTION ALGORITHM: 4933 → 1000")
print("="*80)

df = pd.read_excel("Final Master.xlsx")

# Convert numeric columns
numeric_cols = ['Daily 1 Year Beta to SPX', 'Russell 2000 Index', 'MSCI EAFE Index',
                '1 Year Sharpe', '12 Month Return', '12 month volatility', 'Correlation with SPX']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("\n[1/3] Computing selection scores...")

# Selection score components
df['sharpe_score'] = df['1 Year Sharpe'].fillna(0)
df['quality_score'] = df['sharpe_score'] / (df['sharpe_score'].max() + 0.001)

# Tier-2 popularity (avoid concentration)
tier2_counts = df['category_tier2'].value_counts()
df['tier2_popularity'] = df['category_tier2'].map(tier2_counts)
df['uniqueness_score'] = 1 / (1 + df['tier2_popularity'] / df['tier2_popularity'].max())

# Data completeness
df['data_quality'] = df[numeric_cols].notna().sum(axis=1) / len(numeric_cols)

# Composite selection score
df['selection_score'] = (
    0.40 * df['quality_score'] +
    0.30 * df['uniqueness_score'] +
    0.20 * df['data_quality'] +
    0.10 * (df['Correlation with SPX'].fillna(0.5) / 1.0)  # Diversification bonus
)

print(f"  Selection scores: min={df['selection_score'].min():.3f}, max={df['selection_score'].max():.3f}")

print("\n[2/3] Applying strategic allocation...")

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

selected = []

for tier1, target in targets.items():
    tier1_assets = df[df['category_tier1'] == tier1].copy()

    if len(tier1_assets) == 0:
        print(f"  {tier1}: No assets found")
        continue

    # Sort by selection score
    tier1_assets = tier1_assets.sort_values('selection_score', ascending=False)

    # Take top N or all if fewer than target
    n_select = min(len(tier1_assets), target)
    selected_tier1 = tier1_assets.head(n_select)
    selected.append(selected_tier1)

    print(f"  {tier1}: {n_select}/{target} selected from {len(tier1_assets)} available")

# Combine selections
final_df = pd.concat(selected, ignore_index=True)

print(f"\nTotal selected: {len(final_df)} assets")

print("\n[3/3] Generating final master list...")

# Build final output
final_output = pd.DataFrame({
    'Bloomberg_Ticker': final_df['Bloomberg_Ticker'],
    'Name': final_df['Name'],
    'Long_Description': final_df['Long_Description'],
    'category_tier1': final_df['category_tier1'],
    'category_tier2': final_df['category_tier2'],
    'category_tags': final_df['category_tags'],
    'source': final_df['source'],

    # Key metrics
    'Sharpe_1Y': final_df['1 Year Sharpe'],
    'Return_12M': final_df['12 Month Return'],
    'Vol_12M': final_df['12 month volatility'],
    'Beta_SPX': final_df['Daily 1 Year Beta to SPX'],
    'Correlation_SPX': final_df['Correlation with SPX'],

    # Scores
    'Selection_Score': final_df['selection_score'],
})

# Sort by Tier-1, then by selection score
final_output = final_output.sort_values(['category_tier1', 'Selection_Score'], ascending=[True, False])

# Save
final_output.to_excel("Final 1000 Asset Master List.xlsx", index=False)

print(f"\n✓ Final selection complete")
print(f"  File: Final 1000 Asset Master List.xlsx")
print(f"  Total assets: {len(final_output)}")

print(f"\nDistribution by Tier-1:")
print(final_output['category_tier1'].value_counts().sort_index())

print(f"\nQuality metrics:")
print(f"  Avg Sharpe: {final_output['Sharpe_1Y'].mean():.3f}")
print(f"  Avg Return: {final_output['Return_12M'].mean():.3f}")
print(f"  Avg Vol: {final_output['Vol_12M'].mean():.3f}")

print(f"\nSource distribution:")
print(final_output['source'].value_counts())

print("\n" + "="*80)
print("✓ FINAL 1000-ASSET MASTER LIST READY")
print("="*80)
