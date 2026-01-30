"""
=============================================================================
P0 DEDUPLICATION ANALYSIS - FAST VERSION
=============================================================================

INPUT FILES:
- /Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/News/Step 2 Data Processing - Final1000/Final Master.xlsx
  Description: Master dataset containing 4,933 assets with beta vectors, performance metrics, and classifications
  Required Format: Excel file with 30+ beta columns, performance data, and category classifications
  Key Columns: Daily 1 Year Beta to SPX, Russell 2000 Index, MSCI EAFE Index, 1 Year Sharpe, category_tier2, category_tags, Long_Description

OUTPUT FILES:
- /Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/News/Step 3 Data Analysis/Dedup Report - Top 30 Groups.xlsx
  Description: Comprehensive deduplication report with KEEP/REMOVE recommendations for top 30 proxy groups
  Format: Excel file with action recommendations, performance metrics, and group analysis
  Contents: Proxy group assignments, selection scores, recommended actions, and quality rankings

VERSION HISTORY:
v1.0.0 (2025-10-16): Initial release with 30-dimensional beta vector clustering
v1.1.0 (2025-10-17): Added composite proxy scoring with multiple signals
v1.2.0 (2025-11-06): Enhanced documentation and error handling
v1.3.0 (2025-11-06): Fast version to avoid hanging

PURPOSE:
- Fast beta vector clustering (simplified)
- Composite proxy scoring (simplified)
- Quick clustering to identify proxy groups
- Output: Dedup report with recommendations
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("P0 DEDUPLICATION ANALYSIS - FAST VERSION")
print("="*80)

# Load data
print("\n[1/4] Loading Final Master.xlsx...")
df = pd.read_excel("../Step 2 Data Processing - Final1000/Final Master.xlsx")
print(f"Loaded {len(df)} assets with {len(df.columns)} columns")

# Simplified beta columns (use only first 10 for speed)
beta_columns = [
    'Daily 1 Year Beta to SPX',
    'Russell 2000 Index',
    'Nasdaq-100 Index',
    'Russell 1000 Value Index',
    'Russell 1000 Growth Index',
    'MSCI EAFE Index',
    'MSCI Emerging Markets Index',
    'US Generic Govt 10 Yr',
    'Gold',
    'Bitcoin'
]

# Data cleaning
print("\n[2/4] Cleaning quantitative data...")
numeric_cols = beta_columns + ['1 Year Sharpe', '12 month volatility', '12 Month Return']

df_clean = df.copy()
for col in numeric_cols:
    if col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# Fill missing values with median
for col in numeric_cols:
    if col in df_clean.columns:
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)

print(f"  Data cleaned, missing values filled with median")

# Build simplified beta matrix
print("\n[3/4] Building simplified beta vectors...")
beta_matrix = df_clean[beta_columns].fillna(0).values
print(f"  Beta matrix shape: {beta_matrix.shape}")

# Quick similarity calculation (sample-based)
print("  Computing similarities (sampled for speed)...")
sample_size = min(1000, len(df_clean))  # Sample for speed
sample_indices = np.random.choice(len(df_clean), sample_size, replace=False)
df_sample = df_clean.iloc[sample_indices].copy()

# Simple beta distance calculation
beta_sample = df_sample[beta_columns].values
beta_distances = np.zeros((sample_size, sample_size))

for i in range(sample_size):
    for j in range(i+1, sample_size):
        # Simple euclidean distance
        dist = np.linalg.norm(beta_sample[i] - beta_sample[j])
        beta_distances[i, j] = dist
        beta_distances[j, i] = dist

# Convert to similarity
max_dist = beta_distances.max()
beta_similarity = 1 - (beta_distances / (max_dist + 0.001))

# Simple tier-2 matching
tier2_match = (df_sample['category_tier2'].values[:, None] == df_sample['category_tier2'].values[None, :]).astype(float)

# Composite score
proxy_score = 0.7 * beta_similarity + 0.3 * tier2_match

print("\n[4/4] Fast clustering...")
threshold = 0.75
print(f"  Using similarity threshold: {threshold}")

# Simple grouping
clusters = np.arange(sample_size)  # Each asset starts as its own cluster
group_id = 1

for i in range(sample_size):
    if clusters[i] != i:  # Already assigned to a group
        continue
    
    # Find similar unassigned assets
    similar = np.where((proxy_score[i] > threshold) & (clusters == np.arange(sample_size)))[0]
    
    if len(similar) > 1:
        # Assign all to same group
        for idx in similar:
            clusters[idx] = group_id
        group_id += 1
    else:
        clusters[i] = group_id
        group_id += 1

# Add cluster info back to sample
df_sample['proxy_group'] = clusters
df_sample['proxy_score'] = proxy_score.max(axis=1)

# Analyze groups
print(f"\n{'='*80}")
print(f"DEDUPLICATION RESULTS (SAMPLED)")
print(f"{'='*80}")

n_groups = len(np.unique(clusters))
print(f"\nTotal proxy groups identified: {n_groups}")
print(f"Sample size: {sample_size} (from {len(df_clean)} total assets)")

# Find significant proxy groups (>1 asset)
group_sizes = pd.Series(clusters).value_counts()
multi_groups = group_sizes[group_sizes > 1]
print(f"Groups with >1 asset (potential duplicates): {len(multi_groups)}")
print(f"Assets involved in duplicates: {multi_groups.sum()}")

# Output: Top proxy groups for review
print(f"\n{'='*80}")
print(f"TOP PROXY GROUPS FOR REVIEW")
print(f"{'='*80}\n")

dedup_results = []
group_id = 1
for grp in multi_groups.sort_values(ascending=False).head(30).index:
    group_assets = df_sample[df_sample['proxy_group'] == grp].copy()
    group_assets = group_assets.sort_values('1 Year Sharpe', ascending=False, na_position='last')

    print(f"\nGROUP {group_id}: {len(group_assets)} assets")
    print(f"  Tier-1: {group_assets.iloc[0]['category_tier1']} | Tier-2: {group_assets.iloc[0]['category_tier2']}")

    for idx, (_, row) in enumerate(group_assets.iterrows(), 1):
        action = "KEEP" if idx == 1 else "REMOVE"
        sharpe = row['1 Year Sharpe']
        vol = row['12 month volatility']
        ret = row['12 Month Return']
        print(f"    {idx}. [{action}] {row['Bloomberg_Ticker']:15} | Sharpe: {sharpe:>7} | Vol: {vol:>7} | Ret: {ret:>7}")

        dedup_results.append({
            'proxy_group': grp,
            'group_size': len(group_assets),
            'rank_in_group': idx,
            'action': action,
            'Bloomberg_Ticker': row['Bloomberg_Ticker'],
            'Name': row['Name'],
            'Tier1': row['category_tier1'],
            'Tier2': row['category_tier2'],
            'Source': row['source'],
            'Sharpe_1Y': row['1 Year Sharpe'],
            'Vol_12M': row['12 month volatility'],
            'Return_12M': row['12 Month Return'],
        })

    group_id += 1

# Save dedup report
dedup_df = pd.DataFrame(dedup_results)
dedup_df.to_excel("Dedup Report - Top 30 Groups.xlsx", index=False)

print(f"\n{'='*80}")
print(f"✓ DEDUP REPORT GENERATED")
print(f"{'='*80}")
print(f"File: Dedup Report - Top 30 Groups.xlsx")
print(f"Total assets in top groups: {len(dedup_df)}")
print(f"Potential removals (rank > 1): {len(dedup_df[dedup_df['rank_in_group'] > 1])}")
print(f"Assets to KEEP: {len(dedup_df[dedup_df['action'] == 'KEEP'])}")

print(f"\n{'='*80}")
print(f"SUMMARY")
print(f"{'='*80}")
print(f"Fast analysis completed on sample of {sample_size} assets")
print(f"Estimated duplicate reduction: ~{multi_groups.sum() - len(multi_groups)} assets")
print(f"Next steps: Review groups, then apply to full dataset if needed")

print(f"\nEOF")
