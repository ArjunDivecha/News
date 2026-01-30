"""
=============================================================================
P0 DEDUPLICATION ANALYSIS - FULL DATASET OPTIMIZED
=============================================================================

INPUT FILES:
- /Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/News/Step 2 Data Processing - Final1000/Final Master.xlsx
  Description: Master dataset containing 4,933 assets with beta vectors, performance metrics, and classifications
  Required Format: Excel file with 30+ beta columns, performance data, and category classifications
  Key Columns: Daily 1 Year Beta to SPX, Russell 2000 Index, MSCI EAFE Index, 1 Year Sharpe, category_tier2, category_tags, Long_Description

OUTPUT FILES:
- /Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/News/Step 3 Data Analysis/Dedup Report - Full Dataset.xlsx
  Description: Comprehensive deduplication report with KEEP/REMOVE recommendations for all proxy groups
  Format: Excel file with action recommendations, performance metrics, and group analysis
  Contents: Proxy group assignments, selection scores, recommended actions, and quality rankings

VERSION HISTORY:
v1.0.0 (2025-10-16): Initial release with 30-dimensional beta vector clustering
v1.1.0 (2025-10-17): Added composite proxy scoring with multiple signals
v1.2.0 (2025-11-06): Enhanced documentation and error handling
v1.3.0 (2025-11-06): Fast version to avoid hanging
v1.4.0 (2025-11-06): Full dataset optimized version

PURPOSE:
- Full dataset beta vector clustering (optimized)
- Composite proxy scoring (efficient)
- Optimized clustering to identify proxy groups
- Output: Complete dedup report with recommendations
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("P0 DEDUPLICATION ANALYSIS - FULL DATASET OPTIMIZED")
print("="*80)

# Load data
print("\n[1/5] Loading Final Master.xlsx...")
df = pd.read_excel("../Step 2 Data Processing - Final1000/Final Master.xlsx")
print(f"Loaded {len(df)} assets with {len(df.columns)} columns")

# Key beta columns (use most important ones for efficiency)
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
    'Bitcoin',
    'Brent',
    'WTI',
    'Dollar',
    'Euro',
    'Jpy Yen'
]

# Data cleaning
print("\n[2/5] Cleaning quantitative data...")
numeric_cols = beta_columns + ['1 Year Sharpe', '12 month volatility', '12 Month Return']

df_clean = df.copy()
for col in numeric_cols:
    if col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# Fill missing values with median for robustness
print("  Filling missing values with median...")
for col in numeric_cols:
    if col in df_clean.columns:
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            print(f"    {col}: {missing_count} missing filled with median {median_val:.3f}")

# Build beta matrix
print("\n[3/5] Building optimized beta vectors...")
beta_matrix = df_clean[beta_columns].fillna(0).values
print(f"  Beta matrix shape: {beta_matrix.shape}")

# Standardize betas for better distance calculation
print("  Standardizing betas...")
scaler = StandardScaler()
beta_matrix_scaled = scaler.fit_transform(beta_matrix)

# Efficient similarity calculation using vectorized operations
print("\n[4/5] Computing similarities efficiently...")
print(f"  Computing {len(df_clean)} x {len(df_clean)} similarity matrix...")

# Use chunked computation to manage memory
chunk_size = 500
n_assets = len(df_clean)
proxy_score = np.zeros((n_assets, n_assets))

print("  Processing in chunks...", end='', flush=True)
for i in range(0, n_assets, chunk_size):
    if i % 1000 == 0:
        print('.', end='', flush=True)
    
    end_i = min(i + chunk_size, n_assets)
    
    for j in range(i, n_assets, chunk_size):
        end_j = min(j + chunk_size, n_assets)
        
        # Compute beta distances for this chunk
        chunk_i = beta_matrix_scaled[i:end_i]
        chunk_j = beta_matrix_scaled[j:end_j]
        
        # Vectorized euclidean distance
        if i == j:
            # Same chunk - compute full matrix
            distances = np.linalg.norm(chunk_i[:, None, :] - chunk_j[None, :, :], axis=2)
        else:
            # Different chunks - compute cross distances
            distances = np.linalg.norm(chunk_i[:, None, :] - chunk_j[None, :, :], axis=2)
        
        # Convert to similarity
        max_dist = distances.max() if distances.max() > 0 else 1.0
        beta_similarity = 1 - (distances / max_dist)
        
        # Add tier-2 matching
        tier2_i = df_clean['category_tier2'].iloc[i:end_i].values
        tier2_j = df_clean['category_tier2'].iloc[j:end_j].values
        tier2_match = (tier2_i[:, None] == tier2_j[None, :]).astype(float)
        
        # Composite score
        chunk_score = 0.7 * beta_similarity + 0.3 * tier2_match
        
        # Store in full matrix
        proxy_score[i:end_i, j:end_j] = chunk_score
        if i != j:  # Mirror for symmetric matrix
            proxy_score[j:end_j, i:end_i] = chunk_score.T

print(' Done!')

# Final clustering
print("\n[5/5] Optimized clustering on full dataset...")
threshold = 0.75
print(f"  Using similarity threshold: {threshold}")

# Efficient grouping using vectorized operations
print("  Assigning assets to groups...", end='', flush=True)
clusters = np.arange(n_assets)  # Each asset starts as its own cluster
group_id = 1
processed = 0

for i in range(n_assets):
    if i % 500 == 0:
        print('.', end='', flush=True)
    
    if clusters[i] != i:  # Already assigned to a group
        continue
    
    # Find similar unassigned assets (vectorized)
    unassigned_mask = (clusters == np.arange(n_assets))
    similar_mask = (proxy_score[i] > threshold) & unassigned_mask
    similar_assets = np.where(similar_mask)[0]
    
    if len(similar_assets) > 1:
        # Assign all to same group
        clusters[similar_assets] = group_id
        group_id += 1
    else:
        clusters[i] = group_id
        group_id += 1
    
    processed += 1

print(' Done!')

# Add cluster info to dataframe
df_clean['proxy_group'] = clusters
df_clean['proxy_score'] = proxy_score.max(axis=1)

# Analyze groups
print(f"\n{'='*80}")
print(f"DEDUPLICATION RESULTS - FULL DATASET")
print(f"{'='*80}")

n_groups = len(np.unique(clusters))
print(f"\nTotal proxy groups identified: {n_groups}")
print(f"Total assets processed: {len(df_clean)}")

# Find significant proxy groups (>1 asset)
group_sizes = pd.Series(clusters).value_counts()
multi_groups = group_sizes[group_sizes > 1]
print(f"Groups with >1 asset (potential duplicates): {len(multi_groups)}")
print(f"Assets involved in duplicates: {multi_groups.sum()}")
print(f"Potential reduction: {multi_groups.sum() - len(multi_groups)} assets could be removed")

print(f"\nProxy group size distribution:")
size_dist = group_sizes.value_counts().sort_index()
for size, count in size_dist.head(10).items():
    print(f"  Groups of size {size}: {count}")

# Output: Top proxy groups for review
print(f"\n{'='*80}")
print(f"TOP 30 PROXY GROUPS FOR REVIEW")
print(f"{'='*80}\n")

dedup_results = []
group_id = 1
for grp in multi_groups.sort_values(ascending=False).head(30).index:
    group_assets = df_clean[df_clean['proxy_group'] == grp].copy()
    group_assets = group_assets.sort_values('1 Year Sharpe', ascending=False, na_position='last')

    print(f"\nGROUP {group_id}: {len(group_assets)} assets")
    print(f"  Tier-1: {group_assets.iloc[0]['category_tier1']} | Tier-2: {group_assets.iloc[0]['category_tier2']}")

    for idx, (_, row) in enumerate(group_assets.iterrows(), 1):
        action = "KEEP" if idx == 1 else "REMOVE"
        sharpe = row['1 Year Sharpe']
        vol = row['12 month volatility']
        ret = row['12 Month Return']
        print(f"    {idx}. [{action}] {row['Bloomberg_Ticker']:15} | Sharpe: {sharpe:>7.3f} | Vol: {vol:>7.2f} | Ret: {ret:>7.2f}")

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
output_file = "Dedup Report - Full Dataset.xlsx"
dedup_df.to_excel(output_file, index=False)

print(f"\n{'='*80}")
print(f"✓ DEDUP REPORT GENERATED")
print(f"{'='*80}")
print(f"File: {output_file}")
print(f"Total assets in top 30 groups: {len(dedup_df)}")
print(f"Potential removals (rank > 1): {len(dedup_df[dedup_df['rank_in_group'] > 1])}")
print(f"Assets to KEEP: {len(dedup_df[dedup_df['action'] == 'KEEP'])}")

print(f"\n{'='*80}")
print(f"FULL DATASET SUMMARY")
print(f"{'='*80}")
print(f"Starting assets: {len(df_clean)}")
print(f"Total proxy groups: {n_groups}")
print(f"Assets in multi-asset groups: {multi_groups.sum()}")
print(f"Estimated removable assets: {multi_groups.sum() - len(multi_groups)}")
print(f"Estimated remaining assets: {len(df_clean) - (multi_groups.sum() - len(multi_groups))}")
print(f"\nNext steps: Review top groups, validate selections, then proceed to final 1000 selection.")

print(f"\nEOF")
