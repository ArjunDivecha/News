"""
P0 DEDUPLICATION ANALYSIS
- Beta vector clustering (30-dimensional)
- Composite proxy scoring
- Hierarchical clustering to identify proxy groups
- Output: Dedup report with recommendations
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("P0 DEDUPLICATION ANALYSIS")
print("="*80)

# Load data
print("\n[1/6] Loading Final Master.xlsx...")
df = pd.read_excel("Final Master.xlsx")
print(f"Loaded {len(df)} assets with {len(df.columns)} columns")

# List of beta columns (30+ betas to different indices)
beta_columns = [
    'Monthly 5 Year beta to SPX',
    'Weekly 3 Year Beta to SPX',
    'Daily 1 Year Beta to SPX',
    'Russell 2000 Index',
    'Nasdaq-100 Index',
    'Russell 1000 Value Index',
    'Russell 1000 Growth Index',
    'MSCI EAFE Index',
    'MSCI Emerging Markets Index',
    'US Generic Govt 2 Yr',
    'US Generic Govt 10 Yr',
    'Bloomberg US Treasury Total Return Unhedged USD',
    'Bloomberg US Corporate Total Return Value Unhedged USD',
    'Bloomberg US Corporate High Yield Total Return Index Value Unhedged USD',
    'Bloomberg Global Agg Treasuries Total Return Index Value Unhedged USD',
    'J.P. Morgan EMBI Global Diversified Composite',
    'Bloomberg US Treasury Inflation Notes TR Index Value Unhedged USD',
    '10yr-2yr',
    'Brent',
    'WTI',
    'Gold',
    'Copper',
    'Commodity Index',
    'Bloomberg Agriculture Subindex',
    'Dollar',
    'Euro',
    'Jpy Yen',
    'GBP',
    'CNY',
    'JPMorgan Funds - Emerging Markets Debt Fund',
    'Bitcoin',
    'ETHUSD Curncy',
    'Bitwise 10 Crypto Index Fund',
    'MSCI US REIT Index',
    'FTSE EPRA NAREIT DEVELOPED Total Return Index USD',
    'iShares U.S. Infrastructure ETF'
]

# Data cleaning
print("\n[2/6] Cleaning quantitative data...")
numeric_cols = beta_columns + ['Correlation with SPX', '12 month volatility',
                               '1 Year Sharpe', '3 year sharpe',
                               '36 Month Return', '12 Month Return']

df_clean = df.copy()
for col in numeric_cols:
    if col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# Check for missing data
print(f"  Missing data summary:")
for col in beta_columns[:5]:
    missing = df_clean[col].isna().sum()
    if missing > 0:
        print(f"    {col}: {missing} missing ({missing/len(df_clean)*100:.1f}%)")

# Build beta matrix (fill NaN with 0 for distance calculation)
print("\n[3/6] Building 30-dimensional beta vectors...")
beta_matrix = df_clean[beta_columns].fillna(0).values
print(f"  Beta matrix shape: {beta_matrix.shape}")
print(f"  Standardizing betas...")
scaler = StandardScaler()
beta_matrix_scaled = scaler.fit_transform(beta_matrix)

# Calculate pairwise beta distances (Euclidean)
print("\n[4/6] Calculating beta vector similarities...")
print(f"  Computing {len(df_clean)} x {len(df_clean)} distance matrix...")
beta_distances = pdist(beta_matrix_scaled, metric='euclidean')
beta_dist_matrix = squareform(beta_distances)

# Normalize distances to [0,1] where 1 = identical, 0 = completely different
max_dist = beta_dist_matrix.max()
beta_similarity = 1 - (beta_dist_matrix / max_dist)

print(f"  Beta similarity range: {beta_similarity.min():.3f} to {beta_similarity.max():.3f}")

# Calculate additional signals for proxy scoring
print("\n[5/6] Computing proxy scoring signals...")

# Signal 1: Sharpe similarity (1Y and 3Y)
sharpe_1y = pd.to_numeric(df_clean['1 Year Sharpe'], errors='coerce').fillna(0).values
sharpe_3y = pd.to_numeric(df_clean['3 year sharpe'], errors='coerce').fillna(0).values
sharpe_avg = (sharpe_1y + sharpe_3y) / 2
sharpe_dist = np.abs(sharpe_avg[:, None] - sharpe_avg[None, :])
sharpe_max = sharpe_dist.max()
sharpe_similarity = 1 - (sharpe_dist / (sharpe_max + 0.001))  # Avoid division by zero

# Signal 2: Tier-2 match
tier2_match = (df_clean['category_tier2'].values[:, None] == df_clean['category_tier2'].values[None, :]).astype(float)

# Signal 3: Tag overlap (Jaccard similarity)
def tag_overlap(tags1, tags2):
    if pd.isna(tags1) or pd.isna(tags2):
        return 0
    set1 = set(str(tags1).split(', '))
    set2 = set(str(tags2).split(', '))
    if len(set1 | set2) == 0:
        return 0
    return len(set1 & set2) / len(set1 | set2)

tag_similarity = np.zeros((len(df_clean), len(df_clean)))
for i in range(len(df_clean)):
    for j in range(len(df_clean)):
        tag_similarity[i, j] = tag_overlap(df_clean.iloc[i]['category_tags'],
                                          df_clean.iloc[j]['category_tags'])

# Signal 4: Description similarity (simple: length similarity + keyword match)
desc_similarity = np.zeros((len(df_clean), len(df_clean)))
for i in range(len(df_clean)):
    for j in range(len(df_clean)):
        desc1 = str(df_clean.iloc[i]['Long_Description']).lower()
        desc2 = str(df_clean.iloc[j]['Long_Description']).lower()
        # Simple similarity: character overlap
        if len(desc1) > 0 and len(desc2) > 0:
            common = sum(1 for c in desc1 if c in desc2)
            desc_similarity[i, j] = min(common / max(len(desc1), len(desc2)), 1.0)

# Composite proxy score
print(f"  Fusing signals: beta(0.40) + tier2(0.20) + tags(0.15) + sharpe(0.15) + desc(0.10)")
proxy_score = (
    0.40 * beta_similarity +
    0.20 * tier2_match +
    0.15 * tag_similarity +
    0.15 * sharpe_similarity +
    0.10 * desc_similarity
)

# Find proxy groups using hierarchical clustering
print("\n[6/6] Clustering into proxy groups...")
print(f"  Running hierarchical clustering on {len(df_clean)} assets...")
# Convert similarity to distance for clustering
proxy_distance = 1 - proxy_score
np.fill_diagonal(proxy_distance, 0)
proxy_dist_condensed = squareform(proxy_distance)
linkage_matrix = linkage(proxy_dist_condensed, method='ward')

# Cut tree at threshold to form clusters
threshold = 0.75  # Groups with proxy_score > 0.75
clusters = fcluster(linkage_matrix, t=(1-threshold), criterion='distance')

df_clean['proxy_group'] = clusters
df_clean['proxy_score'] = proxy_score.max(axis=1)  # Max similarity to any other asset

# Analyze groups
print(f"\n{'='*80}")
print(f"DEDUPLICATION RESULTS")
print(f"{'='*80}")

n_groups = len(np.unique(clusters))
print(f"\nTotal proxy groups identified: {n_groups}")
print(f"Assets in single groups (unique): {(clusters == clusters).sum()}")

# Find significant proxy groups (>1 asset)
group_sizes = pd.Series(clusters).value_counts()
multi_groups = group_sizes[group_sizes > 1]
print(f"Groups with >1 asset (potential duplicates): {len(multi_groups)}")
print(f"Assets involved in duplicates: {multi_groups.sum()}")
print(f"Potential reduction: {multi_groups.sum() - len(multi_groups)} assets could be removed")

print(f"\nProxy group size distribution:")
print(group_sizes.value_counts().sort_index().head(10))

# Output: Top 30 proxy groups for review
print(f"\n{'='*80}")
print(f"TOP 30 PROXY GROUPS FOR REVIEW")
print(f"{'='*80}\n")

dedup_results = []
group_id = 1
for grp in multi_groups.sort_values(ascending=False).head(30).index:
    group_assets = df_clean[df_clean['proxy_group'] == grp].copy()
    group_assets['rank_in_group'] = range(1, len(group_assets) + 1)
    group_assets = group_assets.sort_values('1 Year Sharpe', ascending=False, na_position='last')

    print(f"\nGROUP {group_id}: {len(group_assets)} assets")
    print(f"  Tier-1: {group_assets.iloc[0]['category_tier1']} | Tier-2: {group_assets.iloc[0]['category_tier2']}")
    print(f"  Tags: {str(group_assets.iloc[0]['category_tags'])[:60]}...")

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
            'Avg_Beta_Distance': 0  # Placeholder
        })

    group_id += 1

# Save dedup report
dedup_df = pd.DataFrame(dedup_results)
dedup_df.to_excel("Dedup Report - Top 30 Groups.xlsx", index=False)

print(f"\n{'='*80}")
print(f"✓ DEDUP REPORT GENERATED")
print(f"{'='*80}")
print(f"File: Dedup Report - Top 30 Groups.xlsx")
print(f"Total assets in top 30 groups: {len(dedup_df)}")
print(f"Potential removals (rank > 1): {len(dedup_df[dedup_df['rank_in_group'] > 1])}")
print(f"Assets to KEEP: {len(dedup_df[dedup_df['action'] == 'KEEP'])}")

# Summary statistics
print(f"\n{'='*80}")
print(f"SUMMARY")
print(f"{'='*80}")
print(f"\nIf we remove all rank>1 assets from all {n_groups} groups:")
print(f"  Starting assets: {len(df_clean)}")
print(f"  Estimated removable: ~{(multi_groups.sum() - len(multi_groups))}")
print(f"  Estimated remaining: ~{len(df_clean) - (multi_groups.sum() - len(multi_groups))}")
print(f"\nThis is conservative - needs validation.")
print(f"Next steps: Review top groups, adjust thresholds if needed, then proceed to final 1000 selection.")

EOF
