"""
Filtering/ETF Cluster.py — ETF Cluster Filtering
------------------------------------------------

Purpose
- Reduce a large ETF master list to a representative subset (default 500) by
  grouping on categorical attributes and, when necessary, clustering on simple
  numerical similarity.

How It Works
- Stage 1: Build a composite `group_key` from
  `FUND_ASSET_CLASS_FOCUS|FUND_GEO_FOCUS|FUND_OBJECTIVE_LONG|FUND_STRATEGY` and
  assign a human‑readable `cluster_name` per row.
- Stage 2: If the number of unique groups exceeds the target, perform a
  lightweight sub‑clustering within each group using scaled numerical features
  (`CHG_PCT_1D`, `CURRENT_TRR_1WK`, `CHG_PCT_YTD`, `CHG_PCT_1YR`) and a
  Euclidean distance threshold, selecting the most traded ETF from each
  sub‑cluster.
- Ties and repeated `cluster_name`s are disambiguated by appending a counter
  suffix (e.g., "#2").

File
- `Filtering/ETF Cluster.py`

Inputs
- Input file (default): `ETF Master List.xlsx` (current working directory)
- Excel file must include at least these columns:
  - `Ticker`, `Name`, `Agg Traded Val (M USD)`
  - `FUND_ASSET_CLASS_FOCUS`, `FUND_GEO_FOCUS`, `FUND_OBJECTIVE_LONG`, `FUND_STRATEGY`
  - `CHG_PCT_1D`, `CURRENT_TRR_1WK`, `CHG_PCT_YTD`, `CHG_PCT_1YR`

Outputs
- Prints summary stats to stdout.
- Writes `Filtered ETF List.xlsx` with the selected rows and a `cluster_name`
  column for interpretability.

Usage
- Run the file directly. Input defaults to `ETF Master List.xlsx` in the
  current working directory; output is written to `Filtered ETF List.xlsx`.
  Adjust `file_path` or `target_count` in `main()` as needed.

Notes
- The numerical similarity threshold (`similarity_threshold = 1.0`) is a simple
  heuristic and may be tuned for your dataset.
- Missing values in key fields are filled with "Unknown" or 0 for numerics.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import os

def filter_etfs(file_path, target_count=500):
    """
    Filter an ETF master list to a representative subset.

    The function groups ETFs by key categorical attributes and, if needed,
    applies a simple distance-based numerical sub-clustering within each group
    to select the most liquid representative (by `Agg Traded Val (M USD)`).
    A human-readable `cluster_name` is created for interpretability.
    
    Args:
        file_path (str): Path to the Excel file containing ETF data.
        target_count (int): Target number of ETFs to return (default 500).
    
    Returns:
        pandas.DataFrame: Filtered ETF list including `cluster_name`.
    """
    # Load the data
    print(f"Loading ETF data from {file_path}...")
    df = pd.read_excel(file_path)
    
    # Handle any missing values
    df = df.fillna({
        'FUND_ASSET_CLASS_FOCUS': 'Unknown',
        'FUND_GEO_FOCUS': 'Unknown',
        'FUND_OBJECTIVE_LONG': 'Unknown',
        'FUND_STRATEGY': 'Unknown'
    })
    
    # Convert trading value to numeric
    df['Agg Traded Val (M USD)'] = pd.to_numeric(df['Agg Traded Val (M USD)'], errors='coerce')
    df['Agg Traded Val (M USD)'] = df['Agg Traded Val (M USD)'].fillna(0)
    
    # Convert performance columns to numeric
    numeric_cols = ['CHG_PCT_1D', 'CURRENT_TRR_1WK', 'CHG_PCT_YTD', 'CHG_PCT_1YR']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(0)
    
    # Print initial stats
    print(f"Initial ETF count: {len(df)}")
    print("\nDistribution by Asset Class:")
    print(df['FUND_ASSET_CLASS_FOCUS'].value_counts())
    print("\nDistribution by Geographic Focus:")
    print(df['FUND_GEO_FOCUS'].value_counts().head(10))  # Show top 10
    
    # Stage 1: Group by categorical attributes
    # Create a composite key for initial grouping
    df['group_key'] = (df['FUND_ASSET_CLASS_FOCUS'] + '|' + 
                       df['FUND_GEO_FOCUS'] + '|' + 
                       df['FUND_OBJECTIVE_LONG'] + '|' + 
                       df['FUND_STRATEGY'])
    
    # Create meaningful cluster names
    def create_cluster_name(row):
        """Create a concise, readable cluster name for an ETF row.

        Combines asset class, selective geography/objective/strategy, and a
        simple YTD performance label (e.g., "Bull", "Bear").
        """
        asset_class = row['FUND_ASSET_CLASS_FOCUS']
        geo_focus = row['FUND_GEO_FOCUS']
        objective = row['FUND_OBJECTIVE_LONG']
        strategy = row['FUND_STRATEGY']
        
        # Create base cluster name
        name_parts = []
        
        # Add asset class (always include)
        name_parts.append(asset_class)
        
        # Add geographic focus if not generic
        if geo_focus not in ['Global', 'Unknown', '#N/A N/A']:
            name_parts.append(geo_focus)
        
        # Add objective if meaningful
        if objective not in ['Unknown', '#N/A N/A', 'Broad Market']:
            # Shorten some common objectives
            if objective == 'Aggregate Bond':
                objective = 'Agg Bond'
            elif 'Inflation Protected' in objective:
                objective = 'TIPS'
            name_parts.append(objective)
        
        # Add strategy if meaningful and not "Blend"
        if strategy not in ['Unknown', '#N/A N/A', 'Blend']:
            name_parts.append(strategy)
            
        # Determine price movement pattern
        price_pattern = ""
        # YTD performance classification
        if row['CHG_PCT_YTD'] > 10:
            price_pattern = "Strong Bull"
        elif row['CHG_PCT_YTD'] > 5:
            price_pattern = "Bull"
        elif row['CHG_PCT_YTD'] > 0:
            price_pattern = "Mild Bull"
        elif row['CHG_PCT_YTD'] > -5:
            price_pattern = "Mild Bear"
        elif row['CHG_PCT_YTD'] > -10:
            price_pattern = "Bear"
        else:
            price_pattern = "Strong Bear"
            
        # Combine all parts
        cluster_name = " ".join(name_parts)
        
        # Add price pattern
        cluster_name = f"{cluster_name} ({price_pattern})"
        
        return cluster_name
    
    # Generate cluster names
    df['cluster_name'] = df.apply(create_cluster_name, axis=1)
    
    # Get initial groups
    initial_groups = df.groupby('group_key')
    print(f"\nNumber of unique category combinations: {len(initial_groups)}")
    
    # If we have fewer than target_count groups, we'll select all
    # If more, we need to further cluster within groups
    selected_etfs = []
    cluster_counter = {}  # To keep track of duplicate cluster names
    
    if len(initial_groups) <= target_count:
        # Just pick the highest trading value ETF from each group
        for _, group in initial_groups:
            best_etf_idx = group['Agg Traded Val (M USD)'].idxmax()
            best_etf = group.loc[best_etf_idx].copy()
            
            # Track cluster name counts
            cluster_name = best_etf['cluster_name']
            cluster_counter[cluster_name] = cluster_counter.get(cluster_name, 0) + 1
            if cluster_counter[cluster_name] > 1:
                best_etf['cluster_name'] = f"{cluster_name} #{cluster_counter[cluster_name]}"
                
            selected_etfs.append(best_etf)
        
        print(f"Selected {len(selected_etfs)} ETFs based on category grouping")
    else:
        # We need to do further clustering within larger groups
        # Normalize the numerical features for similarity calculation
        scaler = StandardScaler()
        cluster_id = 1
        
        for group_key, group in initial_groups:
            # If the group only has one ETF, just add it
            if len(group) == 1:
                etf = group.iloc[0].copy()
                selected_etfs.append(etf)
                continue
            
            # If the group is large, do numerical clustering
            if len(group) > 1:
                # Scale the numerical features
                group_numerical = group[numeric_cols].copy()
                if len(group_numerical) > 1:  # Need at least 2 rows for scaling
                    scaled_features = scaler.fit_transform(group_numerical)
                    
                    # Calculate distance matrix
                    distances = euclidean_distances(scaled_features)
                    
                    # Use a simple threshold-based clustering
                    similarity_threshold = 1.0  # Adjust as needed
                    
                    # Track which ETFs have been assigned to clusters
                    assigned = set()
                    sub_clusters = []
                    
                    # Create sub-clusters based on numerical similarity
                    for i in range(len(group)):
                        if i in assigned:
                            continue
                        
                        # Start a new cluster with this ETF
                        cluster = [i]
                        assigned.add(i)
                        
                        # Find similar ETFs
                        for j in range(len(group)):
                            if j != i and j not in assigned and distances[i, j] <= similarity_threshold:
                                cluster.append(j)
                                assigned.add(j)
                        
                        sub_clusters.append(cluster)
                    
                    # Select the highest trading value ETF from each sub-cluster
                    for sub_cluster in sub_clusters:
                        cluster_etfs = group.iloc[sub_cluster]
                        best_etf_idx = cluster_etfs['Agg Traded Val (M USD)'].idxmax()
                        best_etf = cluster_etfs.loc[best_etf_idx].copy()
                        
                        # Refine the cluster name for numerical sub-cluster
                        base_cluster_name = best_etf['cluster_name']
                        
                        # Add cluster number for disambiguation
                        cluster_counter[base_cluster_name] = cluster_counter.get(base_cluster_name, 0) + 1
                        if cluster_counter[base_cluster_name] > 1:
                            best_etf['cluster_name'] = f"{base_cluster_name} #{cluster_counter[base_cluster_name]}"
                        
                        selected_etfs.append(best_etf)
                        cluster_id += 1
                else:
                    # If we can't scale, just add the ETF with highest trading value
                    best_etf_idx = group['Agg Traded Val (M USD)'].idxmax()
                    best_etf = group.loc[best_etf_idx].copy()
                    selected_etfs.append(best_etf)
    
    # Convert to DataFrame
    result_df = pd.DataFrame(selected_etfs)
    
    # Adjust if we have too many or too few ETFs
    if len(result_df) > target_count:
        # Keep the ETFs with highest trading value
        result_df = result_df.sort_values('Agg Traded Val (M USD)', ascending=False).head(target_count)
        print(f"Too many ETFs, keeping top {target_count} by trading value")
    elif len(result_df) < target_count:
        # Add more ETFs from the largest initial groups
        needed = target_count - len(result_df)
        print(f"Too few ETFs, adding {needed} more from largest groups")
        
        # Sort groups by size
        group_sizes = initial_groups.size().sort_values(ascending=False)
        
        additional_etfs = []
        for group_key in group_sizes.index:
            if needed <= 0:
                break
                
            group = df[df['group_key'] == group_key]
            
            # Skip if group is small or we already selected from this group
            if len(group) <= 1:
                continue
                
            # Exclude ETFs already selected
            selected_tickers = set(result_df['Ticker'])
            group = group[~group['Ticker'].isin(selected_tickers)]
            
            if len(group) == 0:
                continue
                
            # Take up to 2 additional ETFs from each large group
            num_to_take = min(needed, 2, len(group))
            top_etfs = group.sort_values('Agg Traded Val (M USD)', ascending=False).head(num_to_take)
            
            # Update cluster names for additional ETFs
            for idx, row in top_etfs.iterrows():
                etf = row.copy()
                
                # Add suffix to cluster name
                base_cluster_name = etf['cluster_name']
                cluster_counter[base_cluster_name] = cluster_counter.get(base_cluster_name, 0) + 1
                if cluster_counter[base_cluster_name] > 1:
                    etf['cluster_name'] = f"{base_cluster_name} #{cluster_counter[base_cluster_name]}"
                
                additional_etfs.append(etf)
                
            needed -= num_to_take
            
        if additional_etfs:
            additional_df = pd.DataFrame(additional_etfs)
            result_df = pd.concat([result_df, additional_df])
    
    # Sort by ticker for output
    result_df = result_df.sort_values('Ticker')
    
    # Print final stats
    print(f"\nFinal ETF count: {len(result_df)}")
    print(f"Reduction: {len(df) - len(result_df)} ETFs removed ({(len(df) - len(result_df)) / len(df) * 100:.2f}%)")
    
    print("\nDistribution by Asset Class (after filtering):")
    print(result_df['FUND_ASSET_CLASS_FOCUS'].value_counts())
    
    print("\nTop 10 ETFs by trading value:")
    top_etfs = result_df.sort_values('Agg Traded Val (M USD)', ascending=False).head(10)
    for _, row in top_etfs.iterrows():
        print(f"{row['Ticker']}: {row['Name']} ({row['Agg Traded Val (M USD)']}M USD) - Cluster: {row['cluster_name']}")
    
    # Print sample of cluster names
    print("\nSample of cluster names:")
    for name in sorted(list(cluster_counter.keys()))[:10]:
        print(f"- {name}: {cluster_counter[name]} ETFs")
    
    return result_df

def main():
    """CLI entrypoint to filter the ETF master list and write output.

    - Validates the presence of `ETF Master List.xlsx` in the CWD
    - Calls `filter_etfs` with `target_count=500`
    - Saves results to `Filtered ETF List.xlsx`
    """
    # File path to the ETF master list
    file_path = "ETF Master List.xlsx"
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return
    
    # Filter ETFs down to about 500
    filtered_etfs = filter_etfs(file_path, target_count=500)
    
    # Save to new Excel file
    output_file = "Filtered ETF List.xlsx"
    filtered_etfs.to_excel(output_file, index=False)
    print(f"\nFiltered ETF list saved to {output_file}")

if __name__ == "__main__":
    main()
