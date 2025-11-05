"""
Bloomberg Indices Clustering and Filtering Program

This program intelligently filters a large list of Bloomberg Indices to a smaller, more representative set 
using clustering algorithms and correlation analysis.

INPUT FILES:
- Bloomberg Indices.xlsx: Excel file containing the complete list of Bloomberg indices with return data

DEPENDENCIES:
- pandas: For data manipulation and Excel file reading
- numpy: For numerical computations and array operations
- scikit-learn: For StandardScaler and AgglomerativeClustering
- matplotlib: For generating correlation heatmap visualization
- seaborn: For enhanced heatmap visualization
- os: For file system operations

OUTPUT FILES:
- Filtered Bloomberg Indices.xlsx: Final filtered list of indices (target: 500) with cluster labels
- Pre-Filtered Bloomberg Indices.xlsx: Intermediate file generated when fixed income indices exceed the limit
- filtered_indices_correlation.png: Correlation heatmap visualization of the filtered indices returns

OVERVIEW:
The script reduces a comprehensive list of Bloomberg Indices (potentially thousands) to a targeted number 
(default: 500) while maintaining diversity across regions, security types, and market behaviors. The 
algorithm prioritizes maintaining different market patterns and reducing redundancy in the fixed income category.

USAGE:
Run from command line: python Bloomberg_Indices_Cluster.py

KEY FEATURES:
1. Intelligent Filtering: Reduces large index lists to a manageable size while preserving market diversity
2. Clustering Algorithm: Uses hierarchical clustering to group similar indices by return patterns
3. Correlation Analysis: Identifies and removes highly correlated indices to avoid redundancy
4. Representative Selection: Chooses the most "average" index from each cluster as the representative
5. Fixed Income Capping: Limits fixed income indices (default: 100) to prevent overrepresentation
6. Categorical Grouping: Creates meaningful groups based on region and security type
7. Market Behavior Classification: Automatically categorizes indices by price patterns and volatility
8. Visualization: Generates correlation heatmap of filtered indices

PARAMETERS:
- target_count: Desired number of indices in final list (default: 500)
- max_fixed_income: Maximum number of fixed income indices (default: 100)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances
import os
import matplotlib.pyplot as plt
import seaborn as sns

def filter_indices(file_path, target_count=500, max_fixed_income=100):
    """
    Filter Bloomberg Indices down to the most representative indices based on 
    correlation clustering and return patterns.
    
    Args:
        file_path (str): Path to Excel file containing Bloomberg Indices data
        target_count (int): Target number of indices in the filtered list
        max_fixed_income (int): Maximum number of fixed income indices to include
    
    Returns:
        pandas.DataFrame: Filtered indices list with cluster labels
    """
    # Load the data
    print(f"Loading Bloomberg Indices data from {file_path}...")
    df = pd.read_excel(file_path)
    
    # Print initial stats
    print(f"Initial indices count: {len(df)}")
    print("\nDistribution by Region:")
    print(df['REGION_OR_COUNTRY'].value_counts())
    print("\nDistribution by Security Type:")
    print(df['SECURITY_TYP'].value_counts())
    
    # Handle any missing values in return columns
    return_cols = ['CHG_PCT_1D', 'CURRENT_TRR_1WK', 'CHG_PCT_YTD', 'CHG_PCT_1M', 
                   'CHG_PCT_3YR', 'CHG_PCT_5YR', 'CHG_PCT_1YR', 'CHG_PCT_3QTR']
    
    # Convert return columns to numeric
    for col in return_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill missing values with median for each return column
    for col in return_cols:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)
    
    # Stage 1: Group by categorical attributes
    # Create a composite key for initial grouping
    df['group_key'] = df['REGION_OR_COUNTRY'] + '|' + df['SECURITY_TYP']
    
    # Create meaningful cluster names
    def create_cluster_name(row):
        region = row['REGION_OR_COUNTRY']
        security_type = row['SECURITY_TYP']
        
        # Create base cluster name
        name_parts = []
        
        # Add security type (always include)
        name_parts.append(security_type)
        
        # Add region if not generic
        if region not in ['Global', 'Unknown', '#N/A N/A', 'Multi-national']:
            name_parts.append(region)
        
        # Determine price movement pattern based on YTD and 1Y returns
        # This gives us a sense of both recent and longer-term performance
        ytd_return = row['CHG_PCT_YTD']
        yr1_return = row['CHG_PCT_1YR']
        
        # Weighted average of YTD (40%) and 1Y (60%) returns
        weighted_return = 0.4 * ytd_return + 0.6 * yr1_return
        
        if weighted_return > 10:
            price_pattern = "Strong Bull"
        elif weighted_return > 5:
            price_pattern = "Bull"
        elif weighted_return > 0:
            price_pattern = "Mild Bull"
        elif weighted_return > -5:
            price_pattern = "Mild Bear"
        elif weighted_return > -10:
            price_pattern = "Bear"
        else:
            price_pattern = "Strong Bear"
            
        # Add volatility indicator based on standard deviation of returns
        returns_std = np.std([row['CHG_PCT_1D'], row['CHG_PCT_1M'], 
                             row['CHG_PCT_YTD'], row['CHG_PCT_1YR']])
        
        if returns_std > np.percentile(df[['CHG_PCT_1D', 'CHG_PCT_1M', 
                                          'CHG_PCT_YTD', 'CHG_PCT_1YR']].std(axis=1), 75):
            volatility = "High Vol"
        elif returns_std < np.percentile(df[['CHG_PCT_1D', 'CHG_PCT_1M', 
                                            'CHG_PCT_YTD', 'CHG_PCT_1YR']].std(axis=1), 25):
            volatility = "Low Vol"
        else:
            volatility = "Med Vol"
        
        # Combine all parts
        cluster_name = " ".join(name_parts)
        
        # Add price pattern and volatility
        cluster_name = f"{cluster_name} ({price_pattern}, {volatility})"
        
        return cluster_name
    
    # Generate cluster names
    df['cluster_name'] = df.apply(create_cluster_name, axis=1)
    
    # Get initial groups
    initial_groups = df.groupby('group_key')
    print(f"\nNumber of unique category combinations: {len(initial_groups)}")
    
    # If we have fewer than target_count groups, we'll need to select all
    # If more, we need to further cluster within groups
    selected_indices = []
    cluster_counter = {}  # To keep track of duplicate cluster names
    
    if len(initial_groups) <= target_count:
        # Just pick the most representative index from each group
        for _, group in initial_groups:
            # For each group, we'll select the index with the most "average" return pattern
            # This is the one that best represents the group's typical behavior
            
            # Calculate the average return pattern for the group
            group_avg_returns = group[return_cols].mean()
            
            # Find the index with the closest return pattern to the group average
            min_distance = float('inf')
            representative_idx = None
            
            for idx, row in group.iterrows():
                # Calculate Euclidean distance to the group average
                distance = np.sqrt(((row[return_cols] - group_avg_returns) ** 2).sum())
                
                if distance < min_distance:
                    min_distance = distance
                    representative_idx = idx
            
            representative_index = group.loc[representative_idx].copy()
            
            # Track cluster name counts
            cluster_name = representative_index['cluster_name']
            cluster_counter[cluster_name] = cluster_counter.get(cluster_name, 0) + 1
            if cluster_counter[cluster_name] > 1:
                representative_index['cluster_name'] = f"{cluster_name} #{cluster_counter[cluster_name]}"
                
            selected_indices.append(representative_index)
        
        print(f"Selected {len(selected_indices)} indices based on category grouping")
    else:
        # We need to do further clustering within larger groups based on return correlations
        scaler = StandardScaler()
        
        for group_key, group in initial_groups:
            # If the group only has one index, just add it
            if len(group) == 1:
                index_data = group.iloc[0].copy()
                selected_indices.append(index_data)
                continue
            
            # If the group is large, do correlation-based clustering
            if len(group) > 1:
                # Scale the return features
                group_returns = group[return_cols].copy()
                if len(group_returns) > 1:  # Need at least 2 rows for scaling
                    # Calculate correlation matrix for this group
                    corr_matrix = group_returns.T.corr()
                    
                    # Convert correlations to distances (1 - correlation)
                    # Higher correlation = smaller distance
                    distance_matrix = 1 - corr_matrix.abs()
                    
                    # Determine number of clusters based on group size
                    n_clusters = min(max(2, len(group) // 5), 10)  # Between 2 and 10 clusters
                    
                    # Apply hierarchical clustering
                    clustering = AgglomerativeClustering(
                        n_clusters=n_clusters,
                        affinity='precomputed',
                        linkage='average'
                    )
                    
                    # Fit the clustering model
                    cluster_labels = clustering.fit_predict(distance_matrix)
                    
                    # Add cluster labels to the group dataframe
                    group_with_clusters = group.copy()
                    group_with_clusters['temp_cluster'] = cluster_labels
                    
                    # Select the most representative index from each cluster
                    for cluster_id in range(n_clusters):
                        cluster_indices = group_with_clusters[group_with_clusters['temp_cluster'] == cluster_id]
                        
                        if len(cluster_indices) == 0:
                            continue
                        
                        # Calculate the average return pattern for this cluster
                        cluster_avg_returns = cluster_indices[return_cols].mean()
                        
                        # Find the index with the closest return pattern to the cluster average
                        min_distance = float('inf')
                        representative_idx = None
                        
                        for idx, row in cluster_indices.iterrows():
                            # Calculate Euclidean distance to the cluster average
                            distance = np.sqrt(((row[return_cols] - cluster_avg_returns) ** 2).sum())
                            
                            if distance < min_distance:
                                min_distance = distance
                                representative_idx = idx
                        
                        representative_index = cluster_indices.loc[representative_idx].copy()
                        
                        # Refine the cluster name for numerical sub-cluster
                        base_cluster_name = representative_index['cluster_name']
                        
                        # Add cluster number for disambiguation
                        cluster_counter[base_cluster_name] = cluster_counter.get(base_cluster_name, 0) + 1
                        if cluster_counter[base_cluster_name] > 1:
                            representative_index['cluster_name'] = f"{base_cluster_name} #{cluster_counter[base_cluster_name]}"
                        
                        selected_indices.append(representative_index)
                else:
                    # If we can't cluster, just add the single index
                    index_data = group.iloc[0].copy()
                    selected_indices.append(index_data)
    
    # Convert to DataFrame
    result_df = pd.DataFrame(selected_indices)
    
    # First, strictly enforce the fixed income limit
    if 'Fixed Income Index' in result_df['SECURITY_TYP'].values:
        fixed_income_df = result_df[result_df['SECURITY_TYP'] == 'Fixed Income Index']
        if len(fixed_income_df) > max_fixed_income:
            print(f"\nReducing Fixed Income indices from {len(fixed_income_df)} to {max_fixed_income}")
            
            # Keep the most diverse fixed income indices
            # Calculate correlation matrix for fixed income indices
            fi_return_data = fixed_income_df[return_cols]
            
            # Standardize the data
            scaler = StandardScaler()
            fi_scaled = scaler.fit_transform(fi_return_data)
            
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(fi_scaled)
            
            # Calculate average correlation for each index (lower means more unique)
            avg_correlations = np.mean(np.abs(corr_matrix), axis=1)
            fixed_income_df['uniqueness_score'] = 1 - avg_correlations
            
            # Sort by uniqueness and keep top max_fixed_income
            fixed_income_df = fixed_income_df.sort_values('uniqueness_score', ascending=False).head(max_fixed_income)
            fixed_income_df = fixed_income_df.drop('uniqueness_score', axis=1)
            
            # Combine with non-fixed income indices
            non_fi_df = result_df[result_df['SECURITY_TYP'] != 'Fixed Income Index']
            result_df = pd.concat([fixed_income_df, non_fi_df])
    
    # Adjust if we have too many or too few indices
    if len(result_df) > target_count:
        # We need to reduce the number of indices
        # Calculate a "uniqueness score" for each index based on its correlation with others
        
        # First, standardize the return columns
        scaled_returns = StandardScaler().fit_transform(result_df[return_cols])
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(scaled_returns)
        
        # Calculate average correlation for each index
        avg_correlations = np.mean(np.abs(corr_matrix), axis=1)
        
        # Lower average correlation = more unique
        result_df['uniqueness_score'] = 1 - avg_correlations
        
        # Keep indices with highest uniqueness scores
        result_df = result_df.sort_values('uniqueness_score', ascending=False).head(target_count)
        result_df = result_df.drop('uniqueness_score', axis=1)
        
        print(f"Too many indices, keeping top {target_count} most unique indices")
    elif len(result_df) < target_count:
        # Add more indices from the largest initial groups
        needed = target_count - len(result_df)
        print(f"Too few indices, adding {needed} more from largest groups")
        
        # First, analyze the original distribution to set targets for each security type
        original_security_type_dist = df['SECURITY_TYP'].value_counts(normalize=True)
        target_security_counts = {}
        
        # Apply fixed income limit if specified
        fixed_income_count = result_df[result_df['SECURITY_TYP'] == 'Fixed Income Index'].shape[0]
        fixed_income_limit = max(0, min(max_fixed_income - fixed_income_count, needed))
        
        # Calculate target counts for each security type based on original distribution
        # but with the fixed income limit applied
        remaining_needed = needed - fixed_income_limit
        remaining_proportion_sum = sum([prop for type_, prop in original_security_type_dist.items() 
                                      if type_ != 'Fixed Income Index'])
        
        # Set fixed income target
        target_security_counts['Fixed Income Index'] = fixed_income_limit
        
        # Distribute remaining needed indices proportionally among other security types
        for sec_type, proportion in original_security_type_dist.items():
            if sec_type == 'Fixed Income Index':
                continue
                
            # Normalize the proportion relative to non-fixed income types
            normalized_proportion = proportion / remaining_proportion_sum if remaining_proportion_sum > 0 else 0
            target_count_for_type = int(remaining_needed * normalized_proportion)
            current_count = result_df[result_df['SECURITY_TYP'] == sec_type].shape[0]
            target_security_counts[sec_type] = max(0, target_count_for_type)
        
        print("\nTarget distribution by security type:")
        for sec_type, count in target_security_counts.items():
            print(f"- {sec_type}: {count} additional indices needed")
        
        # Now do the same for regions
        original_region_dist = df['REGION_OR_COUNTRY'].value_counts(normalize=True)
        target_region_counts = {}
        
        # Calculate target counts for each region based on original distribution
        for region, proportion in original_region_dist.items():
            target_count_for_region = int(target_count * proportion)
            current_count = result_df[result_df['REGION_OR_COUNTRY'] == region].shape[0]
            target_region_counts[region] = max(0, target_count_for_region - current_count)
        
        # Adjust to ensure we get exactly the needed number of indices
        total_targeted = sum(target_region_counts.values())
        if total_targeted < needed:
            # Distribute the remaining needed indices proportionally
            for region in sorted(target_region_counts.keys(), 
                               key=lambda x: original_region_dist.get(x, 0), 
                               reverse=True):
                if needed - total_targeted > 0:
                    target_region_counts[region] += 1
                    total_targeted += 1
                else:
                    break
        elif total_targeted > needed:
            # Reduce counts proportionally
            for region in sorted(target_region_counts.keys(), 
                               key=lambda x: original_region_dist.get(x, 0)):
                if total_targeted > needed and target_region_counts[region] > 0:
                    target_region_counts[region] -= 1
                    total_targeted -= 1
                if total_targeted == needed:
                    break
        
        print("\nTarget distribution by region:")
        for region, count in target_region_counts.items():
            print(f"- {region}: {count} additional indices needed")
        
        # Now select additional indices to meet both security type and region targets
        additional_indices = []
        
        # First, prioritize meeting security type targets
        for sec_type, target_count_for_type in target_security_counts.items():
            if target_count_for_type <= 0:
                continue
                
            # Get all indices of this security type that haven't been selected yet
            indices_of_type = df[df['SECURITY_TYP'] == sec_type]
            
            # Exclude indices already selected
            selected_tickers = set(result_df['Ticker'])
            indices_of_type = indices_of_type[~indices_of_type['Ticker'].isin(selected_tickers)]
            
            if len(indices_of_type) == 0:
                continue
                
            # Sort by different return patterns to get diverse indices
            # Use YTD return as a simple differentiator
            num_to_take = min(target_count_for_type, len(indices_of_type))
            
            # Take a diverse sample across the YTD return spectrum
            step = max(1, len(indices_of_type) // num_to_take)
            sorted_indices = indices_of_type.sort_values('CHG_PCT_YTD')
            selected_from_type = sorted_indices.iloc[::step][:num_to_take]
            
            # If we didn't get enough, just take the first few
            if len(selected_from_type) < num_to_take:
                remaining = num_to_take - len(selected_from_type)
                remaining_indices = indices_of_type[~indices_of_type.index.isin(selected_from_type.index)].head(remaining)
                selected_from_type = pd.concat([selected_from_type, remaining_indices])
            
            # Update cluster names for additional indices
            for idx, row in selected_from_type.iterrows():
                index_data = row.copy()
                
                # Add suffix to cluster name
                base_cluster_name = index_data['cluster_name']
                cluster_counter[base_cluster_name] = cluster_counter.get(base_cluster_name, 0) + 1
                if cluster_counter[base_cluster_name] > 1:
                    index_data['cluster_name'] = f"{base_cluster_name} #{cluster_counter[base_cluster_name]}"
                
                additional_indices.append(index_data)
            
            # Update needed counts
            needed -= len(selected_from_type)
            target_security_counts[sec_type] -= len(selected_from_type)
            
            # Also update region counts
            for region in target_region_counts:
                count_from_region = len(selected_from_type[selected_from_type['REGION_OR_COUNTRY'] == region])
                target_region_counts[region] = max(0, target_region_counts[region] - count_from_region)
        
        # If we still need more indices, try to balance regions
        if needed > 0:
            print(f"\nStill need {needed} more indices to balance regions")
            
            for region, target_count_for_region in target_region_counts.items():
                if target_count_for_region <= 0 or needed <= 0:
                    continue
                    
                # Get all indices from this region that haven't been selected yet
                indices_from_region = df[df['REGION_OR_COUNTRY'] == region]
                
                # Exclude indices already selected
                selected_tickers = set(result_df['Ticker']) | set(pd.DataFrame(additional_indices)['Ticker'] if additional_indices else [])
                indices_from_region = indices_from_region[~indices_from_region['Ticker'].isin(selected_tickers)]
                
                if len(indices_from_region) == 0:
                    continue
                    
                # Take a diverse sample across security types and return patterns
                num_to_take = min(target_count_for_region, len(indices_from_region), needed)
                
                # First try to balance security types within this region
                selected_from_region = []
                region_sec_types = indices_from_region['SECURITY_TYP'].value_counts(normalize=True)
                
                for sec_type, proportion in region_sec_types.items():
                    sec_type_count = max(1, int(num_to_take * proportion))
                    indices_of_type_from_region = indices_from_region[indices_from_region['SECURITY_TYP'] == sec_type]
                    
                    if len(indices_of_type_from_region) == 0:
                        continue
                        
                    # Take a diverse sample by YTD return
                    step = max(1, len(indices_of_type_from_region) // sec_type_count)
                    sorted_indices = indices_of_type_from_region.sort_values('CHG_PCT_YTD')
                    selected_of_type = sorted_indices.iloc[::step][:sec_type_count]
                    
                    selected_from_region.append(selected_of_type)
                
                # Combine all selected indices from this region
                if selected_from_region:
                    selected_from_region_df = pd.concat(selected_from_region)
                    
                    # Limit to the number we need
                    selected_from_region_df = selected_from_region_df.head(num_to_take)
                    
                    # Update cluster names for additional indices
                    for idx, row in selected_from_region_df.iterrows():
                        index_data = row.copy()
                        
                        # Add suffix to cluster name
                        base_cluster_name = index_data['cluster_name']
                        cluster_counter[base_cluster_name] = cluster_counter.get(base_cluster_name, 0) + 1
                        if cluster_counter[base_cluster_name] > 1:
                            index_data['cluster_name'] = f"{base_cluster_name} #{cluster_counter[base_cluster_name]}"
                        
                        additional_indices.append(index_data)
                    
                    # Update needed counts
                    needed -= len(selected_from_region_df)
                    target_region_counts[region] -= len(selected_from_region_df)
        
        # If we still need more indices, just take any remaining
        if needed > 0:
            print(f"\nStill need {needed} more indices to reach target count")
            
            # Get all indices that haven't been selected yet
            selected_tickers = set(result_df['Ticker']) | set(pd.DataFrame(additional_indices)['Ticker'] if additional_indices else [])
            remaining_indices = df[~df['Ticker'].isin(selected_tickers)]
            
            if len(remaining_indices) > 0:
                # Take a diverse sample
                num_to_take = min(needed, len(remaining_indices))
                
                # Sort by different return patterns and take a diverse sample
                step = max(1, len(remaining_indices) // num_to_take)
                sorted_indices = remaining_indices.sort_values('CHG_PCT_YTD')
                final_additional = sorted_indices.iloc[::step][:num_to_take]
                
                # Update cluster names for additional indices
                for idx, row in final_additional.iterrows():
                    index_data = row.copy()
                    
                    # Add suffix to cluster name
                    base_cluster_name = index_data['cluster_name']
                    cluster_counter[base_cluster_name] = cluster_counter.get(base_cluster_name, 0) + 1
                    if cluster_counter[base_cluster_name] > 1:
                        index_data['cluster_name'] = f"{base_cluster_name} #{cluster_counter[base_cluster_name]}"
                    
                    additional_indices.append(index_data)
        
        if additional_indices:
            additional_df = pd.DataFrame(additional_indices)
            result_df = pd.concat([result_df, additional_df])
    
    # Sort by ticker for output
    result_df = result_df.sort_values('Ticker')
    
    # Print final stats
    print(f"\nFinal indices count: {len(result_df)}")
    print(f"Reduction: {len(df) - len(result_df)} indices removed ({(len(df) - len(result_df)) / len(df) * 100:.2f}%)")
    
    print("\nDistribution by Region (after filtering):")
    print(result_df['REGION_OR_COUNTRY'].value_counts())
    
    print("\nDistribution by Security Type (after filtering):")
    print(result_df['SECURITY_TYP'].value_counts())
    
    # Print sample of cluster names
    print("\nSample of cluster names:")
    for name in sorted(list(cluster_counter.keys()))[:10]:
        print(f"- {name}: {cluster_counter[name]} indices")
    
    # Generate a correlation heatmap of the filtered indices
    try:
        plt.figure(figsize=(12, 10))
        corr_matrix = result_df[return_cols].corr()
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False)
        plt.title('Correlation Heatmap of Filtered Indices Returns')
        plt.tight_layout()
        plt.savefig('filtered_indices_correlation.png')
        print("\nCorrelation heatmap saved as 'filtered_indices_correlation.png'")
    except Exception as e:
        print(f"Could not generate correlation heatmap: {e}")
    
    return result_df

def main():
    # File path to the Bloomberg Indices list
    file_path = "Bloomberg Indices.xlsx"
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return
    
    # Filter indices down to about 500, with max 100 fixed income indices
    # Force the limit to be applied before any other selection
    all_indices = pd.read_excel(file_path)
    fixed_income_indices = all_indices[all_indices['SECURITY_TYP'] == 'Fixed Income Index']
    
    # If we have more than max_fixed_income fixed income indices, pre-filter them
    max_fixed_income = 100
    if len(fixed_income_indices) > max_fixed_income:
        print(f"Pre-filtering Fixed Income indices from {len(fixed_income_indices)} to {max_fixed_income}")
        
        # Convert return columns to numeric
        return_cols = ['CHG_PCT_1D', 'CURRENT_TRR_1WK', 'CHG_PCT_YTD', 'CHG_PCT_1M', 
                       'CHG_PCT_3YR', 'CHG_PCT_5YR', 'CHG_PCT_1YR', 'CHG_PCT_3QTR']
        
        for col in return_cols:
            fixed_income_indices[col] = pd.to_numeric(fixed_income_indices[col], errors='coerce')
            fixed_income_indices[col] = fixed_income_indices[col].fillna(fixed_income_indices[col].median())
        
        # Cluster fixed income indices by return patterns
        scaler = StandardScaler()
        scaled_returns = scaler.fit_transform(fixed_income_indices[return_cols])
        
        # Calculate distance matrix based on correlation
        corr_matrix = np.corrcoef(scaled_returns)
        distance_matrix = 1 - np.abs(corr_matrix)  # Convert correlation to distance
        
        # Determine number of clusters
        n_clusters = max_fixed_income
        
        # Apply hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )
        
        # Fit the clustering model
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        # Add cluster labels to the dataframe
        fixed_income_indices['cluster'] = cluster_labels
        
        # Select one representative from each cluster
        selected_indices = []
        for cluster_id in range(n_clusters):
            cluster_members = fixed_income_indices[fixed_income_indices['cluster'] == cluster_id]
            
            if len(cluster_members) == 0:
                continue
                
            # Select the index with return pattern closest to cluster average
            cluster_avg = cluster_members[return_cols].mean()
            min_distance = float('inf')
            representative_idx = None
            
            for idx, row in cluster_members.iterrows():
                distance = np.sqrt(((row[return_cols] - cluster_avg) ** 2).sum())
                if distance < min_distance:
                    min_distance = distance
                    representative_idx = idx
            
            if representative_idx is not None:
                selected_indices.append(representative_idx)
        
        # Filter the original dataframe to keep only selected fixed income indices
        # and all non-fixed income indices
        filtered_indices = all_indices[
            (~all_indices['SECURITY_TYP'].eq('Fixed Income Index')) | 
            (all_indices.index.isin(selected_indices))
        ]
        
        # Save the pre-filtered file
        pre_filtered_file = "Pre-Filtered Bloomberg Indices.xlsx"
        filtered_indices.to_excel(pre_filtered_file, index=False)
        print(f"Pre-filtered indices saved to {pre_filtered_file}")
        
        # Use the pre-filtered file for clustering
        file_path = pre_filtered_file
    
    # Now run the main clustering algorithm
    filtered_indices = filter_indices(file_path, target_count=500, max_fixed_income=max_fixed_income)
    
    # Save to new Excel file
    output_file = "Filtered Bloomberg Indices.xlsx"
    filtered_indices.to_excel(output_file, index=False)
    print(f"\nFiltered Bloomberg Indices list saved to {output_file}")

if __name__ == "__main__":
    main()
