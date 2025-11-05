# Bloomberg Indices Clustering and Filtering Program

This program intelligently filters a large list of Bloomberg Indices to a smaller, more representative set using clustering algorithms and correlation analysis.

## Input Files/Dependencies

- **Bloomberg Indices.xlsx**: Excel file containing the complete list of Bloomberg indices with return data
- **pandas**: For data manipulation and Excel file reading
- **numpy**: For numerical computations and array operations
- **scikit-learn**: For StandardScaler and AgglomerativeClustering
- **matplotlib**: For generating correlation heatmap visualization
- **seaborn**: For enhanced heatmap visualization
- **os**: For file system operations

## Output Files

- **Filtered Bloomberg Indices.xlsx**: Final filtered list of indices (target: 500) with cluster labels
- **Pre-Filtered Bloomberg Indices.xlsx**: Intermediate file generated when fixed income indices exceed the limit
- **filtered_indices_correlation.png**: Correlation heatmap visualization of the filtered indices returns

## Overview

The script reduces a comprehensive list of Bloomberg Indices (potentially thousands) to a targeted number (default: 500) while maintaining diversity across regions, security types, and market behaviors. The algorithm prioritizes maintaining different market patterns and reducing redundancy in the fixed income category.

## Key Features

1. **Intelligent Filtering**: Reduces large index lists to a manageable size while preserving market diversity
2. **Clustering Algorithm**: Uses hierarchical clustering to group similar indices by return patterns
3. **Correlation Analysis**: Identifies and removes highly correlated indices to avoid redundancy
4. **Representative Selection**: Chooses the most "average" index from each cluster as the representative
5. **Fixed Income Capping**: Limits fixed income indices (default: 100) to prevent overrepresentation
6. **Categorical Grouping**: Creates meaningful groups based on region and security type
7. **Market Behavior Classification**: Automatically categorizes indices by price patterns and volatility
8. **Visualization**: Generates correlation heatmap of filtered indices

## How It Works

### Stage 1: Data Loading and Preprocessing
- Loads Excel data containing Bloomberg Indices
- Converts return columns to numeric format
- Fills missing values with median values
- Displays initial distribution statistics

### Stage 2: Fixed Income Pre-filtering
- If fixed income indices exceed the limit (100), they are pre-clustered
- Hierarchical clustering groups similar fixed income indices
- One representative is selected from each cluster
- This ensures fixed income indices don't dominate the final list

### Stage 3: Main Filtering Algorithm
1. **Categorical Grouping**: Creates groups based on region + security type combinations
2. **Cluster Naming**: Generates descriptive names including:
   - Security type and region
   - Price movement pattern (Strong Bull, Bull, Mild Bull, Mild Bear, Bear, Strong Bear)
   - Volatility level (High Vol, Med Vol, Low Vol)
3. **Clustering Process**:
   - If fewer groups than target count: Selects one representative from each group
   - If more groups than target count: Performs hierarchical clustering within larger groups
4. **Representative Selection**: Identifies the index closest to each cluster's average return pattern

### Stage 4: Final Adjustment
- If too few indices selected: Adds more from underrepresented categories
- If too many indices selected: Keeps only the most unique ones (lowest correlation)
- Balances representation by region and security type

### Stage 5: Visualization and Output
- Generates a correlation heatmap of the filtered indices
- Saves results to Excel file
- Displays final statistics and distributions

## Input Requirements

### File Format
- Excel file with Bloomberg Indices data
- Must contain columns: 'Ticker', 'REGION_OR_COUNTRY', 'SECURITY_TYP'
- Must contain return columns: 
  - 'CHG_PCT_1D' (1 day change percentage)
  - 'CURRENT_TRR_1WK' (1 week total return)
  - 'CHG_PCT_YTD' (year-to-date change percentage)
  - 'CHG_PCT_1M' (1 month change percentage)
  - 'CHG_PCT_3YR' (3 year change percentage)
  - 'CHG_PCT_5YR' (5 year change percentage)
  - 'CHG_PCT_1YR' (1 year change percentage)
  - 'CHG_PCT_3QTR' (3 quarter change percentage)

### Sample Excel Structure

| Ticker | REGION_OR_COUNTRY | SECURITY_TYP | CHG_PCT_1D | CURRENT_TRR_1WK | CHG_PCT_YTD | CHG_PCT_1M | CHG_PCT_3YR | CHG_PCT_5YR | CHG_PCT_1YR | CHG_PCT_3QTR |
|--------|-------------------|--------------|-----------|------------------|------------|-----------|------------|------------|------------|--------------|
| SPX    | United States     | Equity Index | 0.5       | 1.2              | 10.3       | 2.1       | 15.4       | 20.1       | 12.7       | 8.2          |
| NDX    | United States     | Equity Index | -0.2      | 0.8              | 15.7       | 3.2       | 22.1       | 35.6       | 18.9       | 12.4         |

## Usage

### Basic Execution
```bash
python Bloomberg_Indices_Cluster.py
```

### Customizing Parameters
Edit the `filter_indices` function call in `main()` to change:
- `target_count`: Desired number of indices in final list (default: 500)
- `max_fixed_income`: Maximum number of fixed income indices (default: 100)

## Algorithm Details

### Price Pattern Classification
Based on weighted average of YTD (40%) and 1Y (60%) returns:
- Strong Bull: > 10%
- Bull: > 5%
- Mild Bull: > 0%
- Mild Bear: > -5%
- Bear: > -10%
- Strong Bear: ≤ -10%

### Volatility Classification
Based on standard deviation of returns:
- High Vol: Above 75th percentile of all standard deviations
- Med Vol: Between 25th and 75th percentiles
- Low Vol: Below 25th percentile

## Error Handling

- Gracefully handles missing values by filling with median
- Continues processing even if visualization fails
- Checks for file existence before processing
- Handles edge cases with small data groups

## Customization

To modify filtering behavior:
1. Adjust target count in `filter_indices()` call
2. Change max_fixed_income parameter
3. Modify return columns list if needed
4. Adjust clustering parameters in AgglomerativeClustering calls
5. Customize cluster naming logic in `create_cluster_name()` function
