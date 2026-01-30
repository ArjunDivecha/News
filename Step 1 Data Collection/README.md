# News Data Collection Pipeline

This folder contains the complete data collection pipeline for gathering and processing financial market data from multiple sources including Bloomberg, Goldman Sachs, and ETF providers. The pipeline is designed to create comprehensive, filtered datasets suitable for news analysis and market monitoring.

## 📁 Overview

The data collection process consists of three main components:

1. **Bloomberg Indices Clustering** - Filters and selects representative market indices
2. **ETF Cluster Filtering** - Creates optimized ETF datasets with categorical grouping
3. **Goldman Sachs Basket Data** - Retrieves and enriches GS basket coverage data

## 🔄 Data Collection Process

### Stage 1: Bloomberg Indices Processing
**Script**: `Bloomberg_Indices_Cluster.py`

**Purpose**: Reduce comprehensive Bloomberg indices list to a representative subset while maintaining market diversity.

**Input**: 
- `Bloomberg Indices.xlsx` - Raw Bloomberg indices data with performance metrics

**Process**:
1. **Data Loading**: Loads Excel data and converts return columns to numeric format
2. **Fixed Income Pre-filtering**: Limits fixed income indices to prevent overrepresentation (max 100)
3. **Categorical Grouping**: Creates groups based on region + security type combinations
4. **Hierarchical Clustering**: Groups similar indices by return patterns using AgglomerativeClustering
5. **Representative Selection**: Chooses the most "average" index from each cluster
6. **Market Classification**: Automatically categorizes indices by price patterns and volatility
7. **Correlation Analysis**: Removes highly correlated indices to avoid redundancy

**Output**:
- `Filtered Bloomberg Indices.xlsx` - Final filtered list (target: 500 indices)
- `Pre-Filtered Bloomberg Indices.xlsx` - Intermediate file when fixed income exceeds limits
- `filtered_indices_correlation.png` - Correlation heatmap visualization

### Stage 2: ETF Dataset Optimization
**Script**: `ETF Cluster.py`

**Purpose**: Create optimized ETF subset maintaining diversity across asset classes, regions, and strategies.

**Input**:
- `ETF Master List.xlsx` - Complete ETF master list with fund data and performance metrics

**Process**:
1. **Categorical Grouping**: Builds composite group keys from fund attributes
2. **Cluster Naming**: Creates human-readable names based on asset class, geography, objectives, and strategy
3. **Liquidity-Based Selection**: Prioritizes ETFs with highest trading volume within groups
4. **Lightweight Sub-clustering**: When groups exceed target, performs numerical similarity clustering
5. **Disambiguation**: Handles ties and repeated cluster names with counter suffixes

**Output**:
- `Filtered ETF List.xlsx` - Optimized ETF subset (target: 500) with cluster labels

### Stage 3: Goldman Sachs Data Enrichment
**Script**: `gs_basket_data.py`

**Purpose**: Retrieve and enrich Goldman Sachs basket coverage data with detailed metadata.

**Input**: None (data fetched directly from Goldman Sachs Marquee API)

**Process**:
1. **API Connection**: Connects to GSCB_FLAGSHIP dataset via Marquee API
2. **Coverage Retrieval**: Fetches basic basket identifiers and coverage data
3. **Asset Resolution**: Builds ticker-to-asset-ID mapping
4. **Metadata Enrichment**: Retrieves detailed descriptions and metadata for each basket
5. **Batch Processing**: Processes data in batches of 200 to comply with API rate limits
6. **Standardization**: Creates consistent column structure for downstream processing

**Output**:
- `GSCB_FLAGSHIP_coverage_with_desc_ALL.xlsx` - Enriched GS basket data with full descriptions
- `GSCB_FLAGSHIP_coverage_Classified.xlsx` - Classified version of the coverage data
- `GSCB_FLAGSHIP_coverage_Classified PROGRESS.xlsx` - Work-in-progress version

## 📊 Key Features

### Intelligent Filtering
- **Market Diversity**: Maintains representation across different regions, security types, and market behaviors
- **Redundancy Reduction**: Uses correlation analysis to avoid duplicate market exposure
- **Liquidity Prioritization**: Selects most liquid instruments within each category

### Advanced Clustering
- **Hierarchical Clustering**: Groups similar financial instruments by return patterns
- **Categorical Intelligence**: Creates meaningful groups based on fund attributes and market characteristics
- **Dynamic Classification**: Automatically categorizes by price movement patterns and volatility levels

### Data Quality
- **Missing Data Handling**: Fills missing values with median calculations
- **Error Handling**: Comprehensive error handling and progress reporting
- **Validation**: Data quality checks and validation steps throughout the pipeline

## 🛠️ Technical Requirements

### Python Dependencies
- **pandas**: Data manipulation and Excel file operations
- **numpy**: Numerical computations and array operations
- **scikit-learn**: Machine learning algorithms (StandardScaler, AgglomerativeClustering)
- **matplotlib**: Visualization and plotting
- **gs_quant**: Goldman Sachs API integration

### System Requirements
- **Memory**: Optimized for high-memory systems (128GB recommended)
- **Processing**: Parallel processing capabilities utilized
- **Storage**: Sufficient space for large Excel files and intermediate datasets

## 📁 File Structure

```
Data Collection/
├── README.md                              # This documentation
├── Bloomberg_Indices_Cluster.py           # Bloomberg indices processing script
├── Bloomberg_Indices_Cluster_documentation.md  # Detailed Bloomberg documentation
├── ETF Cluster.py                         # ETF clustering and filtering script
├── gs_basket_data.py                      # Goldman Sachs data retrieval script
├── gs_basket_data_with_headings.py        # Enhanced GS data script with headings
├── Bloomberg Indices.xlsx                 # Raw Bloomberg indices data (input)
├── ETF Master List.xlsx                   # Raw ETF master list (input)
├── Filtered Bloomberg Indices.xlsx        # Processed Bloomberg indices (output)
├── Pre-Filtered Bloomberg Indices.xlsx    # Intermediate Bloomberg data
├── Filtered ETF List.xlsx                 # Processed ETF list (output)
├── GSCB_FLAGSHIP_coverage_with_desc_ALL.xlsx    # Enriched GS data (output)
├── GSCB_FLAGSHIP_coverage_Classified.xlsx       # Classified GS data
├── GSCB_FLAGSHIP_coverage_Classified PROGRESS.xlsx  # Work-in-progress GS data
├── Betas from Bloomberg.xlsx              # Bloomberg beta data
└── filtered_indices_correlation.png       # Correlation visualization
```

## 🚀 Usage Instructions

### Running the Complete Pipeline

1. **Prepare Input Files**:
   - Ensure `Bloomberg Indices.xlsx` is present with required columns
   - Ensure `ETF Master List.xlsx` is present with fund data
   - Set up Goldman Sachs API credentials in `gs_basket_data.py`

2. **Execute Scripts in Order**:
   ```bash
   # Step 1: Process Bloomberg Indices
   python Bloomberg_Indices_Cluster.py
   
   # Step 2: Process ETF Data
   python "ETF Cluster.py"
   
   # Step 3: Retrieve Goldman Sachs Data
   python gs_basket_data.py
   ```

3. **Monitor Progress**:
   - Each script provides detailed progress reporting
   - Check console output for processing statistics
   - Verify output files are generated correctly

### Customization Options

#### Bloomberg Indices Parameters
```python
# In Bloomberg_Indices_Cluster.py, modify:
target_count = 500          # Desired number of indices
max_fixed_income = 100      # Maximum fixed income indices
```

#### ETF Clustering Parameters
```python
# In ETF Cluster.py, modify:
target_count = 500          # Desired number of ETFs
```

#### Goldman Sachs API Settings
```python
# In gs_basket_data.py, modify:
batch_size = 200            # API batch processing size
```

## 📈 Output Specifications

### Filtered Bloomberg Indices.xlsx
- **Target Size**: 500 indices (configurable)
- **Columns**: Original data + cluster_name for interpretability
- **Diversity**: Balanced representation across regions and security types
- **Quality**: Low correlation between selected indices

### Filtered ETF List.xlsx
- **Target Size**: 500 ETFs (configurable)
- **Columns**: Original data + cluster_name with categorical information
- **Liquidity**: Highest trading volume ETFs within each category
- **Coverage**: Diverse asset classes, regions, and strategies

### Goldman Sachs Coverage Files
- **Format**: Excel with standardized column headings
- **Metadata**: Full descriptions and asset information
- **Structure**: Ticker symbols, asset IDs, names, and financial metrics

## 🔍 Data Quality & Validation

### Automated Checks
- **Missing Data**: Automatically filled with mean/median values
- **Data Types**: Ensured numeric conversion for performance metrics
- **Correlation Analysis**: Visual and statistical validation of diversity
- **Progress Tracking**: Detailed logging of processing steps

### Manual Validation Steps
1. **Review Output Statistics**: Check distribution of regions, security types, and categories
2. **Visual Inspection**: Examine correlation heatmaps for redundancy
3. **Sample Verification**: Validate representative selections within clusters
4. **API Response Check**: Confirm Goldman Sachs data completeness

## 🐛 Troubleshooting

### Common Issues

**Missing Input Files**
- Ensure all required Excel files are present in the directory
- Verify file paths match those specified in scripts

**API Connection Errors**
- Check Goldman Sachs API credentials are valid
- Verify network connectivity and API rate limits

**Memory Issues**
- Reduce batch sizes for processing large datasets
- Close unnecessary applications to free memory

**Clustering Errors**
- Verify input data contains required columns
- Check for sufficient data points for clustering algorithms

### Error Recovery
- Scripts include comprehensive error handling
- Intermediate files allow resuming from checkpoints
- Progress logs help identify failure points

## 📝 Version History

- **v1.0.0** (2025-10-10): Initial Goldman Sachs data retrieval
- **v1.1.0** (2025-10-16): Bloomberg and ETF clustering implementation
- **v1.2.0** (2025-10-17): Enhanced filtering and correlation analysis
- **v1.3.0** (2025-11-06): Comprehensive documentation and visualization features

## 📞 Support

For technical issues or questions about the data collection pipeline:

1. Check this documentation for common solutions
2. Review individual script documentation files
3. Examine console output for specific error messages
4. Verify input data formats and requirements

---

**Last Updated**: November 6, 2025  
**System Requirements**: M4 Max Mac with 128GB RAM (optimized for high-memory, parallel processing)  
**Data Format**: Excel (.xlsx) for all input/output files  
**Documentation Level**: Comprehensive (10th-grade readable)
