# Financial Analytics Pipeline - Complete Workflow Guide

## Overview
This repository contains a comprehensive financial analytics pipeline that processes ~4,933 assets (ETFs, Bloomberg Indices, and Goldman Sachs baskets) through classification, deduplication, factor analysis, and strategic selection to create a diversified investment dataset.

## System Architecture
The pipeline follows a **data flow architecture** with three main stages:
1. **Data Collection & Filtering** - Raw data acquisition and intelligent filtering
2. **Classification & Integration** - Taxonomy assignment and dataset unification  
3. **Analysis & Selection** - Performance analytics and strategic portfolio optimization

---

## 🔄 Complete Program Execution Sequence

### **STAGE 1: DATA COLLECTION & FILTERING**

#### 1.1 Bloomberg Indices Processing
```bash
# Step 1: Filter Bloomberg Indices from raw dataset
python "Data Collection/Bloomberg_Indices_Cluster.py"
```
**INPUT:** `Data Collection/Bloomberg Indices.xlsx`
**OUTPUT:** `Data Collection/Filtered Bloomberg Indices.xlsx` (438 indices)
**PURPOSE:** Reduces thousands of Bloomberg indices to 500 most representative using correlation-based clustering

#### 1.2 ETF Processing  
```bash
# Step 2: Filter ETF master list
python "Data Collection/ETF Cluster.py"
```
**INPUT:** `Data Collection/ETF Master List.xlsx`
**OUTPUT:** `Data Collection/Filtered ETF List.xlsx` (500 ETFs)
**PURPOSE:** Reduces 1,619 ETFs to 500 most representative using categorical grouping and liquidity-based selection

#### 1.3 Goldman Sachs Data Retrieval
```bash
# Step 3: Fetch Goldman Sachs basket data from API
python "Data Collection/gs_basket_data.py"
```
**INPUT:** None (API-based data retrieval)
**OUTPUT:** `Data Collection/GSCB_FLAGSHIP_coverage_with_desc_ALL.xlsx` (2,667 baskets)
**PURPOSE:** Retrieves enriched basket data from Goldman Sachs Marquee API

---

### **STAGE 2: CLASSIFICATION & INTEGRATION**

#### 2.1 Individual Asset Classification
```bash
# Step 4: Classify filtered Bloomberg indices
python classify_bloomberg_full.py
```
**INPUT:** `Data Collection/Filtered Bloomberg Indices.xlsx`
**OUTPUT:** `Filtered Bloomberg Indices Classified.xlsx`
**PURPOSE:** Assigns unified taxonomy to Bloomberg indices using Haiku 4.5 LLM

```bash
# Step 5: Classify filtered ETFs
python classify_etfs_full.py
```
**INPUT:** `Data Collection/Filtered ETF List.xlsx`
**OUTPUT:** `ETF Master List Classified.xlsx`
**PURPOSE:** Assigns unified taxonomy to ETFs using Haiku 4.5 LLM

```bash
# Step 6: Classify Goldman Sachs baskets
python classify_goldman_full.py
```
**INPUT:** `Data Collection/GSCB_FLAGSHIP_coverage_with_desc_ALL.xlsx`
**OUTPUT:** `Data Collection/GSCB_FLAGSHIP_coverage_Classified.xlsx`
**PURPOSE:** Assigns unified taxonomy to Goldman baskets using Haiku 4.5 LLM

```bash
# Step 7: Classify specialized thematic ETFs
python classify_thematic_etfs_full.py
```
**INPUT:** `Thematic ETFs.xlsx`
**OUTPUT:** `Thematic ETFs Classified.xlsx`
**PURPOSE:** Specialized classification for thematic ETFs with enhanced taxonomy

#### 2.2 Dataset Integration
```bash
# Step 8: Merge all classified datasets into unified master file
python merge_classified_files.py
```
**INPUTS:** 
- `ETF Master List Classified.xlsx` (1,619 ETFs)
- `Filtered Bloomberg Indices Classified.xlsx` (438 indices)
- `Data Collection/GSCB_FLAGSHIP_coverage_Classified.xlsx` (2,667 baskets)
- `Thematic ETFs Classified.xlsx` (231 ETFs)

**OUTPUT:** `Master Asset List Classified.xlsx` (~4,933 assets)
**PURPOSE:** Creates unified master dataset with standardized structure and source tracking

---

### **STAGE 3: ANALYSIS & SELECTION**

#### 3.1 Performance Analytics
```bash
# Step 9: Category performance analysis
python p1_category_analysis.py
```
**INPUT:** `Master Asset List Classified.xlsx`
**OUTPUTS:** 
- `P1_Tier1_Analysis.xlsx` (Tier-1 category statistics)
- `P1_Tier2_Analysis.xlsx` (Tier-2 category analysis)
- `P1_Source_Analysis.xlsx` (Source distribution analysis)
**PURPOSE:** Analyzes performance metrics across asset categories and identifies coverage gaps

#### 3.2 Deduplication Analysis
```bash
# Step 10: Identify duplicate/proxy assets
python p0_deduplication_analysis.py
```
**INPUT:** `Master Asset List Classified.xlsx`
**OUTPUT:** `Dedup Report - Top 30 Groups.xlsx`
**PURPOSE:** Uses 30-dimensional beta vector clustering to identify highly similar assets for removal

#### 3.3 Factor Decomposition
```bash
# Step 11: Create enriched asset profiles
python p2_factor_decomposition.py
```
**INPUT:** `Master Asset List Classified.xlsx`
**OUTPUT:** `P2_Enriched_Asset_Profiles.xlsx` (4,934 assets with 19 analytical columns)
**PURPOSE:** Extracts factor exposures and creates ML-ready feature matrix with risk metrics

#### 3.4 Strategic Portfolio Selection
```bash
# Step 12: Final asset selection algorithm
python final_selection_algorithm.py
```
**INPUT:** `Master Asset List Classified.xlsx`
**OUTPUT:** `Final 1000 Asset Master List.xlsx`
**PURPOSE:** Reduces 4,933 assets to ~1,000 using strategic allocation targets and quality scoring

---

### **SPECIALIZED TOOLS (Optional)**

#### Meme Stock Discovery
```bash
# Optional: Real-time meme stock identification
python MemeFinder.py
```
**INPUT:** None (web-based data retrieval)
**OUTPUT:** `search_based_meme_stocks.json`
**PURPOSE:** Uses Exa search API and Yahoo Finance to identify trending meme stocks

---

## 📊 Data Flow Summary

```
RAW DATA SOURCES:
├── Bloomberg Indices.xlsx
├── ETF Master List.xlsx  
├── Goldman Sachs API (GSCB_FLAGSHIP)
└── Thematic ETFs.xlsx

↓ STAGE 1: FILTERING
├── Filtered Bloomberg Indices.xlsx (438)
├── Filtered ETF List.xlsx (500)
└── GSCB_FLAGSHIP_coverage_with_desc_ALL.xlsx (2,667)

↓ STAGE 2: CLASSIFICATION
├── Filtered Bloomberg Indices Classified.xlsx
├── ETF Master List Classified.xlsx
├── GSCB_FLAGSHIP_coverage_Classified.xlsx
└── Thematic ETFs Classified.xlsx

↓ INTEGRATION
└── Master Asset List Classified.xlsx (~4,933 assets)

↓ STAGE 3: ANALYSIS
├── P1_*_Analysis.xlsx (performance analytics)
├── Dedup Report - Top 30 Groups.xlsx
├── P2_Enriched_Asset_Profiles.xlsx (4,934 assets)
└── Final 1000 Asset Master List.xlsx

```

---

## 🔧 Prerequisites & Dependencies

### **Required Python Libraries:**
```bash
pip install pandas numpy scipy scikit-learn
pip install anthropic exa-py yfinance
pip install matplotlib seaborn openpyxl
pip install gs-quant  # For Goldman Sachs API access
```

### **API Keys Required:**
- **ANTHROPIC_API_KEY**: For Haiku 4.5 LLM classification
- **EXA_API_KEY**: For meme stock web search (optional)
- **Goldman Sachs API**: Embedded in source for basket data retrieval

### **Input Data Requirements:**
- `Bloomberg Indices.xlsx`: Raw Bloomberg indices with performance data
- `ETF Master List.xlsx`: Complete ETF database with metadata
- `Thematic ETFs.xlsx`: Specialized thematic ETF list

---

## 🎯 Key Outputs & Deliverables

### **Primary Deliverables:**
1. **`Final 1000 Asset Master List.xlsx`** - Optimized portfolio of ~1,000 diversified assets
2. **`P2_Enriched_Asset_Profiles.xlsx`** - Comprehensive asset profiles with factor exposures
3. **`Master Asset List Classified.xlsx`** - Unified dataset with taxonomy assignments

### **Analytical Reports:**
- **Performance Analytics:** Tier-1/Tier-2 category analysis and source distribution
- **Deduplication Report:** Proxy group identification and removal recommendations
- **Factor Analysis:** Market, size, geographic, and emerging market tilts

### **Classification Results:**
- Individual classified datasets for each source
- Unified taxonomy assignments across all asset types
- Progress tracking files for large classification jobs

---

## ⚡ Performance & Runtime

### **Estimated Execution Times:**
- **Data Collection:** 5-10 minutes (API-dependent)
- **Classification:** 30-60 minutes (LLM API rate limits)
- **Analysis & Selection:** 5-10 minutes
- **Total Pipeline:** 45-80 minutes

### **Resource Requirements:**
- **Memory:** 8GB+ recommended for large datasets
- **Storage:** ~500MB for all intermediate and final files
- **Network:** Stable internet connection for API calls

---

## 🔍 Quality Assurance

### **Data Validation:**
- Input file existence checks before processing
- Missing data handling with mean imputation
- Duplicate detection and removal during merging
- Progress saving and recovery for long-running processes

### **Error Handling:**
- Graceful API failure handling with retry logic
- Comprehensive logging and progress reporting
- Checkpoint files for process interruption recovery
- Input validation and format requirements

---

## 📈 Usage Examples

### **Complete Pipeline Execution:**
```bash
# Run entire pipeline in sequence
./run_complete_pipeline.sh
```

### **Individual Stage Execution:**
```bash
# Run only classification stage
python classify_bloomberg_full.py
python classify_etfs_full.py
python classify_goldman_full.py
python classify_thematic_etfs_full.py
python merge_classified_files.py
```

### **Analysis-Only Mode:**
```bash
# Skip data collection, use existing master file
python p1_category_analysis.py
python p0_deduplication_analysis.py
python p2_factor_decomposition.py
python final_selection_algorithm.py
```

---

## 📝 File Organization

```
News/
├── README.md                           # This file
├── Core Pipeline/
│   ├── p0_deduplication_analysis.py    # Proxy identification
│   ├── p1_category_analysis.py         # Performance analytics
│   ├── p2_factor_decomposition.py      # Factor analysis
│   └── final_selection_algorithm.py    # Portfolio optimization
├── Classification/
│   ├── unified_asset_classifier.py     # Unified taxonomy system
│   ├── etf_classifier.py               # ETF classification
│   ├── classify_bloomberg_full.py      # Bloomberg classification
│   ├── classify_goldman_full.py        # Goldman classification
│   ├── classify_thematic_etfs_full.py  # Thematic ETF classification
│   └── merge_classified_files.py       # Dataset integration
├── Data Collection/
│   ├── Bloomberg_Indices_Cluster.py     # Bloomberg filtering
│   ├── ETF Cluster.py                  # ETF filtering
│   ├── gs_basket_data.py               # Goldman API data
│   └── gs_basket_data_with_headings.py # Bloomberg integration
├── Tools/
│   └── MemeFinder.py                   # Meme stock discovery
└── Outputs/
    ├── Final 1000 Asset Master List.xlsx
    ├── P2_Enriched_Asset_Profiles.xlsx
    ├── Master Asset List Classified.xlsx
    └── [Various analytical reports]
```

---

## 🆘 Troubleshooting

### **Common Issues:**
1. **API Rate Limits:** Classification programs include automatic rate limiting and retry logic
2. **Missing Input Files:** Programs validate input file existence before execution
3. **Memory Issues:** Process large datasets in chunks with progress saving
4. **API Authentication:** Ensure valid API keys are properly configured

### **Recovery Procedures:**
- Use `*_PROGRESS.xlsx` files to resume interrupted classification jobs
- Individual programs can be run independently for targeted updates
- Check log outputs for detailed error information and guidance

---

## 📞 Support

For questions or issues with the pipeline:
1. Check individual program documentation for specific requirements
2. Verify API credentials and network connectivity
3. Review error logs for detailed troubleshooting information
4. Ensure sufficient system resources (memory, disk space)

---

**Last Updated:** November 6, 2025  
**Version:** 1.0.0  
**Maintainer:** Financial Analytics Team
