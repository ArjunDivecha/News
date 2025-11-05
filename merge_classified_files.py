"""Merge all 4 classified datasets into one master file"""
import pandas as pd
import os

print("="*80)
print("MERGING 4 CLASSIFIED DATASETS INTO MASTER FILE")
print("="*80)

# Define file paths
etf_path = "ETF Master List Classified.xlsx"
bloomberg_path = "Filtered Bloomberg Indices Classified.xlsx"
goldman_path = "Data Collection/GSCB_FLAGSHIP_coverage_Classified.xlsx"
thematic_path = "Thematic ETFs Classified.xlsx"

dfs = []

# 1. ETF Master List
print("\n[1/4] Loading ETF Master List...")
df_etf = pd.read_excel(etf_path)
df_etf_clean = pd.DataFrame({
    'Bloomberg_Ticker': df_etf['Ticker'],
    'Name': df_etf['Name'],
    'Long_Description': df_etf['CIE_DES'],
    'category_tier1': df_etf['category_tier1'],
    'category_tier2': df_etf['category_tier2'],
    'category_tags': df_etf['category_tags'],
    'source': 'ETF'
})
dfs.append(df_etf_clean)
print(f"   Loaded {len(df_etf_clean)} rows from ETF Master List")

# 2. Bloomberg Indices
print("[2/4] Loading Bloomberg Indices...")
df_bloomberg = pd.read_excel(bloomberg_path)
df_bloomberg_clean = pd.DataFrame({
    'Bloomberg_Ticker': df_bloomberg['Ticker'],
    'Name': df_bloomberg['Index Name'],
    'Long_Description': df_bloomberg['LONG_COMP_NAME'],
    'category_tier1': df_bloomberg['category_tier1'],
    'category_tier2': df_bloomberg['category_tier2'],
    'category_tags': df_bloomberg['category_tags'],
    'source': 'Bloomberg'
})
dfs.append(df_bloomberg_clean)
print(f"   Loaded {len(df_bloomberg_clean)} rows from Bloomberg Indices")

# 3. Goldman Baskets
print("[3/4] Loading Goldman Baskets...")
df_goldman = pd.read_excel(goldman_path)
# For Goldman, use the Bloomberg column if available, otherwise use Index Name as ticker
bloomberg_col = df_goldman.get('Bloomberg', df_goldman['Index Name'])
df_goldman_clean = pd.DataFrame({
    'Bloomberg_Ticker': bloomberg_col,
    'Name': df_goldman['Index Name'],
    'Long_Description': df_goldman['description'].fillna(''),
    'category_tier1': df_goldman['category_tier1'],
    'category_tier2': df_goldman['category_tier2'],
    'category_tags': df_goldman['category_tags'],
    'source': 'Goldman'
})
dfs.append(df_goldman_clean)
print(f"   Loaded {len(df_goldman_clean)} rows from Goldman Baskets")

# 4. Thematic ETFs
print("[4/4] Loading Thematic ETFs...")
df_thematic = pd.read_excel(thematic_path)
# Clean up column names (remove newlines)
df_thematic.columns = df_thematic.columns.str.replace('\n', '', regex=False).str.strip()

df_thematic_clean = pd.DataFrame({
    'Bloomberg_Ticker': df_thematic['Ticker'],
    'Name': df_thematic['Name'],
    'Long_Description': df_thematic['CIE_DES'],
    'category_tier1': df_thematic['category_tier1'],
    'category_tier2': df_thematic['category_tier2'],
    'category_tags': df_thematic['category_tags'],
    'source': 'Thematic ETF'
})
dfs.append(df_thematic_clean)
print(f"   Loaded {len(df_thematic_clean)} rows from Thematic ETFs")

# Merge all
print("\n" + "="*80)
print("MERGING...")
master = pd.concat(dfs, ignore_index=True)

print(f"\nTotal rows before dedup check: {len(master)}")

# Check for exact duplicates (same ticker + source)
duplicates = master[master.duplicated(subset=['Bloomberg_Ticker', 'source'], keep=False)]
if len(duplicates) > 0:
    print(f"⚠️  Found {len(duplicates)} exact duplicates within same source")
    print("   Removing duplicates, keeping first occurrence...")
    master = master.drop_duplicates(subset=['Bloomberg_Ticker', 'source'], keep='first')

print(f"Total rows after dedup: {len(master)}")

# Distribution by source
print("\n" + "="*80)
print("MASTER LIST COMPOSITION:")
print("="*80)
print(f"\nSource Distribution:")
print(master['source'].value_counts().to_string())

print(f"\nTier-1 Distribution:")
print(master['category_tier1'].value_counts().to_string())

print(f"\nMissing Data Check:")
print(f"  Bloomberg_Ticker: {master['Bloomberg_Ticker'].isna().sum()} missing")
print(f"  Name: {master['Name'].isna().sum()} missing")
print(f"  Long_Description: {master['Long_Description'].isna().sum()} missing")
print(f"  category_tier1: {master['category_tier1'].isna().sum()} missing")
print(f"  category_tier2: {master['category_tier2'].isna().sum()} missing")
print(f"  category_tags: {master['category_tags'].isna().sum()} missing")

# Save master file
output_file = "Master Asset List Classified.xlsx"
master.to_excel(output_file, index=False)

print(f"\n" + "="*80)
print(f"✓ MASTER FILE SAVED")
print(f"="*80)
print(f"  File: {output_file}")
print(f"  Total assets: {len(master)}")
print(f"  Columns: {', '.join(master.columns.tolist())}")
print(f"  Size: ~{len(master) * 7 / 1000:.1f}MB")

# Sample rows
print(f"\nSample rows (first 3):")
for idx in range(min(3, len(master))):
    row = master.iloc[idx]
    print(f"\n{idx+1}. {row['Bloomberg_Ticker']} ({row['source']})")
    print(f"   Name: {row['Name'][:70]}")
    print(f"   Tier-1: {row['category_tier1']} | Tier-2: {row['category_tier2']}")
    print(f"   Tags: {str(row['category_tags'])[:80]}...")
