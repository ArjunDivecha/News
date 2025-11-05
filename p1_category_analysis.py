"""
P1 CATEGORY ANALYSIS & INSIGHTS
- Category performance (Tier-1, Tier-2, Source)
- Data quality validation
- Coverage gaps and opportunities
"""

import pandas as pd
import numpy as np

print("="*80)
print("P1 CATEGORY PERFORMANCE ANALYSIS")
print("="*80)

df = pd.read_excel("Final Master.xlsx")

# Convert numeric columns
numeric_cols = ['1 Year Sharpe', '3 year sharpe', '12 Month Return', '36 Month Return', '12 month volatility', 'Correlation with SPX']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print("\n[1/3] Tier-1 Analysis")
tier1_stats = df.groupby('category_tier1').agg({
    '1 Year Sharpe': ['count', 'mean', 'median', 'std'],
    '12 Month Return': ['mean', 'median'],
    '12 month volatility': ['mean', 'median'],
}).round(3)
print(tier1_stats)

print("\n[2/3] Tier-2 Analysis (Top 20)")
tier2_stats = df.groupby('category_tier2').agg({
    '1 Year Sharpe': ['count', 'mean', 'median'],
    '12 Month Return': 'mean',
}).round(3)
tier2_stats.columns = ['Count', 'Avg_Sharpe', 'Median_Sharpe', 'Avg_Return']
tier2_stats = tier2_stats.sort_values('Count', ascending=False).head(20)
print(tier2_stats)

print("\n[3/3] Source Analysis")
source_stats = df.groupby('source').agg({
    '1 Year Sharpe': ['count', 'mean', 'median'],
    '12 Month Return': 'mean',
    '12 month volatility': 'mean',
}).round(3)
print(source_stats)

# Save analysis
tier1_stats.to_excel("P1_Tier1_Analysis.xlsx")
tier2_stats.to_excel("P1_Tier2_Analysis.xlsx")
source_stats.to_excel("P1_Source_Analysis.xlsx")

print("\n✓ P1 Category analysis complete")
