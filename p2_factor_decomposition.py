"""
P2 FACTOR DECOMPOSITION & ENRICHED PROFILES
- Extract factor exposures from beta vectors
- Create enriched asset profiles for RAG
- Generate feature matrix for ML
"""

import pandas as pd
import numpy as np

print("="*80)
print("P2 FACTOR DECOMPOSITION & ENRICHED PROFILES")
print("="*80)

df = pd.read_excel("Final Master.xlsx")

# Convert numeric columns
numeric_cols = ['Daily 1 Year Beta to SPX', 'Russell 2000 Index', 'MSCI EAFE Index',
                'MSCI Emerging Markets Index', '1 Year Sharpe', '3 year sharpe',
                '12 Month Return', '36 Month Return', '12 month volatility', 'Correlation with SPX']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print("\n[1/2] Extracting factor exposures...")

# Factor decomposition
df['market_exposure'] = df['Daily 1 Year Beta to SPX'].fillna(0)
df['size_tilt'] = (df['Russell 2000 Index'].fillna(0) - df['Daily 1 Year Beta to SPX'].fillna(0))
df['geographic_tilt'] = (df['MSCI EAFE Index'].fillna(0) - df['Daily 1 Year Beta to SPX'].fillna(0))
df['em_tilt'] = df['MSCI Emerging Markets Index'].fillna(0)

# Quality metrics
df['sharpe_consistency'] = np.where(
    (df['1 Year Sharpe'].notna()) & (df['3 year sharpe'].notna()),
    1 - np.abs(df['1 Year Sharpe'] - df['3 year sharpe']),
    0
)

df['risk_adjusted_return'] = df['12 Month Return'] / (df['12 month volatility'] + 0.001)

print("  Factors extracted:")
print(f"    - Market exposure: mean={df['market_exposure'].mean():.3f}")
print(f"    - Size tilt: mean={df['size_tilt'].mean():.3f}")
print(f"    - Geographic tilt: mean={df['geographic_tilt'].mean():.3f}")
print(f"    - EM tilt: mean={df['em_tilt'].mean():.3f}")

print("\n[2/2] Creating enriched asset profiles...")

# Build enriched profiles
profiles = pd.DataFrame({
    'Bloomberg_Ticker': df['Bloomberg_Ticker'],
    'Name': df['Name'],
    'Source': df['source'],
    'Tier1': df['category_tier1'],
    'Tier2': df['category_tier2'],
    'Tags': df['category_tags'],

    # Performance metrics
    'Sharpe_1Y': df['1 Year Sharpe'],
    'Sharpe_3Y': df['3 year sharpe'],
    'Return_12M': df['12 Month Return'],
    'Return_36M': df['36 Month Return'],
    'Vol_12M': df['12 month volatility'],
    'Correlation_SPX': df['Correlation with SPX'],

    # Factor exposures
    'Market_Exposure': df['market_exposure'],
    'Size_Tilt': df['size_tilt'],
    'Geographic_Tilt': df['geographic_tilt'],
    'EM_Tilt': df['em_tilt'],

    # Quality metrics
    'Sharpe_Consistency': df['sharpe_consistency'],
    'Risk_Adjusted_Return': df['risk_adjusted_return'],

    # Ranking within category
    'Sharpe_Percentile_Tier2': df.groupby('category_tier2')['1 Year Sharpe'].rank(pct=True),
})

# Save enriched profiles
profiles.to_excel("P2_Enriched_Asset_Profiles.xlsx", index=False)

print(f"  Created enriched profiles for {len(profiles)} assets")
print(f"  Saved to: P2_Enriched_Asset_Profiles.xlsx")

print("\n✓ P2 Factor decomposition complete")
