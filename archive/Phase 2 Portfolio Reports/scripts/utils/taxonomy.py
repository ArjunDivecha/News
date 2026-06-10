#!/usr/bin/env python3
"""
=============================================================================
TAXONOMY MAPPING MODULE
=============================================================================

Defines the Phase 1 taxonomy and mapping from yfinance sectors/countries
to the unified taxonomy system.

This ensures classification consistency between Phase 1 and Phase 2.
=============================================================================
"""

# =============================================================================
# TIER-1 CATEGORIES (7 categories)
# =============================================================================

TIER1_CATEGORIES = [
    "Equities",
    "Fixed Income",
    "Commodities",
    "Currencies (FX)",
    "Multi-Asset / Thematic",
    "Volatility / Risk Premia",
    "Alternative / Synthetic",
]

# =============================================================================
# TIER-2 CATEGORIES (by Tier-1)
# =============================================================================

TIER2_BY_TIER1 = {
    "Equities": [
        "Global Indices",
        "Sector Indices",
        "Country/Regional",
        "Thematic/Factor",
        "Real Estate / REITs",
    ],
    "Fixed Income": [
        "Sovereign Bonds",
        "Corporate Credit",
        "Credit Spreads",
        "Yield Curves",
    ],
    "Commodities": [
        "Energy",
        "Metals",
        "Agriculture",
    ],
    "Currencies (FX)": [
        "Majors",
        "EM FX",
    ],
    "Multi-Asset / Thematic": [
        "Cross-Asset Indices",
        "Inflation/Growth Themes",
    ],
    "Volatility / Risk Premia": [
        "Vol Indices",
        "Carry/Value Factors",
    ],
    "Alternative / Synthetic": [
        "Quant/Style Baskets",
        "Custom/Proprietary",
    ],
}

# =============================================================================
# TIER-3 TAGS (multi-select)
# =============================================================================

TIER3_REGIONS = [
    "US", "Europe", "Asia", "EM", "Global", 
    "China", "Japan", "India", "Canada", "APAC", "Australia",
    "Brazil", "Korea", "Taiwan", "UK", "Germany", "France",
]

TIER3_SECTORS = [
    "Tech", "Energy", "Financials", "Healthcare", "Industrials", 
    "Consumer", "Defensive", "ESG", "Dividend", "Growth", "Value", 
    "Momentum", "Quality", "Infrastructure", "Real Estate", "Utilities",
    "Materials", "Communications",
]

TIER3_STRATEGIES = [
    "Active", "Passive", "Thematic", "Quantitative", "Factor-Based",
    "Low Volatility", "Defensive", "Long/Short", "Options-Based",
]

# =============================================================================
# YFINANCE SECTOR → TAXONOMY MAPPING
# =============================================================================

YFINANCE_SECTOR_MAP = {
    # yfinance sector → (tier1, tier2, tier3_tags)
    'Technology': ('Equities', 'Sector Indices', ['Tech', 'Equity']),
    'Financial Services': ('Equities', 'Sector Indices', ['Financials', 'Equity']),
    'Healthcare': ('Equities', 'Sector Indices', ['Healthcare', 'Equity']),
    'Consumer Cyclical': ('Equities', 'Sector Indices', ['Consumer', 'Equity']),
    'Consumer Defensive': ('Equities', 'Sector Indices', ['Consumer', 'Defensive', 'Equity']),
    'Industrials': ('Equities', 'Sector Indices', ['Industrials', 'Equity']),
    'Energy': ('Equities', 'Sector Indices', ['Energy', 'Equity']),
    'Basic Materials': ('Equities', 'Sector Indices', ['Materials', 'Equity']),
    'Communication Services': ('Equities', 'Sector Indices', ['Communications', 'Equity']),
    'Utilities': ('Equities', 'Sector Indices', ['Utilities', 'Defensive', 'Equity']),
    'Real Estate': ('Equities', 'Real Estate / REITs', ['Real Estate', 'Equity']),
}

# Default for unknown sectors
DEFAULT_SECTOR_MAPPING = ('Equities', 'Country/Regional', ['Equity'])

# =============================================================================
# YFINANCE COUNTRY → TIER-3 REGION TAG MAPPING
# =============================================================================

YFINANCE_COUNTRY_MAP = {
    # yfinance country → tier3 region tag
    'United States': 'US',
    'China': 'China',
    'Hong Kong': 'China',
    'Japan': 'Japan',
    'United Kingdom': 'Europe',
    'Germany': 'Europe',
    'France': 'Europe',
    'Italy': 'Europe',
    'Spain': 'Europe',
    'Netherlands': 'Europe',
    'Switzerland': 'Europe',
    'Sweden': 'Europe',
    'Norway': 'Europe',
    'Denmark': 'Europe',
    'Finland': 'Europe',
    'Belgium': 'Europe',
    'Austria': 'Europe',
    'Portugal': 'Europe',
    'Ireland': 'Europe',
    'India': 'India',
    'Canada': 'Canada',
    'Australia': 'Australia',
    'Brazil': 'EM',
    'Mexico': 'EM',
    'South Korea': 'Asia',
    'Taiwan': 'Asia',
    'Singapore': 'Asia',
    'Malaysia': 'Asia',
    'Thailand': 'Asia',
    'Indonesia': 'EM',
    'Philippines': 'EM',
    'Vietnam': 'EM',
    'South Africa': 'EM',
    'Turkey': 'EM',
    'Poland': 'EM',
    'Russia': 'EM',
    'Saudi Arabia': 'EM',
    'United Arab Emirates': 'EM',
    'Israel': 'EM',
    'Chile': 'EM',
    'Colombia': 'EM',
    'Peru': 'EM',
    'Argentina': 'EM',
    'Egypt': 'EM',
    'Pakistan': 'EM',
    'Bangladesh': 'EM',
    'New Zealand': 'APAC',
}

# Default region
DEFAULT_REGION = 'Global'


def map_stock_to_taxonomy(yf_info: dict) -> dict:
    """
    Map a stock's yfinance info to Phase 1 taxonomy.
    
    Args:
        yf_info: Dict from yfinance Ticker.info
        
    Returns:
        Dict with tier1, tier2, tier3_tags
    """
    sector = yf_info.get('sector', '')
    country = yf_info.get('country', '')
    
    # Get sector mapping
    if sector in YFINANCE_SECTOR_MAP:
        tier1, tier2, tier3_tags = YFINANCE_SECTOR_MAP[sector]
    else:
        tier1, tier2, tier3_tags = DEFAULT_SECTOR_MAPPING
    
    # Add region tag from country
    tier3_tags = list(tier3_tags)  # Copy to avoid mutation
    region = YFINANCE_COUNTRY_MAP.get(country, DEFAULT_REGION)
    if region not in tier3_tags:
        tier3_tags.append(region)
    
    return {
        'tier1': tier1,
        'tier2': tier2,
        'tier3_tags': tier3_tags,
    }


def get_region_from_country(country: str) -> str:
    """Get region tag from country name."""
    return YFINANCE_COUNTRY_MAP.get(country, DEFAULT_REGION)


# =============================================================================
# HAIKU CLASSIFICATION PROMPT (same as Phase 1)
# =============================================================================

HAIKU_TAXONOMY = """
TIER-1 CATEGORIES (top-level asset class):
1. Equities - Stock indices, ETFs, equity-focused baskets, REITs
2. Fixed Income - Bonds, credit, yield-focused instruments
3. Commodities - Energy, metals, agriculture
4. Currencies (FX) - Currency pairs and FX instruments
5. Multi-Asset / Thematic - Cross-asset, thematic baskets, macro themes
6. Volatility / Risk Premia - VIX, volatility indices, carry strategies
7. Alternative / Synthetic - Quantitative baskets, factor portfolios, proprietary constructs

TIER-2 CATEGORIES (by sub-class/structure):
Equities: Global Indices | Sector Indices | Country/Regional | Thematic/Factor | Real Estate / REITs
Fixed Income: Sovereign Bonds | Corporate Credit | Credit Spreads | Yield Curves
Commodities: Energy | Metals | Agriculture
Currencies: Majors (EUR/USD, GBP/USD, JPY/USD) | EM FX
Multi-Asset: Cross-Asset Indices | Inflation/Growth Themes
Volatility: Vol Indices | Carry/Value Factors
Alternative: Quant/Style Baskets | Custom/Proprietary

TIER-3 TAGS (all applicable - multi-select):
Asset Class: Equity | Credit | FX | Commodity | Multi-Asset
Region: US | Europe | Asia | EM | Global | China | Japan | India | Canada | UAE | APAC | Australia
Sector/Theme: Tech | Energy | Financials | Healthcare | Industrials | Consumer | Defensive | ESG | Dividend | Growth | Value | Momentum | Quality | Infrastructure | Real Estate | Utilities
Strategy: Active | Passive | Equal-Weight | Thematic | Quantitative | Options-Based | Dividend | Factor-Based | Low Volatility | Defensive | Domestic
Special Themes: Stimulus | Going Global | Long/Short
Duration (bonds only): Short (<2Y) | Medium (2-10Y) | Long (>10Y)
"""

HAIKU_SYSTEM_PROMPT = f"""You are an expert asset classification specialist. Your job is to classify financial assets (ETFs, indices, baskets) into a unified taxonomy.

{HAIKU_TAXONOMY}

For each asset, return a JSON object with exactly these fields:
{{
  "identifier": "Ticker or name",
  "tier1": "One of: Equities, Fixed Income, Commodities, Currencies (FX), Multi-Asset / Thematic, Volatility / Risk Premia, Alternative / Synthetic",
  "tier2": "Sub-category appropriate for the tier1 (see examples above)",
  "tier3_tags": ["tag1", "tag2", "tag3"] - list of ALL applicable tags from the taxonomy
}}

Be precise and use ONLY the categories and tags from the taxonomy above. If unsure, choose the closest match.
For assets with no description, use the name/identifier and any available metadata to make best guess."""


if __name__ == "__main__":
    # Test mapping
    test_info = {
        'sector': 'Technology',
        'country': 'United States',
    }
    result = map_stock_to_taxonomy(test_info)
    print(f"Test mapping: {result}")
    
    test_info2 = {
        'sector': 'Financial Services',
        'country': 'Brazil',
    }
    result2 = map_stock_to_taxonomy(test_info2)
    print(f"Test mapping 2: {result2}")
