"""
=============================================================================
MEME STOCK DISCOVERY TOOL - Search-Based Analysis with Volume & Price Metrics
=============================================================================

INPUT FILES:
- None required (data is fetched dynamically from web APIs)
- Requires: Valid Exa API key and internet connectivity
- Requires: Yahoo Finance data access for market metrics

OUTPUT FILES:
- /Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/News/search_based_meme_stocks.json
  Description: JSON file containing identified meme stocks with comprehensive metrics and analysis
  Format: JSON structure with timestamp, search queries, and detailed stock information
  Contents: 
    - timestamp: ISO format timestamp of analysis
    - search_queries: List of search terms used across platforms
    - total_found: Number of meme stocks identified
    - meme_stocks: Array of stock objects with:
      * symbol: Stock ticker (e.g., "GME")
      * name: Company name
      * mentions: Number of times mentioned across sources
      * volume_spike: Current volume relative to 60-day average
      * price_change_1d: 1-day price change percentage
      * price_change_5d: 5-day price change percentage
      * current_volume: Current trading volume
      * avg_volume: 60-day average volume
      * market_cap_b: Market capitalization in billions
      * reasons: List of criteria met for meme classification
      * sample_context: Text snippets showing usage context
      * meme_score: Composite score (mentions × volume_spike × price_volatility)

VERSION HISTORY:
v1.0.0 (2025-10-17): Initial release with multi-source search and volume/price analysis
v1.1.0 (2025-11-06): Enhanced documentation and comprehensive error handling

DEPENDENCIES:
- exa_py: Exa search API client
- yfinance: Yahoo Finance data access
- Python standard libraries: json, re, datetime, collections

SEARCH STRATEGIES:
1. WallStreetBets specific searches (apewisdom.io, quiverquant.com, swaggystocks.com)
2. General meme stock trending searches (unfiltered domains)
3. StockTwits trending analysis (stocktwits.com)
4. Short squeeze opportunity identification
5. Penny stock discussion tracking

MEME STOCK CRITERIA:
Stocks are classified as "meme stocks" if they meet ANY of:
- Volume spike > 1.3x recent average
- 1-day price change > ±8%
- 5-day price change > ±12%
PLUS minimum 2 mentions across search results

FILTERING:
Excludes common words mistaken for tickers: SPY, QQQ, NVDA, AAPL, etc.
Excludes generic terms: THE, AND, FOR, BUY, SELL, etc.
Requires minimum 3-character symbols and 2+ mentions

DATA QUALITY:
- Validates market data availability (minimum 10 trading days)
- Handles API failures gracefully with error logging
- Filters out invalid or delisted symbols
- Provides context snippets for verification

USAGE:
python MemeFinder.py

REQUIRES:
- Valid Exa API key (currently redacted in source)
- Internet connection for API calls
- Yahoo Finance data access

NOTES:
- Program makes ~30-40 API calls (Exa search + Yahoo Finance per ticker)
- Runtime: ~2-5 minutes depending on API response times
- Results are time-sensitive and should be used for informational purposes only
"""

# ============================================================================
# IMPORTS AND CONFIGURATION
# ============================================================================

from exa_py import Exa
import yfinance as yf
import json
import re
import os
from datetime import datetime, timedelta
from collections import defaultdict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

exa = Exa(api_key=os.getenv("EXA_API_KEY"))

print("🔍 Searching for meme stock discussions across multiple sources...")
print("=" * 70)

# ============================================================================
# SEARCH QUERIES CONFIGURATION
# ============================================================================
# Different search strategies targeting various financial discussion platforms
search_queries = [
    # WSB specific
    ("wallstreetbets trending stocks today", ["apewisdom.io", "quiverquant.com", "swaggystocks.com"]),
    ("WSB YOLO plays this week", ["apewisdom.io", "quiverquant.com"]),
    
    # General meme stock searches
    ("meme stocks trending today", []),  # No domain filter - get aggregators
    ("penny stocks reddit today", []),
    
    # StockTwits
    ("trending stocks stocktwits", ["stocktwits.com"]),
    
    # Short squeeze focus
    ("short squeeze stocks today", []),
]

# ============================================================================
# DATA STRUCTURES
# ============================================================================
# Dictionary to track ticker mentions across all search results
# Key: ticker symbol, Value: dict with count, sources, and contexts
all_tickers = defaultdict(lambda: {
    'count': 0,
    'sources': set(),
    'contexts': []
})

print("\n📊 Running searches...\n")

for query, domains in search_queries:
    print(f"🔍 Query: '{query[:50]}...'")
    try:
        search_params = {
            "query": query,
            "text": True,
            "num_results": 10,
            "type": "auto"
        }
        
        if domains:
            search_params["include_domains"] = domains
        
        results = exa.search_and_contents(**search_params)
        
        print(f"   Found {len(results.results)} results")
        
        # Extract tickers from results
        for result in results.results:
            if not hasattr(result, 'text') or not result.text:
                continue
            
            text = result.text
            url_domain = result.url.split('/')[2] if '://' in result.url else result.url
            
            # Extract tickers
            ticker_pattern = r'\$([A-Z]{2,5})\b'
            tickers = set(re.findall(ticker_pattern, text))
            
            # Also extract from common patterns
            word_pattern = r'\b([A-Z]{2,5})\b'
            potential = set(re.findall(word_pattern, text))
            
            # Add potential tickers mentioned in context with price/stock terms
            for ticker in potential:
                # Look for stock context words near the ticker
                ticker_idx = text.find(ticker)
                if ticker_idx != -1:
                    context = text[max(0, ticker_idx-50):min(len(text), ticker_idx+50)].lower()
                    if any(word in context for word in ['stock', 'share', 'price', 'ticker', '$', 'call', 'put']):
                        tickers.add(ticker)
            
            for ticker in tickers:
                all_tickers[ticker]['count'] += 1
                all_tickers[ticker]['sources'].add(url_domain)
                
                # Get context snippet
                idx = text.upper().find(ticker)
                if idx != -1:
                    snippet = text[max(0, idx-80):min(len(text), idx+80)]
                    all_tickers[ticker]['contexts'].append(snippet.replace('\n', ' ').strip())
        
        print(f"   Extracted tickers: {len(tickers)}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")

print(f"\n{'='*70}")
print("📊 Aggregating ticker mentions...")
print(f"{'='*70}\n")

# ============================================================================
# FILTERING AND RANKING
# ============================================================================
# Remove common words, major indices, and generic terms that could be mistaken for tickers
exclude = {
    'SPY', 'QQQ', 'VOO', 'IWM', 'DIA', 'VIX', 'TQQQ', 'SQQQ',
    'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'TSM',
    'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER',
    'WSB', 'CEO', 'USA', 'API', 'ETF', 'IPO', 'DD', 'FAQ', 'PDF', 'URL',
    'IT', 'GO', 'AM', 'SO', 'UP', 'AT', 'IN', 'ON', 'TO', 'OF', 'OR', 'BY',
    'GET', 'NEW', 'NOW', 'DAY', 'SEE', 'USE', 'TOP', 'HOT', 'BIG', 'LOW',
    'HIGH', 'LONG', 'PUT', 'CALL', 'BUY', 'SELL'
}

filtered = {
    ticker: data for ticker, data in all_tickers.items()
    if ticker not in exclude and len(ticker) >= 3 and data['count'] >= 2
}

# Sort by mentions
sorted_tickers = sorted(filtered.items(), key=lambda x: x[1]['count'], reverse=True)

print(f"Total tickers found: {len(all_tickers)}")
print(f"Filtered tickers (2+ mentions, valid symbols): {len(filtered)}")

print(f"\nTop mentioned tickers:")
for i, (ticker, data) in enumerate(sorted_tickers[:20], 1):
    print(f"{i:2d}. ${ticker}: {data['count']} mentions across {len(data['sources'])} sources")

print(f"\n{'='*70}")
print("🔍 Analyzing volume + price for top candidates...")
print(f"{'='*70}\n")

# ============================================================================
# VOLUME/PRICE ANALYSIS
# ============================================================================
# Analyze market data for top mentioned tickers to identify meme stock characteristics
meme_stocks = []

for ticker, data in sorted_tickers[:30]:
    try:
        print(f"Analyzing ${ticker}...", end=" ")
        
        stock = yf.Ticker(ticker)
        hist = stock.history(period="60d")
        
        if hist.empty or len(hist) < 10:
            print("❌ No market data")
            continue
        
        # Volume analysis
        current_volume = hist['Volume'].tail(3).mean()
        avg_volume = hist['Volume'].iloc[:-3].mean()
        volume_spike = (current_volume / avg_volume) if avg_volume > 0 else 0
        
        # Price changes
        current_price = hist['Close'].iloc[-1]
        
        if len(hist) >= 5:
            price_5d_ago = hist['Close'].iloc[-5]
            price_change_5d = ((current_price - price_5d_ago) / price_5d_ago * 100)
        else:
            price_change_5d = 0
        
        if len(hist) >= 2:
            price_1d_ago = hist['Close'].iloc[-2]
            price_change_1d = ((current_price - price_1d_ago) / price_1d_ago * 100)
        else:
            price_change_1d = 0
        
        # Get info
        try:
            info = stock.info
            market_cap = info.get('marketCap', 0)
            market_cap_b = market_cap / 1e9 if market_cap else 0
            name = info.get('longName', ticker)
        except:
            market_cap_b = 0
            name = ticker
        
        # Meme criteria (relaxed to get more results):
        # - Mentioned 2+ times
        # - Volume spike > 1.3x OR price movement > 8%
        
        is_meme = False
        reasons = []
        
        if volume_spike > 1.3:
            is_meme = True
            reasons.append(f"{volume_spike:.1f}x volume")
        
        if abs(price_change_1d) > 8:
            is_meme = True
            reasons.append(f"{price_change_1d:+.1f}% (1d)")
        
        if abs(price_change_5d) > 12:
            is_meme = True
            reasons.append(f"{price_change_5d:+.1f}% (5d)")
        
        if is_meme:
            meme_stocks.append({
                'symbol': ticker,
                'name': name,
                'mentions': data['count'],
                'sources': list(data['sources']),
                'volume_spike': round(volume_spike, 2),
                'price_change_1d': round(price_change_1d, 2),
                'price_change_5d': round(price_change_5d, 2),
                'current_volume': int(current_volume),
                'avg_volume': int(avg_volume),
                'market_cap_b': round(market_cap_b, 2),
                'reasons': reasons,
                'sample_context': data['contexts'][:2] if data['contexts'] else [],
                'meme_score': data['count'] * volume_spike * (1 + abs(price_change_1d)/100)
            })
            print(f"✅ {', '.join(reasons)}")
        else:
            print(f"Vol: {volume_spike:.1f}x, Price: {price_change_1d:+.1f}%")
        
    except Exception as e:
        print(f"❌ {e}")

print(f"\n{'='*70}")
print("🚀 MEME STOCKS FROM MULTI-SOURCE SEARCH")
print(f"{'='*70}\n")

# ============================================================================
# RESULTS OUTPUT
# ============================================================================
# Sort results by composite meme score and display findings
meme_stocks.sort(key=lambda x: x['meme_score'], reverse=True)

if meme_stocks:
    for i, stock in enumerate(meme_stocks, 1):
        print(f"{i}. ${stock['symbol']} - {stock['name']}")
        print(f"   Mentions: {stock['mentions']} across sources")
        print(f"   Volume: {stock['current_volume']:,} (↑{stock['volume_spike']}x)")
        print(f"   Price: {stock['price_change_1d']:+.2f}% (1d), {stock['price_change_5d']:+.2f}% (5d)")
        if stock['market_cap_b'] > 0:
            print(f"   Market Cap: ${stock['market_cap_b']:.2f}B")
        print(f"   🎯 {', '.join(stock['reasons'])}")
        print(f"   Meme Score: {stock['meme_score']:.1f}")
        if stock['sample_context']:
            print(f"   Context: ...{stock['sample_context'][0][:100]}...")
        print()
else:
    print("❌ No meme stocks found with current criteria")

# ============================================================================
# SAVE RESULTS
# ============================================================================
# Export findings to JSON file for further analysis or archiving
output = {
    "timestamp": datetime.now().isoformat(),
    "search_queries": [q for q, _ in search_queries],
    "total_found": len(meme_stocks),
    "meme_stocks": meme_stocks
}

with open('search_based_meme_stocks.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"💾 Results saved to: search_based_meme_stocks.json")
