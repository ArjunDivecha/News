#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: yfinance_utils.py
=============================================================================

DESCRIPTION:
    Utility module for fetching market data from Yahoo Finance (primary) and
    Stooq (fallback). Provides functions to resolve ticker symbols (including
    exchange suffix fallbacks for international equities), retrieve stock/ETF
    metadata (name, sector, industry, currency, exchange, market cap, etc.),
    fetch historical OHLCV price data, and fetch latest prices with 1-day and
    YTD returns for multiple symbols. Includes a rate-limit probe that detects
    Yahoo HTTP 429 responses and switches to Stooq as a fallback data source.
    Designed to be imported as a library; the __main__ block runs a quick
    self-test on a predefined set of tickers.

INPUT FILES:
    (none -- all data is fetched from external APIs over HTTP)

OUTPUT FILES:
    (none -- functions return Python dicts/DataFrames to the caller)

VERSION: 1.0
LAST UPDATED: 2026-06-05
AUTHOR: Arjun Divecha

DEPENDENCIES:
    - yfinance
    - pandas
    - requests

USAGE:
    from yfinance_utils import resolve_ticker, get_stock_info, get_prices
    OR: python yfinance_utils.py  (runs self-test)

NOTES:
    - Requires internet access; Yahoo Finance API may rate-limit.
    - Stooq fallback is used when Yahoo returns HTTP 429 or no data.
    - All functions that access Yahoo expect the standard yfinance library.
=============================================================================
"""

import yfinance as yf
import pandas as pd
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import time
import io
import requests


def resolve_ticker(symbol: str) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Resolve a plain ticker to Yahoo Finance format and get basic info.
    
    Args:
        symbol: Plain ticker (e.g., 'AAPL', 'EWZ')
        
    Returns:
        Tuple of (yf_ticker, info_dict) or (None, None) if failed
    """
    # Try direct lookup first
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Check if we got valid data
        if info and 'symbol' in info and info.get('regularMarketPrice'):
            return symbol, info
            
        # If symbol was modified by yfinance
        if info and 'symbol' in info:
            return info['symbol'], info
            
    except Exception as e:
        pass
    
    # Try with common exchange suffixes for international stocks
    exchange_suffixes = ['.L', '.DE', '.PA', '.AS', '.MI', '.SW', 
                         '.T', '.HK', '.SS', '.SZ', '.TO', '.AX']
    
    for suffix in exchange_suffixes:
        try:
            test_symbol = symbol + suffix
            ticker = yf.Ticker(test_symbol)
            info = ticker.info
            
            if info and info.get('regularMarketPrice'):
                return test_symbol, info
        except Exception:
            continue
    
    return None, None


def get_stock_info(symbol: str) -> Optional[Dict]:
    """
    Get detailed stock/ETF info from Yahoo Finance.
    
    Returns:
        Dict with relevant fields, or None if failed
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        if not info:
            return None
        
        # Determine security type
        quote_type = info.get('quoteType', 'UNKNOWN')
        
        return {
            'symbol': info.get('symbol', symbol),
            'name': info.get('longName') or info.get('shortName', ''),
            'security_type': quote_type,
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'category': info.get('category'),  # For ETFs
            'country': info.get('country'),
            'currency': info.get('currency', 'USD'),
            'exchange': info.get('exchange'),
            'market_cap': info.get('marketCap'),
            'current_price': info.get('regularMarketPrice'),
            'previous_close': info.get('previousClose'),
            'fifty_day_avg': info.get('fiftyDayAverage'),
            'two_hundred_day_avg': info.get('twoHundredDayAverage'),
        }
        
    except Exception as e:
        print(f"Error fetching info for {symbol}: {e}")
        return None


def get_current_price(symbol: str) -> Optional[float]:
    """Get current price for a symbol."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return info.get('regularMarketPrice') or info.get('previousClose')
    except Exception:
        return None


def get_historical_prices(symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Get historical daily prices.
    
    Args:
        symbol: Yahoo Finance ticker
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        DataFrame with OHLCV data
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        if df.empty:
            return None
            
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        
        return df
        
    except Exception as e:
        print(f"Error fetching history for {symbol}: {e}")
        return None


def _yahoo_rate_limited() -> bool:
    """
    Quick probe to detect Yahoo throttling (HTTP 429).
    When this trips, avoid hammering Yahoo and fall back to Stooq.
    """
    url = "https://query1.finance.yahoo.com/v8/finance/chart/SPY?range=1d&interval=1d"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        response = requests.get(url, headers=headers, timeout=8)
        return response.status_code == 429
    except requests.RequestException:
        return True


def _fetch_from_yahoo(symbol: str, start_dt: datetime, end_dt: datetime) -> Dict:
    """Fetch price/returns from Yahoo Finance for one symbol."""
    ticker = yf.Ticker(symbol)
    df = ticker.history(
        start=start_dt.strftime('%Y-%m-%d'),
        end=(end_dt + timedelta(days=1)).strftime('%Y-%m-%d')
    )

    if df.empty:
        return {'status': 'failed', 'error': 'No data from Yahoo'}

    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    latest_row = df.iloc[-1]

    return_1d = None
    if len(df) >= 2:
        prev_close = df.iloc[-2]['Close']
        curr_close = latest_row['Close']
        if prev_close and prev_close > 0:
            return_1d = ((curr_close / prev_close) - 1) * 100

    return_ytd = None
    year_start = f"{end_dt.year}-01-01"
    try:
        ytd_df = ticker.history(
            start=year_start,
            end=(end_dt + timedelta(days=1)).strftime('%Y-%m-%d')
        )
        if len(ytd_df) >= 2:
            ytd_start_price = ytd_df.iloc[0]['Close']
            current_price = latest_row['Close']
            if ytd_start_price and ytd_start_price > 0:
                return_ytd = ((current_price / ytd_start_price) - 1) * 100
    except Exception:
        pass

    return {
        'status': 'success',
        'source': 'yahoo',
        'date': latest_row['Date'],
        'price': float(latest_row['Close']),
        'open': float(latest_row['Open']) if pd.notna(latest_row['Open']) else None,
        'high': float(latest_row['High']) if pd.notna(latest_row['High']) else None,
        'low': float(latest_row['Low']) if pd.notna(latest_row['Low']) else None,
        'volume': float(latest_row['Volume']) if pd.notna(latest_row['Volume']) else None,
        'return_1d': return_1d,
        'return_ytd': return_ytd,
    }


def _fetch_from_stooq(symbol: str, end_dt: datetime) -> Dict:
    """Fetch price/returns from Stooq for one symbol as fallback."""
    stooq_symbol = f"{symbol.lower()}.us"
    url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
    response = requests.get(url, timeout=15)
    response.raise_for_status()

    raw = response.text.strip()
    if not raw or raw == "No data":
        return {'status': 'failed', 'error': 'No data from Stooq'}

    df = pd.read_csv(io.StringIO(raw))
    if df.empty or 'Date' not in df.columns or 'Close' not in df.columns:
        return {'status': 'failed', 'error': 'Malformed Stooq response'}

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date', 'Close']).sort_values('Date')
    df = df[df['Date'] <= pd.Timestamp(end_dt)]

    if df.empty:
        return {'status': 'failed', 'error': 'No Stooq data on/before target date'}

    latest_idx = df.index[-1]
    latest_row = df.loc[latest_idx]
    latest_pos = df.index.get_loc(latest_idx)

    return_1d = None
    if latest_pos >= 1:
        prev_close = float(df.iloc[latest_pos - 1]['Close'])
        curr_close = float(latest_row['Close'])
        if prev_close > 0:
            return_1d = ((curr_close / prev_close) - 1) * 100

    return_ytd = None
    ytd_df = df[df['Date'].dt.year == end_dt.year]
    if len(ytd_df) >= 2:
        ytd_start_price = float(ytd_df.iloc[0]['Close'])
        current_price = float(latest_row['Close'])
        if ytd_start_price > 0:
            return_ytd = ((current_price / ytd_start_price) - 1) * 100

    return {
        'status': 'success',
        'source': 'stooq',
        'date': latest_row['Date'].strftime('%Y-%m-%d'),
        'price': float(latest_row['Close']),
        'open': float(latest_row['Open']) if 'Open' in latest_row and pd.notna(latest_row['Open']) else None,
        'high': float(latest_row['High']) if 'High' in latest_row and pd.notna(latest_row['High']) else None,
        'low': float(latest_row['Low']) if 'Low' in latest_row and pd.notna(latest_row['Low']) else None,
        'volume': float(latest_row['Volume']) if 'Volume' in latest_row and pd.notna(latest_row['Volume']) else None,
        'return_1d': return_1d,
        'return_ytd': return_ytd,
    }


def get_prices_for_date(symbols: List[str], date: str) -> Dict[str, Dict]:
    """
    Get prices for multiple symbols for a specific date.
    Uses the date as end_date and fetches 5 days of history to ensure we get data.
    
    Args:
        symbols: List of Yahoo Finance tickers
        date: Target date (YYYY-MM-DD)
        
    Returns:
        Dict mapping symbol to price data
    """
    results = {}
    
    # Calculate date range (look back 10 days to handle weekends/holidays)
    end_dt = datetime.strptime(date, '%Y-%m-%d')
    start_dt = end_dt - timedelta(days=10)
    
    use_yahoo = not _yahoo_rate_limited()

    for symbol in symbols:
        try:
            # Use Yahoo when available; fallback to Stooq for rate limits / symbol errors.
            if use_yahoo:
                yahoo_result = _fetch_from_yahoo(symbol, start_dt, end_dt)
                if yahoo_result.get('status') == 'success':
                    results[symbol] = yahoo_result
                    continue

            stooq_result = _fetch_from_stooq(symbol, end_dt)
            if stooq_result.get('status') == 'success':
                results[symbol] = stooq_result
            else:
                results[symbol] = {
                    'status': 'failed',
                    'error': stooq_result.get('error', 'No data from Yahoo/Stooq')
                }
        except Exception as e:
            results[symbol] = {'status': 'failed', 'error': str(e)}
    
    return results


def batch_get_info(symbols: List[str], delay: float = 0.1) -> Dict[str, Dict]:
    """
    Get info for multiple symbols with rate limiting.
    
    Args:
        symbols: List of symbols
        delay: Delay between requests (seconds)
        
    Returns:
        Dict mapping symbol to info
    """
    results = {}
    
    for i, symbol in enumerate(symbols):
        if i > 0:
            time.sleep(delay)
        
        yf_ticker, info = resolve_ticker(symbol)
        
        if yf_ticker and info:
            results[symbol] = {
                'yf_ticker': yf_ticker,
                'name': info.get('longName') or info.get('shortName', ''),
                'security_type': info.get('quoteType', 'UNKNOWN'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'category': info.get('category'),
                'country': info.get('country'),
                'currency': info.get('currency', 'USD'),
                'current_price': info.get('regularMarketPrice'),
                'status': 'resolved',
            }
        else:
            results[symbol] = {
                'yf_ticker': None,
                'status': 'failed',
            }
    
    return results


if __name__ == "__main__":
    # Test the utilities
    print("Testing Yahoo Finance utilities...")
    
    # Test ticker resolution
    test_symbols = ['AAPL', 'EWZ', 'INTC', 'ASHR', 'EPOL', 'VTV']
    
    for sym in test_symbols:
        yf_ticker, info = resolve_ticker(sym)
        if yf_ticker:
            sec_type = info.get('quoteType', 'UNK')
            sector = info.get('sector', 'N/A')
            country = info.get('country', 'N/A')
            print(f"  {sym}: ✓ {sec_type} | sector={sector} | country={country}")
        else:
            print(f"  {sym}: ✗ Failed to resolve")
    
    # Test price fetch
    print("\nTesting price fetch...")
    prices = get_prices_for_date(['AAPL', 'EWZ', 'VTV'], '2026-01-31')
    for sym, data in prices.items():
        if data['status'] == 'success':
            print(f"  {sym}: ${data['price']:.2f} | 1d={data.get('return_1d', 0):.2f}%")
        else:
            print(f"  {sym}: {data['status']} - {data.get('error', 'Unknown')}")
    
    print("\nYahoo Finance utilities test complete.")
