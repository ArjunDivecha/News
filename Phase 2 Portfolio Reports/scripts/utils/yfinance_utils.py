#!/usr/bin/env python3
"""
=============================================================================
YAHOO FINANCE UTILITIES - Phase 2 Portfolio Reports
=============================================================================

Utilities for resolving tickers and fetching data from Yahoo Finance.

USAGE:
    from utils.yfinance_utils import resolve_ticker, get_stock_info, get_prices
=============================================================================
"""

import yfinance as yf
import pandas as pd
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import time


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
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_dt.strftime('%Y-%m-%d'), 
                               end=(end_dt + timedelta(days=1)).strftime('%Y-%m-%d'))
            
            if df.empty:
                results[symbol] = {'status': 'failed', 'error': 'No data'}
                continue
            
            # Get the latest available data point
            df.reset_index(inplace=True)
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
            
            # Find closest date to target
            latest_row = df.iloc[-1]
            actual_date = latest_row['Date']
            
            # Calculate daily return if we have enough data
            return_1d = None
            if len(df) >= 2:
                prev_close = df.iloc[-2]['Close']
                curr_close = latest_row['Close']
                if prev_close and prev_close > 0:
                    return_1d = ((curr_close / prev_close) - 1) * 100
            
            # Calculate YTD return
            return_ytd = None
            year_start = f"{end_dt.year}-01-01"
            try:
                ytd_df = ticker.history(start=year_start, 
                                       end=(end_dt + timedelta(days=1)).strftime('%Y-%m-%d'))
                if len(ytd_df) >= 2:
                    ytd_start_price = ytd_df.iloc[0]['Close']
                    current_price = latest_row['Close']
                    if ytd_start_price and ytd_start_price > 0:
                        return_ytd = ((current_price / ytd_start_price) - 1) * 100
            except Exception:
                pass
            
            results[symbol] = {
                'status': 'success',
                'date': actual_date,
                'price': latest_row['Close'],
                'open': latest_row['Open'],
                'high': latest_row['High'],
                'low': latest_row['Low'],
                'volume': latest_row['Volume'],
                'return_1d': return_1d,
                'return_ytd': return_ytd,
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
