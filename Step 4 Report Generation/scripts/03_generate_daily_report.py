#!/usr/bin/env python3
"""
=============================================================================
DAILY REPORT GENERATOR
=============================================================================

INPUT FILES:
- database/market_data.db (daily_prices, category_stats, factor_returns)
- prompts/daily_wrap.md (prompt template)

OUTPUT FILES:
- outputs/daily/daily_wrap_YYYY-MM-DD.md (Markdown report)
- outputs/daily/daily_wrap_YYYY-MM-DD.pdf (PDF report) [if PDF enabled]
- Updates reports table in database

VERSION: 1.0.0
CREATED: 2026-01-30

PURPOSE:
Generate daily market wrap report using multiple LLMs in parallel.
Pulls data from SQLite, injects into prompt template, generates reports.

USAGE:
    python scripts/03_generate_daily_report.py                    # Today's date
    python scripts/03_generate_daily_report.py --date 2026-01-30
    python scripts/03_generate_daily_report.py --provider anthropic # Single provider
    python scripts/03_generate_daily_report.py --test              # Test mode (no LLM)
    python scripts/03_generate_daily_report.py --pdf-engine kimi   # Premium PDF styling
=============================================================================
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sys

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))
from utils.db import (get_db, get_latest_prices, get_assets, 
                      get_tier1_distribution, get_tier2_distribution, save_report)

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
PROMPTS_DIR = PROJECT_DIR / "prompts"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "daily"
DB_PATH = PROJECT_DIR / "database" / "market_data.db"


def load_prompt_template() -> tuple:
    """Load the daily wrap prompt template and split into system/user parts."""
    prompt_path = PROMPTS_DIR / "daily_wrap.md"
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
    
    content = prompt_path.read_text()
    
    # Split on "USER" marker
    if "USER" in content:
        parts = content.split("USER", 1)
        system_prompt = parts[0].replace("SYSTEM", "").strip()
        user_prompt = parts[1].strip() if len(parts) > 1 else ""
    else:
        system_prompt = content
        user_prompt = ""
    
    return system_prompt, user_prompt


def get_tier1_stats(date: str) -> str:
    """Get Tier-1 category statistics for the date."""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT category_value, avg_return, median_return, std_return,
               min_return, max_return, count, percentile_60d, streak_days, streak_direction
        FROM category_stats
        WHERE date = ? AND category_type = 'tier1'
        ORDER BY avg_return DESC
    """, (date,))
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return "No Tier-1 data available"
    
    lines = ["| Category | Avg Return | Median | Std | Min | Max | Count | 60d Pctl | Streak |"]
    lines.append("|----------|------------|--------|-----|-----|-----|-------|----------|--------|")
    
    for row in rows:
        streak_str = f"{row[8]:+d}" if row[8] else "0"
        lines.append(
            f"| {row[0]} | {row[1]:+.2f}% | {row[2]:+.2f}% | {row[3]:.2f}% | "
            f"{row[4]:+.2f}% | {row[5]:+.2f}% | {row[6]} | {row[7]}% | {streak_str} |"
        )
    
    return "\n".join(lines)


def get_tier2_stats(date: str, limit: int = 20) -> str:
    """Get Tier-2 category statistics (top and bottom performers)."""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT category_value, avg_return, count, percentile_60d, streak_days
        FROM category_stats
        WHERE date = ? AND category_type = 'tier2'
        ORDER BY avg_return DESC
    """, (date,))
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return "No Tier-2 data available"
    
    # Top 10 and Bottom 10
    top_10 = rows[:10]
    bottom_10 = rows[-10:][::-1]
    
    lines = ["**TOP 10 BY RETURN:**"]
    lines.append("| Strategy | Avg Return | Count | 60d Pctl | Streak |")
    lines.append("|----------|------------|-------|----------|--------|")
    
    for row in top_10:
        streak_str = f"{row[4]:+d}" if row[4] else "0"
        lines.append(f"| {row[0]} | {row[1]:+.2f}% | {row[2]} | {row[3]}% | {streak_str} |")
    
    lines.append("\n**BOTTOM 10 BY RETURN:**")
    lines.append("| Strategy | Avg Return | Count | 60d Pctl | Streak |")
    lines.append("|----------|------------|-------|----------|--------|")
    
    for row in bottom_10:
        streak_str = f"{row[4]:+d}" if row[4] else "0"
        lines.append(f"| {row[0]} | {row[1]:+.2f}% | {row[2]} | {row[3]}% | {streak_str} |")
    
    return "\n".join(lines)


def get_factor_returns(date: str) -> str:
    """Get factor returns for beta attribution."""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT factor_name, return_1d
        FROM factor_returns
        WHERE date = ?
        ORDER BY return_1d DESC
    """, (date,))
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return "No factor return data available"
    
    lines = ["| Factor | 1-Day Return |"]
    lines.append("|--------|--------------|")
    
    for row in rows:
        lines.append(f"| {row[0]} | {row[1]:+.2f}% |")
    
    return "\n".join(lines)


def get_tier3_stats(date: str) -> Dict[str, str]:
    """Compute Tier-3 tag statistics from daily returns with granular breakdowns."""
    conn = get_db()
    
    # Get daily returns with tier3 tags
    df = pd.read_sql_query("""
        SELECT dp.ticker, dp.return_1d, a.tier3_tags, a.tier1, a.tier2
        FROM daily_prices dp
        JOIN assets a ON dp.ticker = a.ticker
        WHERE dp.date = ? AND dp.return_1d IS NOT NULL
    """, conn, params=[date])
    
    conn.close()
    
    if df.empty:
        return {
            'regional': "No Tier-3 data available",
            'us_sectors': "No US sector data available",
            'developed_countries': "No developed market data available",
            'emerging_countries': "No emerging market data available",
            'dm_vs_em': "No DM/EM comparison available",
            'sector': "No Tier-3 data available", 
            'style': "No Tier-3 data available",
            'strategy': "No Tier-3 data available",
        }
    
    # Parse tags and explode
    import json
    
    def parse_tags(tags_str):
        if pd.isna(tags_str) or not tags_str:
            return []
        try:
            return json.loads(tags_str)
        except:
            return [t.strip() for t in str(tags_str).split(',') if t.strip()]
    
    df['tags_list'] = df['tier3_tags'].apply(parse_tags)
    df['tags_set'] = df['tags_list'].apply(set)
    
    # Explode to one row per tag
    exploded = df.explode('tags_list')
    exploded = exploded[exploded['tags_list'].notna() & (exploded['tags_list'] != '')]
    
    # Define tag categories
    all_regional_tags = ['US', 'Global', 'Europe', 'Asia', 'EM', 'Japan', 'China', 'APAC', 
                         'Canada', 'UK', 'Latin America', 'Domestic', 'International',
                         'Australia', 'Korea', 'India', 'Brazil', 'Mexico', 'Taiwan']
    sector_tags = ['Tech', 'Energy', 'Financials', 'Healthcare', 'Consumer', 'Industrials',
                   'Materials', 'Utilities', 'Real Estate', 'Telecom', 'Commodity']
    style_tags = ['Value', 'Growth', 'Momentum', 'Quality', 'Low Volatility', 'Dividend',
                  'ESG', 'Blend', 'Large-cap', 'Mid-cap', 'Small-cap', 'Equal-Weight']
    strategy_tags = ['Quantitative', 'Factor-Based', 'Options-Based', 'Long/Short', 
                     'Thematic', 'Defensive', 'Active', 'Passive', 'Income', 'Multi-Asset']
    
    # Developed market countries
    dm_countries = ['US', 'Europe', 'Japan', 'Canada', 'UK', 'Australia', 'APAC']
    # Emerging market countries  
    em_countries = ['China', 'EM', 'Korea', 'India', 'Brazil', 'Mexico', 'Taiwan', 'Latin America', 'Asia']
    
    def compute_tag_stats(tags_filter, category_name, min_count=3):
        filtered = exploded[exploded['tags_list'].isin(tags_filter)]
        if filtered.empty:
            return f"No {category_name} tags with data"
        
        stats = filtered.groupby('tags_list')['return_1d'].agg(['mean', 'median', 'std', 'count'])
        stats = stats.sort_values('mean', ascending=False)
        
        lines = [f"| {category_name} | Avg | Median | Std | Count |"]
        lines.append("|----------|------|--------|-----|-------|")
        
        for tag, row in stats.iterrows():
            if row['count'] >= min_count:
                lines.append(f"| {tag} | {row['mean']:+.2f}% | {row['median']:+.2f}% | {row['std']:.2f}% | {int(row['count'])} |")
        
        return "\n".join(lines)
    
    # =========================================================
    # US SECTOR BREAKDOWN - Assets tagged with both 'US' and a sector
    # =========================================================
    us_assets = df[df['tags_set'].apply(lambda x: 'US' in x)]
    us_sector_stats = []
    
    for sector in sector_tags:
        sector_assets = us_assets[us_assets['tags_set'].apply(lambda x: sector in x)]
        if len(sector_assets) >= 3:
            avg_ret = sector_assets['return_1d'].mean()
            med_ret = sector_assets['return_1d'].median()
            std_ret = sector_assets['return_1d'].std()
            count = len(sector_assets)
            us_sector_stats.append((sector, avg_ret, med_ret, std_ret, count))
    
    us_sector_stats.sort(key=lambda x: x[1], reverse=True)
    
    if us_sector_stats:
        us_lines = ["| US Sector | Avg | Median | Std | Count |"]
        us_lines.append("|-----------|------|--------|-----|-------|")
        for sector, avg, med, std, cnt in us_sector_stats:
            us_lines.append(f"| {sector} | {avg:+.2f}% | {med:+.2f}% | {std:.2f}% | {cnt} |")
        us_sectors_str = "\n".join(us_lines)
    else:
        us_sectors_str = "Insufficient US sector data"
    
    # =========================================================
    # DEVELOPED MARKETS BY COUNTRY
    # =========================================================
    dm_stats = []
    for country in dm_countries:
        country_assets = df[df['tags_set'].apply(lambda x: country in x)]
        # Exclude if also tagged as EM
        country_assets = country_assets[~country_assets['tags_set'].apply(lambda x: 'EM' in x)]
        if len(country_assets) >= 3:
            avg_ret = country_assets['return_1d'].mean()
            med_ret = country_assets['return_1d'].median()
            std_ret = country_assets['return_1d'].std()
            count = len(country_assets)
            dm_stats.append((country, avg_ret, med_ret, std_ret, count))
    
    dm_stats.sort(key=lambda x: x[1], reverse=True)
    
    if dm_stats:
        dm_lines = ["| DM Country/Region | Avg | Median | Std | Count |"]
        dm_lines.append("|-------------------|------|--------|-----|-------|")
        for country, avg, med, std, cnt in dm_stats:
            dm_lines.append(f"| {country} | {avg:+.2f}% | {med:+.2f}% | {std:.2f}% | {cnt} |")
        # Add DM aggregate
        all_dm = df[df['tags_set'].apply(lambda x: any(c in x for c in dm_countries) and 'EM' not in x)]
        if len(all_dm) >= 3:
            dm_lines.append(f"| **DM Aggregate** | **{all_dm['return_1d'].mean():+.2f}%** | {all_dm['return_1d'].median():+.2f}% | {all_dm['return_1d'].std():.2f}% | {len(all_dm)} |")
        dm_countries_str = "\n".join(dm_lines)
    else:
        dm_countries_str = "Insufficient developed market data"
    
    # =========================================================
    # EMERGING MARKETS BY COUNTRY
    # =========================================================
    em_stats = []
    for country in em_countries:
        country_assets = df[df['tags_set'].apply(lambda x: country in x)]
        if len(country_assets) >= 3:
            avg_ret = country_assets['return_1d'].mean()
            med_ret = country_assets['return_1d'].median()
            std_ret = country_assets['return_1d'].std()
            count = len(country_assets)
            em_stats.append((country, avg_ret, med_ret, std_ret, count))
    
    em_stats.sort(key=lambda x: x[1], reverse=True)
    
    if em_stats:
        em_lines = ["| EM Country/Region | Avg | Median | Std | Count |"]
        em_lines.append("|-------------------|------|--------|-----|-------|")
        for country, avg, med, std, cnt in em_stats:
            em_lines.append(f"| {country} | {avg:+.2f}% | {med:+.2f}% | {std:.2f}% | {cnt} |")
        # Add EM aggregate
        all_em = df[df['tags_set'].apply(lambda x: 'EM' in x or 'China' in x or 'Latin America' in x)]
        if len(all_em) >= 3:
            em_lines.append(f"| **EM Aggregate** | **{all_em['return_1d'].mean():+.2f}%** | {all_em['return_1d'].median():+.2f}% | {all_em['return_1d'].std():.2f}% | {len(all_em)} |")
        em_countries_str = "\n".join(em_lines)
    else:
        em_countries_str = "Insufficient emerging market data"
    
    # =========================================================
    # DM vs EM COMPARISON
    # =========================================================
    all_dm = df[df['tags_set'].apply(lambda x: any(c in x for c in ['US', 'Europe', 'Japan', 'Canada', 'UK']) and 'EM' not in x and 'China' not in x)]
    all_em = df[df['tags_set'].apply(lambda x: 'EM' in x or 'China' in x or 'Latin America' in x)]
    
    dm_vs_em_lines = ["| Market | Avg | Median | Std | Count | Best | Worst |"]
    dm_vs_em_lines.append("|--------|------|--------|-----|-------|------|-------|")
    
    if len(all_dm) >= 3:
        dm_avg = all_dm['return_1d'].mean()
        dm_med = all_dm['return_1d'].median()
        dm_std = all_dm['return_1d'].std()
        dm_best = all_dm['return_1d'].max()
        dm_worst = all_dm['return_1d'].min()
        dm_vs_em_lines.append(f"| Developed | {dm_avg:+.2f}% | {dm_med:+.2f}% | {dm_std:.2f}% | {len(all_dm)} | {dm_best:+.2f}% | {dm_worst:+.2f}% |")
    
    if len(all_em) >= 3:
        em_avg = all_em['return_1d'].mean()
        em_med = all_em['return_1d'].median()
        em_std = all_em['return_1d'].std()
        em_best = all_em['return_1d'].max()
        em_worst = all_em['return_1d'].min()
        dm_vs_em_lines.append(f"| Emerging | {em_avg:+.2f}% | {em_med:+.2f}% | {em_std:.2f}% | {len(all_em)} | {em_best:+.2f}% | {em_worst:+.2f}% |")
    
    if len(all_dm) >= 3 and len(all_em) >= 3:
        spread = dm_avg - em_avg
        dm_vs_em_lines.append(f"| **DM-EM Spread** | **{spread:+.2f}%** | | | | | |")
    
    dm_vs_em_str = "\n".join(dm_vs_em_lines)
    
    # =========================================================
    # NEW TIER-3 CATEGORIES
    # =========================================================
    
    # DURATION BREAKDOWN (Fixed Income)
    duration_tags = ['Short (<2Y)', 'Medium (2-10Y)', 'Long (>10Y)']
    duration_stats = []
    for dur in duration_tags:
        dur_assets = df[df['tags_set'].apply(lambda x: dur in x)]
        if len(dur_assets) >= 3:
            avg_ret = dur_assets['return_1d'].mean()
            med_ret = dur_assets['return_1d'].median()
            std_ret = dur_assets['return_1d'].std()
            count = len(dur_assets)
            duration_stats.append((dur, avg_ret, med_ret, std_ret, count))
    
    duration_stats.sort(key=lambda x: x[1], reverse=True)
    
    if duration_stats:
        dur_lines = ["| Duration | Avg | Median | Std | Count |"]
        dur_lines.append("|----------|------|--------|-----|-------|")
        for dur, avg, med, std, cnt in duration_stats:
            dur_lines.append(f"| {dur} | {avg:+.2f}% | {med:+.2f}% | {std:.2f}% | {cnt} |")
        duration_str = "\n".join(dur_lines)
    else:
        duration_str = "Insufficient duration data"
    
    # CREDIT QUALITY BREAKDOWN
    credit_tags = ['Credit', 'Investment Grade', 'High Yield', 'Corporate', 'Sovereign']
    credit_stats = []
    for credit in credit_tags:
        credit_assets = df[df['tags_set'].apply(lambda x: credit in x)]
        if len(credit_assets) >= 3:
            avg_ret = credit_assets['return_1d'].mean()
            med_ret = credit_assets['return_1d'].median()
            std_ret = credit_assets['return_1d'].std()
            count = len(credit_assets)
            credit_stats.append((credit, avg_ret, med_ret, std_ret, count))
    
    credit_stats.sort(key=lambda x: x[1], reverse=True)
    
    if credit_stats:
        credit_lines = ["| Credit Type | Avg | Median | Std | Count |"]
        credit_lines.append("|-------------|------|--------|-----|-------|")
        for credit, avg, med, std, cnt in credit_stats:
            credit_lines.append(f"| {credit} | {avg:+.2f}% | {med:+.2f}% | {std:.2f}% | {cnt} |")
        credit_str = "\n".join(credit_lines)
    else:
        credit_str = "Insufficient credit data"
    
    # ACTIVE VS PASSIVE COMPARISON
    active_assets = df[df['tags_set'].apply(lambda x: 'Active' in x)]
    passive_assets = df[df['tags_set'].apply(lambda x: 'Passive' in x)]
    
    active_passive_lines = ["| Management Style | Avg | Median | Std | Count | Best | Worst |"]
    active_passive_lines.append("|------------------|------|--------|-----|-------|------|-------|")
    
    if len(active_assets) >= 3:
        act_avg = active_assets['return_1d'].mean()
        act_med = active_assets['return_1d'].median()
        act_std = active_assets['return_1d'].std()
        act_best = active_assets['return_1d'].max()
        act_worst = active_assets['return_1d'].min()
        active_passive_lines.append(f"| Active | {act_avg:+.2f}% | {act_med:+.2f}% | {act_std:.2f}% | {len(active_assets)} | {act_best:+.2f}% | {act_worst:+.2f}% |")
    
    if len(passive_assets) >= 3:
        pas_avg = passive_assets['return_1d'].mean()
        pas_med = passive_assets['return_1d'].median()
        pas_std = passive_assets['return_1d'].std()
        pas_best = passive_assets['return_1d'].max()
        pas_worst = passive_assets['return_1d'].min()
        active_passive_lines.append(f"| Passive | {pas_avg:+.2f}% | {pas_med:+.2f}% | {pas_std:.2f}% | {len(passive_assets)} | {pas_best:+.2f}% | {pas_worst:+.2f}% |")
    
    if len(active_assets) >= 3 and len(passive_assets) >= 3:
        spread = act_avg - pas_avg
        active_passive_lines.append(f"| **Active-Passive Spread** | **{spread:+.2f}%** | | | | | |")
    
    active_passive_str = "\n".join(active_passive_lines)
    
    # FX EXPOSURE
    fx_tags = ['FX', 'USD', 'EUR', 'JPY', 'GBP', 'CNH', 'CNY', 'Currency']
    fx_stats = []
    for fx in fx_tags:
        fx_assets = df[df['tags_set'].apply(lambda x: fx in x)]
        if len(fx_assets) >= 3:
            avg_ret = fx_assets['return_1d'].mean()
            med_ret = fx_assets['return_1d'].median()
            std_ret = fx_assets['return_1d'].std()
            count = len(fx_assets)
            fx_stats.append((fx, avg_ret, med_ret, std_ret, count))
    
    fx_stats.sort(key=lambda x: x[1], reverse=True)
    
    if fx_stats:
        fx_lines = ["| FX Exposure | Avg | Median | Std | Count |"]
        fx_lines.append("|-------------|------|--------|-----|-------|")
        for fx, avg, med, std, cnt in fx_stats:
            fx_lines.append(f"| {fx} | {avg:+.2f}% | {med:+.2f}% | {std:.2f}% | {cnt} |")
        fx_str = "\n".join(fx_lines)
    else:
        fx_str = "Insufficient FX data"
    
    # ALTERNATIVES BREAKDOWN
    alternative_tags = ['Alternative', 'Infrastructure', 'Volatility', 'Precious Metals', 'Gold', 'Silver']
    alt_stats = []
    for alt in alternative_tags:
        alt_assets = df[df['tags_set'].apply(lambda x: alt in x)]
        if len(alt_assets) >= 3:
            avg_ret = alt_assets['return_1d'].mean()
            med_ret = alt_assets['return_1d'].median()
            std_ret = alt_assets['return_1d'].std()
            count = len(alt_assets)
            alt_stats.append((alt, avg_ret, med_ret, std_ret, count))
    
    alt_stats.sort(key=lambda x: x[1], reverse=True)
    
    if alt_stats:
        alt_lines = ["| Alternative | Avg | Median | Std | Count |"]
        alt_lines.append("|-------------|------|--------|-----|-------|")
        for alt, avg, med, std, cnt in alt_stats:
            alt_lines.append(f"| {alt} | {avg:+.2f}% | {med:+.2f}% | {std:.2f}% | {cnt} |")
        alternatives_str = "\n".join(alt_lines)
    else:
        alternatives_str = "Insufficient alternatives data"
    
    return {
        'regional': compute_tag_stats(all_regional_tags, "Region"),
        'us_sectors': us_sectors_str,
        'developed_countries': dm_countries_str,
        'emerging_countries': em_countries_str,
        'dm_vs_em': dm_vs_em_str,
        'sector': compute_tag_stats(sector_tags, "Sector"),
        'style': compute_tag_stats(style_tags, "Style"),
        'strategy': compute_tag_stats(strategy_tags, "Strategy"),
        # New Tier-3 categories
        'duration_breakdown': duration_str,
        'credit_quality': credit_str,
        'active_vs_passive': active_passive_str,
        'fx_exposure': fx_str,
        'alternatives': alternatives_str,
    }


def get_goldman_thematic_stats(date: str) -> Dict[str, str]:
    """Compute Goldman thematic basket statistics with thematic groupings."""
    conn = get_db()
    
    # Get Goldman assets with returns
    df = pd.read_sql_query("""
        SELECT a.ticker, a.name, a.tier2, a.tier3_tags, dp.return_1d
        FROM assets a
        JOIN daily_prices dp ON a.ticker = dp.ticker
        WHERE a.source = 'Goldman' AND dp.date = ? AND dp.return_1d IS NOT NULL
    """, conn, params=[date])
    
    # Get market average for comparison
    market_avg = pd.read_sql_query("""
        SELECT AVG(return_1d) as mkt_avg FROM daily_prices WHERE date = ?
    """, conn, params=[date]).iloc[0]['mkt_avg']
    
    conn.close()
    
    if df.empty:
        return {
            'top_bottom': "No Goldman data available",
            'by_theme': "No Goldman data available",
            'hedges': "No hedge basket data available",
            'summary': "No Goldman data available",
        }
    
    import json
    
    def parse_tags(tags_str):
        if pd.isna(tags_str) or not tags_str:
            return set()
        try:
            return set(json.loads(tags_str))
        except:
            return set(t.strip() for t in str(tags_str).split(',') if t.strip())
    
    df['tags_set'] = df['tier3_tags'].apply(parse_tags)
    
    # =========================================================
    # TOP 5 / BOTTOM 5 GOLDMAN BASKETS
    # =========================================================
    sorted_df = df.sort_values('return_1d', ascending=False)
    top_5 = sorted_df.head(5)
    bottom_5 = sorted_df.tail(5).iloc[::-1]
    
    top_lines = ["**TOP 5 GOLDMAN BASKETS:**"]
    top_lines.append("| Basket | Return | Category |")
    top_lines.append("|--------|--------|----------|")
    for _, row in top_5.iterrows():
        name = row['name'][:40] if len(row['name']) > 40 else row['name']
        top_lines.append(f"| {name} | {row['return_1d']:+.2f}% | {row['tier2']} |")
    
    top_lines.append("\n**BOTTOM 5 GOLDMAN BASKETS:**")
    top_lines.append("| Basket | Return | Category |")
    top_lines.append("|--------|--------|----------|")
    for _, row in bottom_5.iterrows():
        name = row['name'][:40] if len(row['name']) > 40 else row['name']
        top_lines.append(f"| {name} | {row['return_1d']:+.2f}% | {row['tier2']} |")
    
    top_bottom_str = "\n".join(top_lines)
    
    # =========================================================
    # THEMATIC GROUPINGS
    # =========================================================
    # Define thematic clusters based on name patterns and tags
    theme_patterns = {
        'Hedge Baskets': lambda row: 'Hedge' in str(row['name']),
        'China/Asia Plays': lambda row: any(t in row['tags_set'] for t in ['China', 'Asia', 'Japan']) or 'China' in str(row['name']) or 'Asia' in str(row['name']),
        'Commodity Themes': lambda row: any(t in row['tags_set'] for t in ['Commodity', 'Energy']) or row['tier2'] in ['Metals', 'Energy', 'Agriculture'],
        'Factor Pairs': lambda row: 'Long/Short' in row['tags_set'] or 'Pair' in str(row['name']) or 'Short' in str(row['name']),
        'Credit/Rates': lambda row: 'Credit' in row['tags_set'] or row['tier2'] in ['Corporate Credit', 'Yield Curves', 'Credit Spreads', 'Sovereign Bonds'],
        'Momentum/Quant': lambda row: any(t in row['tags_set'] for t in ['Momentum', 'Quantitative', 'Factor-Based']),
        'Defensive/Low Vol': lambda row: any(t in row['tags_set'] for t in ['Defensive', 'Low Volatility']),
    }
    
    theme_stats = []
    for theme_name, filter_func in theme_patterns.items():
        theme_df = df[df.apply(filter_func, axis=1)]
        if len(theme_df) >= 3:
            avg = theme_df['return_1d'].mean()
            med = theme_df['return_1d'].median()
            best = theme_df.loc[theme_df['return_1d'].idxmax()]
            worst = theme_df.loc[theme_df['return_1d'].idxmin()]
            theme_stats.append({
                'theme': theme_name,
                'count': len(theme_df),
                'avg': avg,
                'median': med,
                'best_name': best['name'][:30],
                'best_ret': best['return_1d'],
                'worst_name': worst['name'][:30],
                'worst_ret': worst['return_1d'],
            })
    
    theme_stats.sort(key=lambda x: x['avg'], reverse=True)
    
    theme_lines = ["| Theme | Count | Avg | Best | Worst |"]
    theme_lines.append("|-------|-------|-----|------|-------|")
    for ts in theme_stats:
        theme_lines.append(
            f"| {ts['theme']} | {ts['count']} | {ts['avg']:+.2f}% | "
            f"{ts['best_name'][:20]} ({ts['best_ret']:+.1f}%) | "
            f"{ts['worst_name'][:20]} ({ts['worst_ret']:+.1f}%) |"
        )
    
    by_theme_str = "\n".join(theme_lines)
    
    # =========================================================
    # HEDGE BASKETS DETAIL
    # =========================================================
    hedge_df = df[df['name'].str.contains('Hedge', case=False, na=False)]
    if len(hedge_df) >= 3:
        hedge_sorted = hedge_df.sort_values('return_1d', ascending=False)
        hedge_lines = ["| Hedge Basket | Return | What It Hedges |"]
        hedge_lines.append("|--------------|--------|----------------|")
        for _, row in hedge_sorted.iterrows():
            name = row['name'][:35] if len(row['name']) > 35 else row['name']
            # Infer what it hedges from name
            hedge_target = "General"
            if 'China' in row['name']:
                hedge_target = "China exposure"
            elif 'Anglo' in row['name']:
                hedge_target = "Anglo/UK exposure"
            elif 'CNH' in row['name'] or 'CNY' in row['name']:
                hedge_target = "CNH/CNY FX"
            elif 'Japan' in row['name'] or 'JPY' in row['name']:
                hedge_target = "Japan/JPY"
            elif 'EU' in row['name'] or 'Europe' in row['name']:
                hedge_target = "Europe exposure"
            elif 'Brazil' in row['name'] or 'BZ' in row['name']:
                hedge_target = "Brazil exposure"
            elif 'Credit' in row['name']:
                hedge_target = "Credit risk"
            elif 'Vol' in row['name']:
                hedge_target = "Volatility"
            hedge_lines.append(f"| {name} | {row['return_1d']:+.2f}% | {hedge_target} |")
        hedge_str = "\n".join(hedge_lines)
    else:
        hedge_str = "Insufficient hedge basket data"
    
    # =========================================================
    # GOLDMAN SUMMARY
    # =========================================================
    gs_avg = df['return_1d'].mean()
    gs_med = df['return_1d'].median()
    gs_std = df['return_1d'].std()
    gs_vs_mkt = gs_avg - market_avg if market_avg else 0
    
    pct_positive = (df['return_1d'] > 0).mean() * 100
    pct_beat_mkt = (df['return_1d'] > market_avg).mean() * 100 if market_avg else 0
    
    summary_lines = [
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Goldman Basket Count | {len(df)} |",
        f"| Average Return | {gs_avg:+.2f}% |",
        f"| Median Return | {gs_med:+.2f}% |",
        f"| Std Dev | {gs_std:.2f}% |",
        f"| vs Market Avg | {gs_vs_mkt:+.2f}% |",
        f"| % Positive | {pct_positive:.0f}% |",
        f"| % Beat Market | {pct_beat_mkt:.0f}% |",
    ]
    
    summary_str = "\n".join(summary_lines)
    
    return {
        'top_bottom': top_bottom_str,
        'by_theme': by_theme_str,
        'hedges': hedge_str,
        'summary': summary_str,
    }


def get_unusual_flags(date: str) -> str:
    """Identify unusual patterns in the data."""
    conn = get_db()
    
    # Find categories with extreme returns (|z-score| > 1.5)
    df = pd.read_sql_query("""
        SELECT category_type, category_value, avg_return, std_return, percentile_60d, streak_days
        FROM category_stats
        WHERE date = ?
    """, conn, params=[date])
    
    conn.close()
    
    if df.empty:
        return "No unusual patterns detected (insufficient data)"
    
    flags = []
    
    # Check for extreme percentiles (only if percentile_60d is not None)
    df_with_percentile = df[df['percentile_60d'].notna()]
    if not df_with_percentile.empty:
        extremes = df_with_percentile[(df_with_percentile['percentile_60d'] <= 10) | (df_with_percentile['percentile_60d'] >= 90)]
        for _, row in extremes.iterrows():
            direction = "high" if row['percentile_60d'] >= 90 else "low"
            flags.append(f"- {row['category_value']} at {row['percentile_60d']}th percentile (historically {direction})")
    
    # Check for long streaks (only if streak_days is not None)
    df_with_streaks = df[df['streak_days'].notna()]
    if not df_with_streaks.empty:
        long_streaks = df_with_streaks[df_with_streaks['streak_days'].abs() >= 3]
        for _, row in long_streaks.iterrows():
            direction = "positive" if row['streak_days'] > 0 else "negative"
            flags.append(f"- {row['category_value']}: {abs(row['streak_days'])}-day {direction} streak")
    
    # Check for extreme returns
    if df['avg_return'].std() > 0:
        df['z_score'] = (df['avg_return'] - df['avg_return'].mean()) / df['avg_return'].std()
        outliers = df[df['z_score'].abs() > 1.5]
        for _, row in outliers.iterrows():
            direction = "outperforming" if row['z_score'] > 0 else "underperforming"
            flags.append(f"- {row['category_value']}: {direction} with z-score of {row['z_score']:.1f}")
    
    if not flags:
        return "No significant unusual patterns detected today"
    
    return "\n".join(flags[:10])  # Limit to top 10


def get_rsi_analysis(date: str) -> str:
    """Analyze RSI-14 momentum indicators to identify overbought/oversold conditions."""
    conn = get_db()
    
    df = pd.read_sql_query("""
        SELECT dp.ticker, a.tier1, a.tier2, dp.rsi_14, dp.return_1d
        FROM daily_prices dp
        JOIN assets a ON dp.ticker = a.ticker
        WHERE dp.date = ? AND dp.rsi_14 IS NOT NULL
    """, conn, params=[date])
    
    conn.close()
    
    if df.empty or df['rsi_14'].isna().all():
        return "No RSI data available"
    
    lines = ["**RSI-14 MOMENTUM ANALYSIS:**\n"]
    
    # Overbought (RSI > 70)
    overbought = df[df['rsi_14'] > 70].copy()
    if not overbought.empty:
        by_tier1 = overbought.groupby('tier1').agg({
            'ticker': 'count',
            'rsi_14': 'mean',
            'return_1d': 'mean'
        }).sort_values('ticker', ascending=False)
        
        lines.append("**OVERBOUGHT (RSI > 70):**")
        lines.append("| Tier-1 | Count | Avg RSI | Avg 1D Return |")
        lines.append("|--------|-------|---------|---------------|")
        for tier1, row in by_tier1.head(5).iterrows():
            lines.append(f"| {tier1} | {int(row['ticker'])} | {row['rsi_14']:.1f} | {row['return_1d']:+.2f}% |")
    
    # Oversold (RSI < 30)
    oversold = df[df['rsi_14'] < 30].copy()
    if not oversold.empty:
        by_tier1 = oversold.groupby('tier1').agg({
            'ticker': 'count',
            'rsi_14': 'mean',
            'return_1d': 'mean'
        }).sort_values('ticker', ascending=False)
        
        lines.append("\n**OVERSOLD (RSI < 30):**")
        lines.append("| Tier-1 | Count | Avg RSI | Avg 1D Return |")
        lines.append("|--------|-------|---------|---------------|")
        for tier1, row in by_tier1.head(5).iterrows():
            lines.append(f"| {tier1} | {int(row['ticker'])} | {row['rsi_14']:.1f} | {row['return_1d']:+.2f}% |")
    
    # RSI extremes summary
    total_overbought = len(overbought)
    total_oversold = len(oversold)
    total_assets = len(df)
    
    lines.append(f"\n**Summary:** {total_overbought} assets overbought ({total_overbought/total_assets*100:.1f}%), "
                 f"{total_oversold} assets oversold ({total_oversold/total_assets*100:.1f}%)")
    
    if total_overbought == 0 and total_oversold == 0:
        return "RSI analysis: No extreme momentum conditions detected (all assets within 30-70 range)"
    
    return "\n".join(lines)


def get_5d_momentum_analysis(date: str) -> str:
    """Analyze 5-day momentum patterns vs 1-day moves."""
    conn = get_db()
    
    df = pd.read_sql_query("""
        SELECT dp.ticker, a.tier1, a.tier2, dp.return_1d, dp.return_5d
        FROM daily_prices dp
        JOIN assets a ON dp.ticker = a.ticker
        WHERE dp.date = ? AND dp.return_5d IS NOT NULL
    """, conn, params=[date])
    
    conn.close()
    
    if df.empty or df['return_5d'].isna().all():
        return "No 5-day momentum data available"
    
    lines = ["**5-DAY MOMENTUM ANALYSIS:**\n"]
    
    # Tier-1 5D momentum summary
    tier1_stats = df.groupby('tier1').agg({
        'return_1d': 'mean',
        'return_5d': 'mean',
        'ticker': 'count'
    }).sort_values('return_5d', ascending=False)
    
    lines.append("| Tier-1 | 1-Day | 5-Day | Momentum Trend |")
    lines.append("|--------|-------|-------|----------------|")
    
    for tier1, row in tier1_stats.iterrows():
        # Determine trend direction
        if row['return_5d'] > 1.0 and row['return_1d'] > 0:
            trend = "Strong Up ↑↑"
        elif row['return_5d'] > 0 and row['return_1d'] > 0:
            trend = "Up ↑"
        elif row['return_5d'] < -1.0 and row['return_1d'] < 0:
            trend = "Strong Down ↓↓"
        elif row['return_5d'] < 0 and row['return_1d'] < 0:
            trend = "Down ↓"
        elif row['return_5d'] > 0 and row['return_1d'] < 0:
            trend = "Pullback (5D up, 1D down)"
        elif row['return_5d'] < 0 and row['return_1d'] > 0:
            trend = "Bounce (5D down, 1D up)"
        else:
            trend = "Flat"
        
        lines.append(f"| {tier1} | {row['return_1d']:+.2f}% | {row['return_5d']:+.2f}% | {trend} |")
    
    # Identify reversals (5D and 1D moves in opposite directions)
    df['reversal'] = (df['return_1d'] * df['return_5d']) < 0
    reversal_count = df['reversal'].sum()
    total = len(df)
    
    lines.append(f"\n**Reversals:** {reversal_count} assets ({reversal_count/total*100:.1f}%) showing 1D vs 5D divergence")
    
    return "\n".join(lines)


def get_volatility_regime(date: str) -> str:
    """Analyze volatility regime by comparing 30D vs 240D volatility."""
    conn = get_db()
    
    df = pd.read_sql_query("""
        SELECT dp.ticker, a.tier1, dp.volatility_30d, dp.volatility_240d
        FROM daily_prices dp
        JOIN assets a ON dp.ticker = a.ticker
        WHERE dp.date = ? AND dp.volatility_30d IS NOT NULL AND dp.volatility_240d IS NOT NULL
    """, conn, params=[date])
    
    conn.close()
    
    if df.empty:
        return "No volatility regime data available"
    
    # Calculate vol ratio (30D / 240D)
    df['vol_ratio'] = df['volatility_30d'] / df['volatility_240d'].replace(0, float('nan'))
    df = df.dropna(subset=['vol_ratio'])
    
    if df.empty:
        return "Insufficient volatility data for regime analysis"
    
    lines = ["**VOLATILITY REGIME ANALYSIS (30D vs 240D):**\n"]
    
    # Regime classification
    df['vol_regime'] = df['vol_ratio'].apply(
        lambda x: 'Vol Expansion' if x > 1.2 else ('Vol Compression' if x < 0.8 else 'Normal')
    )
    
    # By Tier-1
    tier1_regime = df.groupby('tier1').agg({
        'volatility_30d': 'mean',
        'volatility_240d': 'mean',
        'vol_ratio': 'mean'
    }).sort_values('vol_ratio', ascending=False)
    
    lines.append("| Tier-1 | 30D Vol | 240D Vol | Ratio | Regime |")
    lines.append("|--------|---------|----------|-------|--------|")
    
    for tier1, row in tier1_regime.iterrows():
        if row['vol_ratio'] > 1.2:
            regime = "**EXPANSION**"
        elif row['vol_ratio'] < 0.8:
            regime = "Compression"
        else:
            regime = "Normal"
        lines.append(f"| {tier1} | {row['volatility_30d']:.1f}% | {row['volatility_240d']:.1f}% | {row['vol_ratio']:.2f}x | {regime} |")
    
    # Summary
    expansion_pct = (df['vol_regime'] == 'Vol Expansion').mean() * 100
    compression_pct = (df['vol_regime'] == 'Vol Compression').mean() * 100
    
    if expansion_pct > 50:
        regime_status = "ELEVATED VOLATILITY REGIME - Short-term vol exceeds long-term across majority of assets"
    elif compression_pct > 50:
        regime_status = "LOW VOLATILITY REGIME - Vol compression detected, potential for mean reversion"
    else:
        regime_status = "NORMAL VOLATILITY REGIME - No significant vol regime shift detected"
    
    lines.append(f"\n**Regime Status:** {regime_status}")
    lines.append(f"Vol Expansion: {expansion_pct:.0f}% | Vol Compression: {compression_pct:.0f}%")
    
    return "\n".join(lines)


def get_historical_context(date: str) -> Dict[str, str]:
    """Generate comprehensive historical context including streaks, extremes, and regime."""
    conn = get_db()
    
    # Get category stats
    df = pd.read_sql_query("""
        SELECT category_type, category_value, avg_return, streak_days, percentile_60d
        FROM category_stats
        WHERE date = ?
    """, conn, params=[date])
    
    conn.close()
    
    # STREAKS
    streaks_lines = []
    if not df.empty and 'streak_days' in df.columns:
        df_with_streaks = df[df['streak_days'].notna()]
        if not df_with_streaks.empty:
            long_streaks = df_with_streaks[df_with_streaks['streak_days'].abs() >= 3]
            if not long_streaks.empty:
                streaks_lines.append("**ACTIVE STREAKS (3+ days):**")
                for _, row in long_streaks.sort_values('streak_days', key=abs, ascending=False).head(10).iterrows():
                    direction = "winning" if row['streak_days'] > 0 else "losing"
                    streaks_lines.append(f"- {row['category_value']}: {abs(int(row['streak_days']))}-day {direction} streak")
            else:
                streaks_lines.append("No categories with 3+ day streaks.")
        else:
            streaks_lines.append("Streak tracking not available (single-day data).")
    else:
        streaks_lines.append("Streak tracking not available (single-day data).")
    
    streaks_str = "\n".join(streaks_lines)
    
    # EXTREMES
    extremes_lines = []
    if not df.empty:
        # Top/Bottom performers today
        tier1_df = df[df['category_type'] == 'tier1'].sort_values('avg_return')
        if not tier1_df.empty:
            best = tier1_df.iloc[-1]
            worst = tier1_df.iloc[0]
            extremes_lines.append(f"**TODAY'S EXTREMES:**")
            extremes_lines.append(f"- Best Tier-1: {best['category_value']} ({best['avg_return']:+.2f}%)")
            extremes_lines.append(f"- Worst Tier-1: {worst['category_value']} ({worst['avg_return']:+.2f}%)")
        
        tier2_df = df[df['category_type'] == 'tier2'].sort_values('avg_return')
        if not tier2_df.empty:
            best = tier2_df.iloc[-1]
            worst = tier2_df.iloc[0]
            extremes_lines.append(f"- Best Tier-2: {best['category_value']} ({best['avg_return']:+.2f}%)")
            extremes_lines.append(f"- Worst Tier-2: {worst['category_value']} ({worst['avg_return']:+.2f}%)")
        
        # Historical percentile extremes
        df_with_pctl = df[df['percentile_60d'].notna()]
        if not df_with_pctl.empty:
            extreme_high = df_with_pctl[df_with_pctl['percentile_60d'] >= 95]
            extreme_low = df_with_pctl[df_with_pctl['percentile_60d'] <= 5]
            if not extreme_high.empty or not extreme_low.empty:
                extremes_lines.append("\n**60-DAY HISTORICAL EXTREMES:**")
                for _, row in extreme_high.iterrows():
                    extremes_lines.append(f"- {row['category_value']}: 95th+ percentile (extreme high)")
                for _, row in extreme_low.iterrows():
                    extremes_lines.append(f"- {row['category_value']}: 5th- percentile (extreme low)")
    
    if not extremes_lines:
        extremes_lines.append("Historical extremes analysis not available (single-day data).")
    
    extremes_str = "\n".join(extremes_lines)
    
    # SIMILAR DAYS
    similar_days_str = "Similar day pattern matching requires historical database (in development)."
    
    # REGIME INDICATORS
    regime_lines = ["**CURRENT REGIME INDICATORS:**"]
    if not df.empty:
        tier1_df = df[df['category_type'] == 'tier1']
        if not tier1_df.empty:
            avg_return = tier1_df['avg_return'].mean()
            if avg_return > 0.5:
                regime_lines.append("- Market Direction: RISK-ON (broad-based gains)")
            elif avg_return < -0.5:
                regime_lines.append("- Market Direction: RISK-OFF (broad-based losses)")
            else:
                regime_lines.append("- Market Direction: MIXED/NEUTRAL")
            
            # Dispersion
            dispersion = tier1_df['avg_return'].std()
            if dispersion > 2.0:
                regime_lines.append(f"- Cross-Asset Dispersion: HIGH ({dispersion:.2f}% std) - Rotation/Dislocation")
            elif dispersion > 1.0:
                regime_lines.append(f"- Cross-Asset Dispersion: ELEVATED ({dispersion:.2f}% std)")
            else:
                regime_lines.append(f"- Cross-Asset Dispersion: NORMAL ({dispersion:.2f}% std)")
    else:
        regime_lines.append("Regime analysis not available (insufficient data).")
    
    regime_str = "\n".join(regime_lines)
    
    return {
        'streaks': streaks_str,
        'extremes': extremes_str,
        'similar_days': similar_days_str,
        'regime_indicators': regime_str
    }


def get_correlation_analysis(date: str) -> Dict[str, str]:
    """Analyze rolling correlations and detect regime changes."""
    conn = get_db()
    
    # Get correlations for the date
    df = pd.read_sql_query("""
        SELECT ac.*, a.name, a.tier1, a.tier2
        FROM asset_correlations ac
        JOIN assets a ON ac.ticker = a.ticker
        WHERE ac.date = ?
    """, conn, params=[date])
    
    conn.close()
    
    if df.empty:
        return {
            'regime_changes': "No correlation data available",
            'factor_attribution': "No correlation data available",
            'correlation_summary': "No correlation data available",
        }
    
    # =========================================================
    # REGIME CHANGES
    # =========================================================
    regime_changes = df[df['regime_change'] == 1]
    
    if len(regime_changes) > 0:
        rc_lines = ["**CORRELATION REGIME CHANGES DETECTED:**\n"]
        rc_lines.append("| Ticker | Name | Best Factor | R² | Tier-1 |")
        rc_lines.append("|--------|------|-------------|-----|--------|")
        
        for _, row in regime_changes.head(15).iterrows():
            name = row['name'][:30] if len(row['name']) > 30 else row['name']
            r_sq = row['r_squared_best'] if row['r_squared_best'] else 0
            rc_lines.append(f"| {row['ticker']} | {name} | {row['best_factor']} | {r_sq:.2f} | {row['tier1']} |")
        
        rc_lines.append(f"\n**Total regime changes today: {len(regime_changes)}**")
        regime_changes_str = "\n".join(rc_lines)
    else:
        regime_changes_str = "No significant correlation regime changes detected today."
    
    # =========================================================
    # FACTOR ATTRIBUTION FOR TOP MOVERS
    # =========================================================
    # Join with daily returns to get top movers
    conn = get_db()
    movers = pd.read_sql_query("""
        SELECT ac.ticker, ac.best_factor, ac.r_squared_best,
               ac.corr_spx, ac.corr_russell2000, ac.corr_commodities,
               dp.return_1d, a.name, a.tier1
        FROM asset_correlations ac
        JOIN daily_prices dp ON ac.ticker = dp.ticker AND ac.date = dp.date
        JOIN assets a ON ac.ticker = a.ticker
        WHERE ac.date = ?
        ORDER BY ABS(dp.return_1d) DESC
        LIMIT 20
    """, conn, params=[date])
    conn.close()
    
    if not movers.empty:
        fa_lines = ["**TOP MOVERS - FACTOR ATTRIBUTION:**\n"]
        fa_lines.append("| Ticker | Return | Best Factor | R² | Likely Driver |")
        fa_lines.append("|--------|--------|-------------|-----|---------------|")
        
        for _, row in movers.head(10).iterrows():
            ret = row['return_1d'] if row['return_1d'] else 0
            r_sq = row['r_squared_best'] if row['r_squared_best'] else 0
            best_factor = row['best_factor'] if row['best_factor'] else 'Unknown'
            
            # Determine likely driver
            if r_sq >= 0.5:
                driver = f"Factor-driven ({best_factor})"
            elif r_sq >= 0.3:
                driver = f"Partial factor ({best_factor})"
            else:
                driver = "Idiosyncratic/Alpha"
            
            fa_lines.append(f"| {row['ticker']} | {ret:+.2f}% | {best_factor} | {r_sq:.2f} | {driver} |")
        
        factor_attribution_str = "\n".join(fa_lines)
    else:
        factor_attribution_str = "No top mover data available."
    
    # =========================================================
    # CORRELATION SUMMARY BY TIER-1
    # =========================================================
    tier1_corr = df.groupby('tier1').agg({
        'r_squared_best': 'mean',
        'best_factor': lambda x: x.mode().iloc[0] if len(x) > 0 else 'N/A',
        'ticker': 'count'
    }).sort_values('r_squared_best', ascending=False)
    
    summary_lines = ["**CORRELATION SUMMARY BY TIER-1:**\n"]
    summary_lines.append("| Tier-1 | Avg R² | Most Common Factor | Count |")
    summary_lines.append("|--------|--------|-------------------|-------|")
    
    for tier1, row in tier1_corr.iterrows():
        if tier1:
            r_sq = row['r_squared_best'] if pd.notna(row['r_squared_best']) else 0
            summary_lines.append(f"| {tier1} | {r_sq:.2f} | {row['best_factor']} | {int(row['ticker'])} |")
    
    # Overall stats
    overall_r_sq = df['r_squared_best'].mean()
    summary_lines.append(f"\n**Overall average R²: {overall_r_sq:.2f}** (higher = more factor-driven, lower = more idiosyncratic)")
    
    # Factor frequency
    factor_counts = df['best_factor'].value_counts().head(5)
    summary_lines.append("\n**Most common best factors:**")
    for factor, count in factor_counts.items():
        pct = count / len(df) * 100
        summary_lines.append(f"- {factor}: {count} assets ({pct:.1f}%)")
    
    correlation_summary_str = "\n".join(summary_lines)
    
    return {
        'regime_changes': regime_changes_str,
        'factor_attribution': factor_attribution_str,
        'correlation_summary': correlation_summary_str,
    }


def prepare_data_summary(date: str) -> Dict[str, str]:
    """Prepare all data summaries for prompt injection."""
    tier3 = get_tier3_stats(date)
    goldman = get_goldman_thematic_stats(date)
    historical = get_historical_context(date)
    correlations = get_correlation_analysis(date)
    
    return {
        'date': date,
        'tier1_stats': get_tier1_stats(date),
        'tier2_stats': get_tier2_stats(date),
        'tier3_regional_stats': tier3['regional'],
        'tier3_us_sectors': tier3['us_sectors'],
        'tier3_developed_countries': tier3['developed_countries'],
        'tier3_emerging_countries': tier3['emerging_countries'],
        'tier3_dm_vs_em': tier3['dm_vs_em'],
        'tier3_sector_stats': tier3['sector'],
        'tier3_style_stats': tier3['style'],
        'tier3_strategy_stats': tier3['strategy'],
        # New Tier-3 categories
        'duration_breakdown': tier3.get('duration_breakdown', 'No duration data'),
        'credit_quality': tier3.get('credit_quality', 'No credit data'),
        'active_vs_passive': tier3.get('active_vs_passive', 'No active/passive data'),
        'fx_exposure': tier3.get('fx_exposure', 'No FX data'),
        'alternatives': tier3.get('alternatives', 'No alternatives data'),
        # Goldman data
        'goldman_top_bottom': goldman['top_bottom'],
        'goldman_by_theme': goldman['by_theme'],
        'goldman_hedges': goldman['hedges'],
        'goldman_summary': goldman['summary'],
        'factor_returns': get_factor_returns(date),
        # New enhanced data fields
        'rsi_analysis': get_rsi_analysis(date),
        'momentum_5d_analysis': get_5d_momentum_analysis(date),
        'volatility_regime': get_volatility_regime(date),
        # Historical context (now computed)
        'streaks': historical['streaks'],
        'extremes': historical['extremes'],
        'similar_days': historical['similar_days'],
        'regime_indicators': historical['regime_indicators'],
        # Correlation analysis (new)
        'correlation_regime_changes': correlations['regime_changes'],
        'factor_attribution': correlations['factor_attribution'],
        'correlation_summary': correlations['correlation_summary'],
    }


def inject_data_into_prompt(user_prompt: str, data: Dict[str, str]) -> str:
    """Inject data summaries into the user prompt template."""
    result = user_prompt
    
    # Replace placeholders
    for key, value in data.items():
        placeholder = "{" + key + "}"
        result = result.replace(placeholder, str(value))
    
    # Add the unusual flags at the end
    result += f"\n\nUNUSUAL PATTERNS DETECTED:\n{data.get('unusual_flags', '')}"
    
    return result


def generate_daily_report(date: str, providers: List[str] = None,
                         test_mode: bool = False, verbose: bool = True,
                         pdf_engine: str = 'weasyprint') -> Dict:
    """
    Generate daily market wrap report.
    
    Args:
        date: Date string (YYYY-MM-DD)
        providers: List of LLM providers to use (default: all)
        test_mode: If True, skip LLM calls and return test output
        verbose: Print progress
        
    Returns:
        Dict with generation results
    """
    if providers is None:
        providers = ['anthropic']  # Default to Anthropic only for cost
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"GENERATING DAILY REPORT FOR {date}")
        print(f"{'='*70}")
        print(f"Providers: {', '.join(providers)}")
    
    # Load prompt template
    if verbose:
        print("\n[1/5] Loading prompt template...")
    system_prompt, user_prompt_template = load_prompt_template()
    if verbose:
        print(f"      System prompt: {len(system_prompt)} chars")
        print(f"      User prompt template: {len(user_prompt_template)} chars")
    
    # Prepare data
    if verbose:
        print("\n[2/5] Preparing data summary...")
    data = prepare_data_summary(date)
    data['unusual_flags'] = get_unusual_flags(date)
    if verbose:
        print(f"      Tier-1 stats: {len(data['tier1_stats'])} chars")
        print(f"      Tier-2 stats: {len(data['tier2_stats'])} chars")
        print(f"      Factor returns: {len(data['factor_returns'])} chars")
    
    # Inject data into prompt
    if verbose:
        print("\n[3/5] Injecting data into prompt...")
    user_prompt = inject_data_into_prompt(user_prompt_template, data)
    if verbose:
        print(f"      Final user prompt: {len(user_prompt)} chars")
    
    # Generate reports
    results = {'date': date, 'providers': {}, 'data_summary': data}
    
    if test_mode:
        if verbose:
            print("\n[4/5] TEST MODE - Skipping LLM generation...")
        results['providers']['test'] = {
            'content': f"# TEST REPORT FOR {date}\n\nThis is a test report.\n\n" + 
                      f"Data Summary:\n{data['tier1_stats']}\n",
            'model': 'test',
            'provider': 'test',
            'time_ms': 0,
        }
    else:
        if verbose:
            print(f"\n[4/5] Generating reports from {len(providers)} provider(s)...")
        
        # Import LLM utilities
        from utils.llm import generate_parallel, generate_report as gen_single
        
        if len(providers) == 1:
            # Single provider
            provider = providers[0]
            if verbose:
                print(f"      Calling {provider}...")
            result = gen_single(system_prompt, user_prompt, provider, 'daily', 12000)
            results['providers'][provider] = result
            if verbose:
                if 'error' in result:
                    print(f"      ERROR: {result['error']}")
                else:
                    print(f"      Success: {len(result.get('content', ''))} chars, {result.get('time_ms', 0)}ms")
        else:
            # Multiple providers in parallel
            if verbose:
                print(f"      Calling {len(providers)} providers in parallel...")
            parallel_results = generate_parallel(system_prompt, user_prompt, providers, 'daily', 12000)
            results['providers'] = parallel_results
            for provider, result in parallel_results.items():
                if verbose:
                    if 'error' in result:
                        print(f"      [{provider}] ERROR: {result['error']}")
                    else:
                        print(f"      [{provider}] Success: {len(result.get('content', ''))} chars")
    
    # Save outputs
    if verbose:
        print("\n[5/5] Saving outputs...")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Import PDF converter based on engine choice
    pdf_available = False
    pdf_converter = None
    
    if pdf_engine == 'kimi':
        try:
            from utils.pdf_kimi import convert_report as kimi_convert
            pdf_converter = kimi_convert
            pdf_available = True
            if verbose:
                print(f"      Using Kimi PDF engine (Playwright + Jinja2)")
        except ImportError as e:
            if verbose:
                print(f"      WARNING: Kimi PDF engine not available: {e}")
                print("               Install with: pip install playwright jinja2 && playwright install chromium")
    
    if not pdf_available and pdf_engine in ['weasyprint', 'auto']:
        try:
            from utils.pdf_weasyprint import convert_report as weasy_convert
            pdf_converter = weasy_convert
            pdf_available = True
            if verbose:
                print(f"      Using WeasyPrint PDF engine")
        except ImportError:
            pass
    
    if not pdf_available:
        try:
            from utils.pdf import convert_report as reportlab_convert
            pdf_converter = reportlab_convert
            pdf_available = True
            if verbose:
                print(f"      Using ReportLab PDF engine")
        except ImportError:
            if verbose:
                print("      WARNING: No PDF conversion available")
                print("               Install with: pip install weasyprint markdown")
    
    saved_files = []
    for provider, result in results['providers'].items():
        if 'content' in result and result['content']:
            # Save markdown
            md_filename = f"daily_wrap_{date}_{provider}.md"
            md_path = OUTPUT_DIR / md_filename
            md_path.write_text(result['content'])
            saved_files.append(md_path)
            
            # Convert to PDF
            if pdf_available and pdf_converter:
                pdf_path = pdf_converter(str(md_path))
                if pdf_path:
                    saved_files.append(Path(pdf_path))
                    if verbose:
                        print(f"      Converted to PDF: {Path(pdf_path).name}")
            
            # Save to database
            report_id = f"daily_{date}_{provider}"
            save_report(
                report_id=report_id,
                report_type='daily',
                report_date=date,
                content_md=result['content'],
                model_name=result.get('model', 'unknown'),
                tokens_input=result.get('tokens_input'),
                tokens_output=result.get('tokens_output'),
                generation_time_ms=result.get('time_ms'),
            )
            
            if verbose:
                print(f"      Saved: {md_filename}")
    
    results['saved_files'] = [str(f) for f in saved_files]
    
    if verbose:
        print(f"\n{'='*70}")
        print("DAILY REPORT GENERATION COMPLETE")
        print(f"{'='*70}")
        print(f"\nReports saved to: {OUTPUT_DIR}")
        for f in saved_files:
            print(f"  - {f.name}")
    
    return results


def test_report_generation(date: str) -> bool:
    """Test the report generation pipeline."""
    print("\n" + "=" * 70)
    print("TESTING REPORT GENERATION")
    print("=" * 70)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Prompt template loads
    print("\n[TEST 1] Load prompt template...")
    try:
        system, user = load_prompt_template()
        if len(system) > 100 and len(user) > 100:
            print(f"         PASSED (system={len(system)} chars, user={len(user)} chars)")
            tests_passed += 1
        else:
            print("         FAILED (prompt too short)")
            tests_failed += 1
    except Exception as e:
        print(f"         FAILED: {e}")
        tests_failed += 1
    
    # Test 2: Data preparation
    print("\n[TEST 2] Prepare data summary...")
    try:
        data = prepare_data_summary(date)
        if data['tier1_stats'] != "No Tier-1 data available":
            print(f"         PASSED (tier1={len(data['tier1_stats'])} chars)")
            tests_passed += 1
        else:
            print("         FAILED (no tier-1 data)")
            tests_failed += 1
    except Exception as e:
        print(f"         FAILED: {e}")
        tests_failed += 1
    
    # Test 3: Prompt injection
    print("\n[TEST 3] Inject data into prompt...")
    try:
        _, user_template = load_prompt_template()
        data = prepare_data_summary(date)
        injected = inject_data_into_prompt(user_template, data)
        if "{date}" not in injected and date in injected:
            print(f"         PASSED (injected={len(injected)} chars)")
            tests_passed += 1
        else:
            print("         FAILED (placeholders not replaced)")
            tests_failed += 1
    except Exception as e:
        print(f"         FAILED: {e}")
        tests_failed += 1
    
    # Test 4: Test mode generation
    print("\n[TEST 4] Test mode generation...")
    try:
        result = generate_daily_report(date, test_mode=True, verbose=False)
        if result['providers'].get('test', {}).get('content'):
            print(f"         PASSED (test report generated)")
            tests_passed += 1
        else:
            print("         FAILED (no test report)")
            tests_failed += 1
    except Exception as e:
        print(f"         FAILED: {e}")
        tests_failed += 1
    
    # Test 5: Output file created
    print("\n[TEST 5] Output file created...")
    try:
        result = generate_daily_report(date, test_mode=True, verbose=False)
        if result['saved_files']:
            test_file = Path(result['saved_files'][0])
            if test_file.exists():
                print(f"         PASSED ({test_file.name})")
                tests_passed += 1
            else:
                print("         FAILED (file not found)")
                tests_failed += 1
        else:
            print("         FAILED (no files saved)")
            tests_failed += 1
    except Exception as e:
        print(f"         FAILED: {e}")
        tests_failed += 1
    
    print("\n" + "-" * 70)
    print(f"TESTS COMPLETED: {tests_passed} passed, {tests_failed} failed")
    print("-" * 70)
    
    return tests_failed == 0


def get_last_trading_day(target_date: str = None) -> str:
    """
    Get the last trading day with data in the database.
    
    If target_date is a weekend or has no data, walks back to find
    the most recent trading day with actual data.
    
    Args:
        target_date: Date string YYYY-MM-DD (defaults to today)
        
    Returns:
        Date string of last trading day with data
    """
    if target_date is None:
        target_date = datetime.now().strftime('%Y-%m-%d')
    
    # Parse the date
    dt = datetime.strptime(target_date, '%Y-%m-%d')
    
    # Check if weekend - walk back to Friday
    if dt.weekday() == 5:  # Saturday
        dt = dt - timedelta(days=1)
        print(f"  Saturday detected, using Friday: {dt.strftime('%Y-%m-%d')}")
    elif dt.weekday() == 6:  # Sunday
        dt = dt - timedelta(days=2)
        print(f"  Sunday detected, using Friday: {dt.strftime('%Y-%m-%d')}")
    
    # Now check if we have data for this date
    conn = get_db()
    cursor = conn.cursor()
    
    # Look for data, walking back up to 10 days if needed
    for i in range(10):
        check_date = (dt - timedelta(days=i)).strftime('%Y-%m-%d')
        cursor.execute(
            "SELECT COUNT(*) FROM daily_prices WHERE date = ?",
            (check_date,)
        )
        count = cursor.fetchone()[0]
        
        if count > 0:
            if i > 0:
                print(f"  No data for {target_date}, using last available: {check_date} ({count} assets)")
            conn.close()
            return check_date
    
    conn.close()
    
    # If no data found, just return the original (adjusted for weekend)
    return dt.strftime('%Y-%m-%d')


def main():
    parser = argparse.ArgumentParser(description="Generate daily market wrap report")
    parser.add_argument("--date", type=str, default=None,
                       help="Date to generate report for (YYYY-MM-DD). Defaults to last trading day.")
    parser.add_argument("--provider", type=str, nargs='+',
                       choices=['openai', 'anthropic', 'google'],
                       default=['anthropic'],
                       help="LLM provider(s) to use")
    parser.add_argument("--test", action="store_true",
                       help="Test mode (skip LLM calls)")
    parser.add_argument("--pdf-engine", type=str,
                       choices=['weasyprint', 'kimi', 'reportlab', 'auto'],
                       default='weasyprint',
                       help="PDF engine: weasyprint (default), kimi (premium), reportlab")
    parser.add_argument("--run-tests", action="store_true",
                       help="Run pipeline tests")
    args = parser.parse_args()
    
    # Determine the correct date (last trading day)
    if args.date:
        report_date = get_last_trading_day(args.date)
    else:
        report_date = get_last_trading_day(datetime.now().strftime('%Y-%m-%d'))
    
    if args.run_tests:
        success = test_report_generation(report_date)
        return 0 if success else 1
    
    try:
        result = generate_daily_report(
            date=report_date,
            providers=args.provider,
            test_mode=args.test,
            verbose=True,
            pdf_engine=args.pdf_engine,
        )
        
        if any('error' in r for r in result['providers'].values()):
            print("\n⚠ Some providers failed")
            return 1
        
        print("\n✓ Daily report generation complete")
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
