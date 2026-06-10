#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: 04_generate_report.py
=============================================================================

DESCRIPTION:
    Generates a personalized portfolio daily wrap report for a given
    portfolio ID and target date. The script loads a markdown prompt
    template from the project prompts directory, reads portfolio data
    (daily snapshot, aggregates, summary) from a SQLite database via
    utility functions (utils.db), and enriches it with raw broker position
    data from Schwab and IBKR Excel files for account-level detail. The
    prepared data is injected into the prompt template and sent to the
    Claude LLM (via utils.llm) to generate narrative markdown. If the LLM
    output is truncated before all required sections are present,
    targeted continuation requests are made per missing section (up to 2
    attempts each). The final complete markdown is saved to disk, then
    converted to PDF (PrinceXML preferred; WeasyPrint fallback). A report
    record (content, model metadata, token counts, generation time) is
    saved to the database via save_report().

INPUT FILES:
    /Users/arjundivecha/Dropbox/AAA Backup/Master Valuation/Schwab All Accounts Data.xlsx
        Schwab brokerage positions (columns: Account, Name, Value, Bloom,
        Shares, AvgCost, USD P&L). Used to construct account-level P&L
        tables and cash-inclusive portfolio totals.
    /Users/arjundivecha/Dropbox/AAA Backup/Master Valuation/IBKR Account Data.xlsx
        IBKR brokerage positions (columns: Account, Name, Value, Shares,
        AvgCost, Unrealized P&L, Currency). Used similarly to construct
        account-level P&L and portfolio totals.
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/News/Phase 2 Portfolio Reports/prompts/portfolio_daily_wrap.md
        Markdown prompt template with {placeholder} variables that are
        replaced with portfolio summary, contributors, breakdowns, and
        market context before LLM generation.
    Database via utils.db (get_db, get_portfolio, get_holdings,
        get_daily_snapshot, get_aggregates, get_portfolio_summary,
        get_phase1_market_data, save_report) — SQLite database for
        portfolio config, holdings, daily snapshots, and market context.

OUTPUT FILES:
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/News/Phase 2 Portfolio Reports/outputs/{portfolio_id}/portfolio_wrap_{date}.md
        Full markdown report with LLM-generated narrative and data tables.
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/News/Phase 2 Portfolio Reports/outputs/{portfolio_id}/portfolio_wrap_{date}.pdf
        PDF version of the report, generated via PrinceXML (preferred) or
        WeasyPrint (fallback).
    Database via utils.db.save_report() — Report content, model name,
        PDF path, token counts, and generation time stored in the
        portfolio_reports table.

VERSION: 1.0
LAST UPDATED: 2026-06-05
AUTHOR: Arjun Divecha

DEPENDENCIES:
    - pandas
    - argparse, json, subprocess, os, re, uuid (stdlib)
    - pathlib (stdlib)
    - utils.db (local: get_db, get_portfolio, get_holdings,
        get_daily_snapshot, get_aggregates, get_portfolio_summary,
        get_phase1_market_data, save_report)
    - utils.llm (local: generate_report, get_report_model)
    - utils.pdf_prince.convert (local: convert_to_pdf, PRINCE_AVAILABLE)
        — optional: requires PrinceXML system installation
    - markdown (optional: required for WeasyPrint fallback)
    - weasyprint (optional: required for WeasyPrint fallback)

USAGE:
    python 04_generate_report.py --portfolio <PORTFOLIO_ID> --date <YYYY-MM-DD>
    python 04_generate_report.py --portfolio TEST --date 2026-01-31 --quiet

NOTES:
    - The Bloomberg Terminal is not used by this script; it reads cached
      broker Excel files (Schwab, IBKR) instead.
    - Requires a configured SQLite database at the path managed by utils.db.
    - PrinceXML must be installed on the system for professional PDF
      output; WeasyPrint is tried as an automatic fallback otherwise.
    - The script validates that all 9 required report sections are present
      in LLM output, making up to 2 continuation attempts per missing
      section before raising an error.
=============================================================================
"""

import argparse
import sys
import json
import subprocess
import os
import re
from pathlib import Path
from datetime import datetime
import uuid

import pandas as pd

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.db import (
    get_db, get_portfolio, get_holdings, get_daily_snapshot,
    get_aggregates, get_portfolio_summary, get_phase1_market_data,
    save_report
)
from utils.llm import generate_report, get_report_model

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
PROMPTS_DIR = PROJECT_DIR / "prompts"
OUTPUT_DIR = PROJECT_DIR / "outputs"
REPORT_MAX_TOKENS = int(os.getenv('PHASE2_REPORT_MAX_TOKENS', '2600'))
CONTINUATION_MAX_TOKENS = int(os.getenv('PHASE2_CONTINUATION_MAX_TOKENS', '2600'))
SCHWAB_RAW_PATH = Path("/Users/arjundivecha/Dropbox/AAA Backup/Master Valuation/Schwab All Accounts Data.xlsx")
IBKR_RAW_PATH = Path("/Users/arjundivecha/Dropbox/AAA Backup/Master Valuation/IBKR Account Data.xlsx")
PORTFOLIO_TOTALS_SECTION_HEADER = "### 1A. CASH-INCLUSIVE PORTFOLIO TOTALS"
ACCOUNT_PNL_SECTION_HEADER = "### 6A. ACCOUNT P&L DETAIL"

# Require complete section coverage so the report never silently truncates.
REQUIRED_SECTIONS = {
    0: "### 0. EXECUTIVE SYNTHESIS ⭐ START HERE",
    1: "### 1. PORTFOLIO AT A GLANCE",
    2: "### 2. TOP CONTRIBUTORS & DETRACTORS",
    3: "### 3. REGIONAL EXPOSURE ANALYSIS",
    4: "### 4. SECTOR/THEME EXPOSURE",
    5: "### 5. LONG VS SHORT ANALYSIS",
    6: "### 6. P&L ANALYSIS",
    7: "### 7. CONCENTRATION, RISK & SCENARIO ANALYSIS",
    8: "### 8. MARKET CONTEXT FOR PORTFOLIO",
}


def _fmt_currency_md(value, decimals: int = 0) -> str:
    """Format currency values for markdown tables."""
    if value is None or pd.isna(value):
        return "n/a"
    if decimals > 0:
        return f"${float(value):,.{decimals}f}"
    return f"${float(value):,.0f}"


def _fmt_pct_md(value, decimals: int = 2) -> str:
    """Format percentage values for markdown tables."""
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):+.{decimals}f}%"


def _fmt_number_md(value, decimals: int = 2) -> str:
    """Format numeric values for markdown tables."""
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):,.{decimals}f}"


def load_prompt_template() -> tuple:
    """Load the portfolio daily wrap prompt template."""
    prompt_path = PROMPTS_DIR / "portfolio_daily_wrap.md"
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
    
    content = prompt_path.read_text()
    
    # Split on USER to get system and user parts
    if "USER" in content:
        parts = content.split("USER", 1)
        system_prompt = parts[0].replace("SYSTEM", "").strip()
        user_prompt = parts[1].strip()
    else:
        system_prompt = ""
        user_prompt = content
    
    return system_prompt, user_prompt


def format_portfolio_summary(summary: dict, portfolio_totals: dict | None = None) -> str:
    """Format portfolio summary as text."""
    if not summary:
        return "No portfolio summary available."
    
    lines = []
    if portfolio_totals:
        lines.append(
            f"Total Portfolio Value (Cash Incl.): "
            f"${portfolio_totals.get('total_portfolio_value', 0):,.2f}"
        )
        lines.append(
            f"Cash & Equivalents: ${portfolio_totals.get('cash_value', 0):,.2f}"
        )
        daily_pnl = portfolio_totals.get("total_daily_pnl")
        if daily_pnl is not None:
            lines.append(f"Daily P&L (Cash Incl.): ${daily_pnl:,.2f}")
        nav_return = portfolio_totals.get("nav_return_1d")
        if nav_return is not None:
            lines.append(f"NAV Return (1D, Cash Incl.): {nav_return:+.2f}%")
        lines.append("")
    lines.append(f"Total Market Value: ${(summary.get('total_market_value') or 0):,.2f}")
    lines.append(f"Long Exposure: ${(summary.get('total_long_value') or 0):,.2f}")
    lines.append(f"Short Exposure: ${(summary.get('total_short_value') or 0):,.2f}")
    lines.append(f"Net Exposure: ${(summary.get('net_exposure') or 0):,.2f}")
    lines.append(f"Gross Exposure: ${(summary.get('gross_exposure') or 0):,.2f}")
    lines.append(f"")
    lines.append(f"Holdings: {summary.get('holding_count') or 0} total")
    lines.append(f"  - Long positions: {summary.get('long_count') or 0}")
    lines.append(f"  - Short positions: {summary.get('short_count') or 0}")
    lines.append(f"")
    lines.append(f"Portfolio Return (1D): {(summary.get('portfolio_return_1d') or 0):+.2f}%")
    ytd = summary.get('portfolio_return_ytd')
    if ytd is not None:
        lines.append(f"Portfolio Return (YTD): {ytd:+.2f}%")
    lines.append(f"Total Unrealized P&L: ${(summary.get('total_open_pnl') or 0):,.2f}")
    
    return "\n".join(lines)


def format_contributors(contributors: list, label: str) -> str:
    """Format top contributors/detractors as table."""
    if not contributors:
        return f"No {label.lower()} data available."
    
    lines = []
    lines.append(f"| Symbol | Type | Weight | Return | Contribution |")
    lines.append(f"|--------|------|--------|--------|--------------|")
    
    for c in contributors:
        symbol = c.get('symbol', 'N/A')
        pos_type = c.get('position_type', 'LONG')
        weight = c.get('weight', 0) * 100
        ret = c.get('return_1d', 0)
        contrib = c.get('contribution', 0)
        lines.append(f"| {symbol} | {pos_type} | {weight:.1f}% | {ret:+.2f}% | {contrib:+.1f}bp |")
    
    return "\n".join(lines)


def format_aggregates(aggregates: pd.DataFrame, dim_type: str, snapshot: pd.DataFrame = None) -> str:
    """Format aggregates for a specific dimension as table with YTD."""
    dim_data = aggregates[aggregates['dimension_type'] == dim_type].copy()
    
    if dim_data.empty:
        return f"No {dim_type} breakdown available."
    
    # Compute YTD averages from snapshot if available
    ytd_map = {}
    if snapshot is not None and not snapshot.empty:
        # Need to get the dimension field from holdings
        # For tier1/tier2, we need to join with assets table
        # For now, compute weighted YTD from snapshot grouped by the dimension
        conn = get_db()
        
        # Get holdings with their classifications
        portfolio_id = snapshot['portfolio_id'].iloc[0]
        holdings_df = pd.read_sql_query("""
            SELECT h.symbol, h.tier1, h.tier2, h.yf_sector, h.country
            FROM portfolio_holdings h
            WHERE h.portfolio_id = ?
        """, conn, params=[portfolio_id])
        conn.close()
        
        # Merge snapshot with holdings to get classifications
        merged = snapshot.merge(holdings_df, on='symbol', how='left')
        
        # Map dimension type to column name
        dim_col_map = {
            'tier1': 'tier1',
            'tier2': 'tier2',
            'sector': 'yf_sector',
            'region': 'country',
        }
        
        dim_col = dim_col_map.get(dim_type)
        if dim_col and dim_col in merged.columns:
            # Compute weighted YTD for each dimension value
            for dim_val in merged[dim_col].dropna().unique():
                subset = merged[merged[dim_col] == dim_val]
                if not subset.empty and subset['weight'].sum() > 0:
                    # Weighted average YTD
                    weighted_ytd = (subset['return_ytd'] * subset['weight']).sum() / subset['weight'].sum()
                    ytd_map[dim_val] = weighted_ytd
    
    # Sort by total weight descending
    dim_data = dim_data.sort_values('total_weight', ascending=False)
    
    lines = []
    lines.append(f"| {dim_type.title()} | Weight | 1D Ret | YTD Ret | Contribution | Holdings |")
    lines.append(f"|{'-'*20}|--------|--------|---------|--------------|----------|")
    
    for _, row in dim_data.iterrows():
        value = row['dimension_value']
        weight = row['total_weight'] * 100
        ret = row['weighted_return_1d']
        ytd = ytd_map.get(value, 0)
        contrib = row['contribution_1d']
        count = row['holding_count']
        
        if weight > 0.5:  # Only show if meaningful weight
            lines.append(f"| {value:<18} | {weight:>5.1f}% | {ret:>+6.2f}% | {ytd:>+6.2f}% | {contrib:>+9.1f}bp | {count:>8} |")
    
    return "\n".join(lines)


def format_holdings_detail(snapshot: pd.DataFrame) -> str:
    """Format holdings detail as table."""
    if snapshot.empty:
        return "No holdings data available."
    
    # Sort by absolute contribution
    snapshot = snapshot.copy()
    # Coerce nullable/object numeric fields defensively so report generation
    # still works when price fetches fail and values are NULL.
    snapshot['contribution_1d'] = pd.to_numeric(snapshot['contribution_1d'], errors='coerce').fillna(0.0)
    snapshot['weight'] = pd.to_numeric(snapshot['weight'], errors='coerce').fillna(0.0)
    snapshot['price'] = pd.to_numeric(snapshot['price'], errors='coerce').fillna(0.0)
    snapshot['return_1d'] = pd.to_numeric(snapshot['return_1d'], errors='coerce').fillna(0.0)
    snapshot['open_pnl'] = pd.to_numeric(snapshot['open_pnl'], errors='coerce').fillna(0.0)
    snapshot['abs_contrib'] = snapshot['contribution_1d'].abs()
    snapshot = snapshot.sort_values('abs_contrib', ascending=False)
    
    lines = []
    lines.append(f"| Symbol | Type | Weight | Price | 1D Ret | YTD Ret | Contrib | P&L |")
    lines.append(f"|--------|------|--------|-------|--------|---------|---------|-----|")
    
    for _, row in snapshot.head(20).iterrows():  # Top 20 by impact
        symbol = row['symbol']
        pos_type = row['position_type']
        weight = (row['weight'] * 100) if pd.notna(row['weight']) else 0
        price = row['price'] if pd.notna(row['price']) else 0
        ret = row['return_1d'] if pd.notna(row['return_1d']) else 0
        ytd = row['return_ytd'] if pd.notna(row['return_ytd']) else 0
        contrib = row['contribution_1d'] if pd.notna(row['contribution_1d']) else 0
        pnl = row['open_pnl'] if pd.notna(row['open_pnl']) else 0
        
        lines.append(f"| {symbol:<6} | {pos_type:5} | {weight:>5.1f}% | ${price:>7.2f} | {ret:>+5.2f}% | {ytd:>+6.2f}% | {contrib:>+6.1f}bp | ${pnl:>+10,.0f} |")
    
    return "\n".join(lines)


def _parse_schwab_symbol(bloom: object, name: object) -> str:
    """Extract ticker from Schwab Bloomberg-style symbol field."""
    bloom_text = "" if bloom is None or pd.isna(bloom) else str(bloom).strip()
    if bloom_text:
        ticker = bloom_text.split()[0].strip().upper()
        if ticker and ticker != "N/A":
            return ticker

    name_text = "" if name is None or pd.isna(name) else str(name).strip()
    if name_text.upper() == "CASH":
        return "CASH"
    return ""


def _is_cash_position(symbol: object, name: object, currency: object = None) -> bool:
    """Return True when a broker position row represents cash."""
    symbol_text = "" if symbol is None or pd.isna(symbol) else str(symbol).strip().upper()
    name_text = "" if name is None or pd.isna(name) else str(name).strip().upper()
    currency_text = "" if currency is None or pd.isna(currency) else str(currency).strip().upper()

    if name_text in {"CASH", "BASE CASH"}:
        return True
    if symbol_text in {"CASH", "SNAXX", "SNSXX", "BASE_CASH"}:
        return True
    if "CASH" in name_text and currency_text == "BASE":
        return True
    if "MONEY MARKET" in name_text or "MONEY INVESTOR" in name_text:
        return True
    return False


def _build_snapshot_price_map(snapshot: pd.DataFrame | None) -> dict:
    """Create a symbol-level map of return and pricing data from the daily snapshot."""
    if snapshot is None or snapshot.empty:
        return {}

    snapshot_sorted = snapshot.sort_values(
        by=["fetch_status", "market_value_usd"],
        ascending=[True, False],
    ).copy()
    return (
        snapshot_sorted.drop_duplicates(subset=["symbol"])
        .set_index("symbol")[["return_1d", "price", "fetch_status"]]
        .to_dict("index")
    )


def _attach_snapshot_market_data(positions: pd.DataFrame, snapshot: pd.DataFrame | None) -> pd.DataFrame:
    """Enrich raw broker positions with daily return and price context from the snapshot."""
    if positions.empty:
        return positions

    positions = positions.copy()
    price_map = _build_snapshot_price_map(snapshot)
    positions["return_1d"] = positions["symbol"].map(
        lambda symbol: price_map.get(symbol, {}).get("return_1d")
    )
    positions["price"] = positions["symbol"].map(
        lambda symbol: price_map.get(symbol, {}).get("price")
    )
    positions["fetch_status"] = positions["symbol"].map(
        lambda symbol: price_map.get(symbol, {}).get("fetch_status")
    )
    positions["daily_pnl"] = positions["market_value"] * positions["return_1d"] / 100.0
    positions.loc[positions["return_1d"].isna(), "daily_pnl"] = pd.NA

    cash_mask = positions.apply(
        lambda row: _is_cash_position(row["symbol"], row["name"], row["currency"]),
        axis=1,
    )
    positions["is_cash"] = cash_mask
    positions.loc[cash_mask, "return_1d"] = 0.0
    positions.loc[cash_mask, "price"] = 1.0
    positions.loc[cash_mask, "fetch_status"] = "cash"
    positions.loc[cash_mask, "daily_pnl"] = 0.0
    positions.loc[cash_mask & positions["open_pnl"].isna(), "open_pnl"] = 0.0

    return positions


def _load_account_positions(include_cash: bool = True) -> pd.DataFrame:
    """Load raw broker positions with account detail preserved."""
    frames = []

    if SCHWAB_RAW_PATH.exists():
        schwab = pd.read_excel(SCHWAB_RAW_PATH)
        required = ["Account", "Name", "Value", "Bloom", "Shares", "AvgCost", "USD P&L"]
        if all(col in schwab.columns for col in required):
            schwab_df = pd.DataFrame(
                {
                    "broker": "Schwab",
                    "account": schwab["Account"].astype(str).str.strip(),
                    "name": schwab["Name"].astype(str).str.strip(),
                    "symbol": schwab.apply(
                        lambda row: _parse_schwab_symbol(row.get("Bloom"), row.get("Name")),
                        axis=1,
                    ),
                    "position_type": "LONG",
                    "market_value": pd.to_numeric(schwab["Value"], errors="coerce"),
                    "quantity": pd.to_numeric(schwab["Shares"], errors="coerce"),
                    "avg_price": pd.to_numeric(schwab["AvgCost"], errors="coerce"),
                    "open_pnl": pd.to_numeric(schwab["USD P&L"], errors="coerce"),
                    "currency": "USD",
                }
            )
            if not include_cash:
                schwab_df = schwab_df[
                    ~schwab_df.apply(
                        lambda row: _is_cash_position(row["symbol"], row["name"], row["currency"]),
                        axis=1,
                    )
                ]
            frames.append(schwab_df)

    if IBKR_RAW_PATH.exists():
        ibkr = pd.read_excel(IBKR_RAW_PATH)
        required = ["Account", "Name", "Value", "Shares", "AvgCost", "Unrealized P&L", "Currency"]
        if all(col in ibkr.columns for col in required):
            quantities = pd.to_numeric(ibkr["Shares"], errors="coerce")
            market_values = pd.to_numeric(ibkr["Value"], errors="coerce")
            ibkr_df = pd.DataFrame(
                {
                    "broker": "IBKR",
                    "account": ibkr["Account"].astype(str).str.strip(),
                    "name": ibkr["Name"].astype(str).str.strip(),
                    "symbol": ibkr["Name"].astype(str).str.strip().str.upper(),
                    "position_type": [
                        "SHORT" if ((pd.notna(qty) and qty < 0) or (pd.notna(mv) and mv < 0)) else "LONG"
                        for qty, mv in zip(quantities, market_values)
                    ],
                    "market_value": market_values,
                    "quantity": quantities,
                    "avg_price": pd.to_numeric(ibkr["AvgCost"], errors="coerce"),
                    "open_pnl": pd.to_numeric(ibkr["Unrealized P&L"], errors="coerce"),
                    "currency": ibkr["Currency"].astype(str).str.strip(),
                }
            )
            if not include_cash:
                ibkr_df = ibkr_df[
                    ~ibkr_df.apply(
                        lambda row: _is_cash_position(row["symbol"], row["name"], row["currency"]),
                        axis=1,
                    )
                ]
            frames.append(ibkr_df)

    if not frames:
        return pd.DataFrame()

    positions = pd.concat(frames, ignore_index=True)
    positions = positions[positions["symbol"].notna() & (positions["symbol"].str.len() > 0)].copy()
    positions["cost_basis"] = positions["quantity"].abs() * positions["avg_price"]
    positions.loc[positions["avg_price"].isna() | positions["quantity"].isna(), "cost_basis"] = pd.NA

    # If broker P&L is missing but cost basis exists, derive a fallback.
    missing_pnl = positions["open_pnl"].isna() & positions["cost_basis"].notna() & positions["market_value"].notna()
    long_mask = positions["position_type"] == "LONG"
    short_mask = positions["position_type"] == "SHORT"
    positions.loc[missing_pnl & long_mask, "open_pnl"] = (
        positions.loc[missing_pnl & long_mask, "market_value"]
        - positions.loc[missing_pnl & long_mask, "cost_basis"]
    )
    positions.loc[missing_pnl & short_mask, "open_pnl"] = (
        positions.loc[missing_pnl & short_mask, "cost_basis"]
        - positions.loc[missing_pnl & short_mask, "market_value"].abs()
    )

    positions["account_label"] = positions["broker"] + " " + positions["account"]
    return positions.reset_index(drop=True)


def build_account_pnl_data(snapshot: pd.DataFrame) -> dict | None:
    """Build deterministic account-level daily and unrealized P&L tables."""
    if snapshot.empty:
        return None

    positions = snapshot.copy()
    positions['account'] = positions.get('account_number', pd.Series(['Unknown']*len(positions))).fillna('Unknown').astype(str)
    positions['broker'] = "Account"
    positions['name'] = positions.get('security_name', positions['symbol'])
    positions['market_value'] = positions.get('market_value_usd', 0.0)

    positions["sort_mv"] = positions["market_value"].abs().fillna(0.0)
    positions = positions.sort_values(
        by=["broker", "account", "sort_mv", "symbol"],
        ascending=[True, True, False, True],
    )

    account_summaries = []
    account_details = []
    for (broker, account), account_df in positions.groupby(["broker", "account"], sort=False):
        account_df = account_df.copy()
        available_daily = account_df["daily_pnl"].dropna()
        available_open = account_df["open_pnl"].dropna()

        market_value_sum = float(account_df["market_value"].sum())
        daily_pnl_sum = float(available_daily.sum()) if not available_daily.empty else None
        open_pnl_sum = float(available_open.sum()) if not available_open.empty else None
        
        daily_return = None
        if daily_pnl_sum is not None and market_value_sum:
            daily_return = (daily_pnl_sum / market_value_sum) * 100.0

        account_summaries.append(
            {
                "broker": broker,
                "account": account,
                "positions": int(len(account_df)),
                "market_value": market_value_sum,
                "daily_pnl": daily_pnl_sum,
                "daily_return": daily_return,
                "open_pnl": open_pnl_sum,
                "priced_positions": int(account_df["daily_pnl"].notna().sum()),
            }
        )

        detail_rows = []
        for _, row in account_df.iterrows():
            detail_rows.append(
                {
                    "symbol": row["symbol"],
                    "name": row["name"],
                    "position_type": row["position_type"],
                    "quantity": row["quantity"],
                    "market_value": row["market_value"],
                    "return_1d": row["return_1d"],
                    "daily_pnl": row["daily_pnl"],
                    "avg_price": row["avg_price"],
                    "open_pnl": row["open_pnl"],
                }
            )

        account_details.append(
            {
                "broker": broker,
                "account": account,
                "account_label": f"{broker} {account}",
                "rows": detail_rows,
            }
        )

    return {
        "summary_rows": sorted(account_summaries, key=lambda row: abs(row["market_value"]), reverse=True),
        "accounts": account_details,
        "position_count": int(len(positions)),
        "priced_positions": int(positions["daily_pnl"].notna().sum()),
    }


def build_portfolio_totals_data(snapshot: pd.DataFrame, investable_summary: dict | None = None) -> dict | None:
    """Build cash-inclusive portfolio totals and top holdings aggregated across accounts."""
    if snapshot.empty:
        return None

    positions = snapshot.copy()
    positions['market_value'] = positions.get('market_value_usd', 0.0)
    if 'is_cash' not in positions.columns:
        positions['is_cash'] = positions['symbol'].str.contains('CASH', case=False, na=False)
    
    positions["sort_mv"] = positions["market_value"].abs().fillna(0.0)

    total_portfolio_value = float(positions["market_value"].sum())
    cash_value = float(positions.loc[positions["is_cash"], "market_value"].sum())
    invested_positions = positions[~positions["is_cash"]].copy()
    invested_net_exposure = float(invested_positions["market_value"].sum()) if not invested_positions.empty else 0.0
    invested_long_value = float(
        invested_positions.loc[invested_positions["market_value"] > 0, "market_value"].sum()
    ) if not invested_positions.empty else 0.0
    invested_short_value = float(
        invested_positions.loc[invested_positions["market_value"] < 0, "market_value"].sum()
    ) if not invested_positions.empty else 0.0
    invested_gross_exposure = invested_long_value + abs(invested_short_value)

    priced_positions = int(positions["daily_pnl"].notna().sum())
    total_positions = int(len(positions))
    available_open = positions["open_pnl"].dropna()
    available_daily = positions["daily_pnl"].dropna()
    total_open_pnl = float(available_open.sum()) if not available_open.empty else None
    total_daily_pnl = float(available_daily.sum()) if not available_daily.empty else None
    nav_return_1d = None
    if total_daily_pnl is not None and total_portfolio_value:
        nav_return_1d = total_daily_pnl / total_portfolio_value * 100.0

    aggregated_rows = []
    for symbol, group in positions.groupby("symbol", sort=False):
        market_value = float(group["market_value"].sum())
        quantity = pd.to_numeric(group["quantity"], errors="coerce").sum(min_count=1)
        open_values = group["open_pnl"].dropna()
        daily_values = group["daily_pnl"].dropna()
        return_values = group["return_1d"].dropna()
        is_cash = bool(group["is_cash"].all())
        position_type = "CASH" if is_cash else ("LONG" if market_value >= 0 else "SHORT")
        weight_of_nav = (market_value / total_portfolio_value) if total_portfolio_value else None

        aggregated_rows.append(
            {
                "symbol": symbol,
                "position_type": position_type,
                "quantity": float(quantity) if pd.notna(quantity) else None,
                "market_value": market_value,
                "weight_of_nav": weight_of_nav,
                "return_1d": float(return_values.iloc[0]) if not return_values.empty else None,
                "daily_pnl": float(daily_values.sum()) if not daily_values.empty else None,
                "open_pnl": float(open_values.sum()) if not open_values.empty else None,
            }
        )

    aggregated_rows = sorted(
        aggregated_rows,
        key=lambda row: abs(row["market_value"]),
        reverse=True,
    )

    summary_rows = [
        {"metric": "Total Portfolio Value", "value": total_portfolio_value},
        {"metric": "Cash & Equivalents", "value": cash_value},
        {"metric": "Invested Net Exposure", "value": invested_net_exposure},
        {"metric": "Invested Gross Exposure", "value": invested_gross_exposure},
        {"metric": "Daily P&L", "value": total_daily_pnl},
        {"metric": "NAV Return (1D)", "value": nav_return_1d, "kind": "pct"},
        {"metric": "Total Unrealized P&L", "value": total_open_pnl},
        {"metric": "Aggregated Holdings", "value": len(aggregated_rows), "kind": "count"},
    ]

    if investable_summary:
        summary_rows.extend(
            [
                {
                    "metric": "Invested Return (1D)",
                    "value": investable_summary.get("portfolio_return_1d"),
                    "kind": "pct",
                },
                {
                    "metric": "Invested Net Exposure (DB)",
                    "value": investable_summary.get("net_exposure"),
                },
            ]
        )

    return {
        "total_portfolio_value": total_portfolio_value,
        "cash_value": cash_value,
        "invested_net_exposure": invested_net_exposure,
        "invested_gross_exposure": invested_gross_exposure,
        "total_daily_pnl": total_daily_pnl,
        "nav_return_1d": nav_return_1d,
        "total_open_pnl": total_open_pnl,
        "summary_rows": summary_rows,
        "holdings": aggregated_rows,
        "position_count": total_positions,
        "priced_positions": priced_positions,
    }


def build_portfolio_totals_markdown(portfolio_totals_data: dict | None) -> str:
    """Render cash-inclusive portfolio totals and aggregated holdings as markdown."""
    if not portfolio_totals_data or not portfolio_totals_data.get("summary_rows"):
        return ""

    lines = [
        PORTFOLIO_TOTALS_SECTION_HEADER,
        "",
        "Cash-inclusive totals are aggregated directly from the raw Schwab and IBKR broker files, "
        "then rolled up to top-level holdings by symbol.",
        "",
        f"**Coverage:** {portfolio_totals_data['priced_positions']} of {portfolio_totals_data['position_count']} positions have return-based daily P&L.",
        "",
        "**Portfolio Totals**",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]

    for row in portfolio_totals_data["summary_rows"]:
        kind = row.get("kind")
        value = row.get("value")
        if kind == "pct":
            formatted = _fmt_pct_md(value)
        elif kind == "count":
            formatted = "n/a" if value is None else str(int(value))
        else:
            formatted = _fmt_currency_md(value)
        lines.append(f"| {row['metric']} | {formatted} |")

    lines.extend(
        [
            "",
            "**Aggregated Top Holdings**",
            "",
            "| Symbol | Side | Qty | Market Value | % of NAV | 1D Ret | Daily P&L | Unrealized P&L |",
            "|--------|------|-----|--------------|----------|--------|-----------|----------------|",
        ]
    )

    for row in portfolio_totals_data.get("holdings", [])[:20]:
        lines.append(
            f"| {row['symbol']} | {row['position_type']} | { _fmt_number_md(row['quantity']) } | "
            f"{_fmt_currency_md(row['market_value'])} | {_fmt_pct_md((row['weight_of_nav'] or 0) * 100)} | "
            f"{_fmt_pct_md(row['return_1d'])} | {_fmt_currency_md(row['daily_pnl'])} | "
            f"{_fmt_currency_md(row['open_pnl'])} |"
        )

    return "\n".join(lines)


def inject_portfolio_totals_section(content: str, portfolio_totals_markdown: str) -> str:
    """Insert cash-inclusive totals after section 1 of the markdown report."""
    if not portfolio_totals_markdown:
        return content

    if PORTFOLIO_TOTALS_SECTION_HEADER in content:
        pattern = re.compile(
            rf"{re.escape(PORTFOLIO_TOTALS_SECTION_HEADER)}.*?(?=\n### 2\. TOP CONTRIBUTORS & DETRACTORS|\Z)",
            re.DOTALL,
        )
        if pattern.search(content):
            return pattern.sub(portfolio_totals_markdown.rstrip() + "\n\n", content, count=1)

    marker = "### 2. TOP CONTRIBUTORS & DETRACTORS"
    if marker in content:
        return content.replace(marker, portfolio_totals_markdown.rstrip() + "\n\n" + marker, 1)

    return content.rstrip() + "\n\n" + portfolio_totals_markdown.rstrip() + "\n"


def build_account_pnl_markdown(account_pnl_data: dict | None) -> str:
    """Render account-level P&L section as markdown."""
    if not account_pnl_data or not account_pnl_data.get("summary_rows"):
        return ""

    lines = [
        ACCOUNT_PNL_SECTION_HEADER,
        "",
        "Daily P&L is computed from current market value times 1D return when price data is available. "
        "Unrealized P&L comes from broker cost-basis fields when present, with a cost-basis fallback when possible.",
        "",
        f"**Coverage:** {account_pnl_data['priced_positions']} of {account_pnl_data['position_count']} positions have daily return-based P&L.",
        "",
        "**Account Summary**",
        "",
        "| Broker | Account | Positions | Market Value | 1D Ret | Daily P&L | Unrealized P&L |",
        "|--------|---------|-----------|--------------|--------|-----------|----------------|",
    ]

    for row in account_pnl_data["summary_rows"]:
        lines.append(
            f"| {row['broker']} | {row['account']} | {row['positions']} | "
            f"{_fmt_currency_md(row['market_value'])} | {_fmt_pct_md(row.get('daily_return'))} | "
            f"{_fmt_currency_md(row['daily_pnl'])} | "
            f"{_fmt_currency_md(row['open_pnl'])} |"
        )

    for account in account_pnl_data["accounts"]:
        lines.extend(
            [
                "",
                f"**{account['account_label']} Positions**",
                "",
                "| Symbol | Name | Side | Qty | Market Value | 1D Ret | Daily P&L | Avg Cost | Unrealized P&L |",
                "|--------|------|------|-----|--------------|--------|-----------|----------|----------------|",
            ]
        )
        for row in account["rows"]:
            lines.append(
                f"| {row['symbol']} | {row['name']} | {row['position_type']} | "
                f"{_fmt_number_md(row['quantity'])} | {_fmt_currency_md(row['market_value'])} | "
                f"{_fmt_pct_md(row['return_1d'])} | {_fmt_currency_md(row['daily_pnl'])} | "
                f"{_fmt_currency_md(row['avg_price'], decimals=2)} | {_fmt_currency_md(row['open_pnl'])} |"
            )

    return "\n".join(lines)


def inject_account_pnl_section(content: str, account_pnl_markdown: str) -> str:
    """Insert account-level P&L section into the markdown report after section 6."""
    if not account_pnl_markdown:
        return content
    if ACCOUNT_PNL_SECTION_HEADER in content:
        pattern = re.compile(
            rf"{re.escape(ACCOUNT_PNL_SECTION_HEADER)}.*?(?=\n### 7\. CONCENTRATION, RISK & SCENARIO ANALYSIS|\Z)",
            re.DOTALL,
        )
        if pattern.search(content):
            return pattern.sub(account_pnl_markdown.rstrip() + "\n\n", content, count=1)

    marker = "### 7. CONCENTRATION, RISK & SCENARIO ANALYSIS"
    if marker in content:
        return content.replace(marker, account_pnl_markdown.rstrip() + "\n\n" + marker, 1)

    return content.rstrip() + "\n\n" + account_pnl_markdown.rstrip() + "\n"


def get_market_context(date: str) -> str:
    """Get Phase 1 market context if available."""
    market_data = get_phase1_market_data(date)
    
    if not market_data:
        return "Phase 1 market data not available for this date."
    
    lines = []
    
    # Category stats
    category_stats = market_data.get('category_stats')
    if category_stats is not None and not category_stats.empty:
        # Tier 1 summary
        tier1 = category_stats[category_stats['category_type'] == 'tier1']
        if not tier1.empty:
            lines.append("ASSET CLASS PERFORMANCE:")
            lines.append("| Category | Avg Return | Count |")
            lines.append("|----------|------------|-------|")
            for _, row in tier1.iterrows():
                lines.append(f"| {row['category_value']} | {row['avg_return']:+.2f}% | {row['count']} |")
            lines.append("")
    
    # Factor returns
    factor_returns = market_data.get('factor_returns')
    if factor_returns is not None and not factor_returns.empty:
        lines.append("FACTOR RETURNS:")
        for _, row in factor_returns.iterrows():
            ret = row['return_1d'] if pd.notna(row['return_1d']) else 0
            lines.append(f"  {row['factor_name']}: {ret:+.2f}%")
    
    if not lines:
        return "Limited Phase 1 market context available."
    
    return "\n".join(lines)


def prepare_prompt_data(portfolio_id: str, date: str) -> dict:
    """Prepare all data for prompt injection."""
    # Load portfolio info
    portfolio = get_portfolio(portfolio_id)
    if not portfolio:
        raise ValueError(f"Portfolio not found: {portfolio_id}")
    
    # Load summary
    summary = get_portfolio_summary(portfolio_id, date)
    if not summary:
        raise ValueError(f"No summary found for {portfolio_id} on {date}")
    
    # Load aggregates
    aggregates = get_aggregates(portfolio_id, date)
    
    # Load snapshot
    snapshot = get_daily_snapshot(portfolio_id, date)
    portfolio_totals = build_portfolio_totals_data(snapshot, summary)
    
    # Get market context
    market_context = get_market_context(date)
    
    # Prepare data dict
    return {
        'date': date,
        'portfolio_id': portfolio_id,
        'portfolio_name': portfolio.get('portfolio_name', portfolio_id),
        'portfolio_summary': format_portfolio_summary(summary, portfolio_totals),
        'top_contributors': format_contributors(summary.get('top_contributors', []), 'Contributors'),
        'top_detractors': format_contributors(summary.get('top_detractors', []), 'Detractors'),
        'regional_breakdown': format_aggregates(aggregates, 'region', snapshot),
        'tier1_breakdown': format_aggregates(aggregates, 'tier1', snapshot),
        'tier2_breakdown': format_aggregates(aggregates, 'tier2', snapshot),
        'sector_breakdown': format_aggregates(aggregates, 'sector', snapshot),
        'account_breakdown': format_aggregates(aggregates, 'account', snapshot),
        'holdings_detail': format_holdings_detail(snapshot),
        'market_context': market_context,
    }


def inject_data_into_prompt(template: str, data: dict) -> str:
    """Inject data into prompt template."""
    result = template
    for key, value in data.items():
        result = result.replace(f"{{{key}}}", str(value))
    return result


def parse_markdown_narratives(md_content: str) -> dict:
    """Parse LLM-generated markdown to extract narrative sections."""
    import re
    
    narratives = {
        'executive_summary': '',
        'key_takeaways': [],
        'what_to_watch': [],
        'contributors_narrative': '',
        'regional_narrative': '',
        'sector_narrative': '',
        'long_short_narrative': '',
        'pnl_narrative': '',
        'risk_narrative': '',
        'market_context_narrative': '',
    }
    
    # Extract executive summary (the blockquote after EXECUTIVE SYNTHESIS)
    exec_match = re.search(r'EXECUTIVE SYNTHESIS.*?\n\n>\s*\**PORTFOLIO PERFORMANCE[^>]*\*\*:?\s*\n?>\s*(.+?)(?:\n\n|\*\*KEY)', md_content, re.DOTALL | re.IGNORECASE)
    if exec_match:
        narratives['executive_summary'] = exec_match.group(1).strip().replace('> ', '').replace('\n', ' ')
    
    # Extract key takeaways
    takeaways_match = re.search(r'\*\*KEY TAKEAWAYS:\*\*\s*\n((?:\d+\..+?\n)+)', md_content, re.DOTALL)
    if takeaways_match:
        lines = takeaways_match.group(1).strip().split('\n')
        for line in lines:
            cleaned = re.sub(r'^\d+\.\s*', '', line.strip())
            if cleaned:
                narratives['key_takeaways'].append(cleaned)
    
    # Extract what to watch
    watch_match = re.search(r'\*\*WHAT TO WATCH:\*\*\s*\n((?:-.+?\n)+)', md_content, re.DOTALL)
    if watch_match:
        lines = watch_match.group(1).strip().split('\n')
        for line in lines:
            cleaned = line.strip().lstrip('- ')
            if cleaned:
                narratives['what_to_watch'].append(cleaned)
    
    # Extract narrative sections (the paragraphs starting with **NARRATIVE:** or just after tables)
    # Extract narrative sections using more robust logic
    # Strategy: Find the section header, then capture everything until the next major section marker
    # explicitly excluding known tables or subsections if possible, or just grabbing the text blocks.
    
    def extract_rich_narrative(start_marker, end_marker_pattern=r'(?:\n##|\n---)'):
        # Find start of section
        start_match = re.search(rf'{start_marker}', md_content, re.IGNORECASE)
        if not start_match:
            return ''
        
        start_pos = start_match.end()
        
        # Find next major section to cap the search
        remaining_text = md_content[start_pos:]
        end_match = re.search(end_marker_pattern, remaining_text)
        
        if end_match:
            section_text = remaining_text[:end_match.start()]
        else:
            section_text = remaining_text
            
        # Clean up the text: remove tables
        # Remove markdown tables
        text_without_tables = re.sub(r'\|.*\|.*\n\|[-:| ]+\|\n(?:\|.*\|\n)*', '', section_text, flags=re.MULTILINE)
        
        # Remove the "Top 5 Contributors" type subheaders if they remain
        text_without_tables = re.sub(r'\*\*Top 5 [^\n]+\*\*', '', text_without_tables)
        
        # What remains are paragraphs and h3 headers. 
        # Convert h3 headers (### Title) to bold paragraph starts or just keep them?
        # HTML template uses {{ narrative }}, so we can keep HTML-friendly formatting or just text.
        # Let's convert markdown h3 to HTML h4 for the template
        formatted_text = re.sub(r'###\s+(.+)', r'<h4>\1</h4>', text_without_tables)
        
        # Convert bold **Text** to <strong>Text</strong>
        formatted_text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', formatted_text)
        
        # Convert newlines to breaks for simple text blocks, but preserve paragraph structure
        # Better: split into paragraphs
        paragraphs = [p.strip() for p in formatted_text.split('\n\n') if p.strip()]
        
        return '\n\n'.join([f'<p>{p}</p>' if not p.startswith('<h') else p for p in paragraphs])

    narratives['contributors_narrative'] = extract_rich_narrative(r'## 2\. TOP CONTRIBUTORS')
    narratives['regional_narrative'] = extract_rich_narrative(r'## 3\. REGIONAL')
    narratives['sector_narrative'] = extract_rich_narrative(r'## 4\. SECTOR')
    narratives['long_short_narrative'] = extract_rich_narrative(r'## 5\. LONG VS SHORT')
    narratives['pnl_narrative'] = extract_rich_narrative(r'## 6\. P&L')
    narratives['risk_narrative'] = extract_rich_narrative(r'## 7\. CONCENTRATION')
    narratives['market_context_narrative'] = extract_rich_narrative(r'## 8\. MARKET CONTEXT')
    
    return narratives


def generate_pdf_prince(portfolio_id: str, date: str,
                        summary: dict, aggregates: pd.DataFrame,
                        snapshot: pd.DataFrame, pdf_path: Path,
                        md_content: str = None,
                        account_pnl_data: dict | None = None,
                        portfolio_totals_data: dict | None = None) -> bool:
    """Generate PDF using PrinceXML with professional template."""
    try:
        from utils.pdf_prince.convert import convert_to_pdf, PRINCE_AVAILABLE
        
        if not PRINCE_AVAILABLE:
            return False
        
        # Parse markdown for narratives if provided
        narratives = {}
        if md_content:
            narratives = parse_markdown_narratives(md_content)
        
        # Get top contributors/detractors from summary
        top_contributors = summary.get('top_contributors', [])
        top_detractors = summary.get('top_detractors', [])
        
        if isinstance(top_contributors, str):
            top_contributors = json.loads(top_contributors)
        if isinstance(top_detractors, str):
            top_detractors = json.loads(top_detractors)
        
        # Convert aggregates to list of dicts
        agg_list = aggregates.to_dict('records') if not aggregates.empty else []
        
        # Convert snapshot to list of dicts
        holdings_list = []
        if not snapshot.empty:
            for _, row in snapshot.iterrows():
                holdings_list.append({
                    'symbol': row['symbol'],
                    'position_type': row['position_type'],
                    'weight': row['weight'] if pd.notna(row['weight']) else 0,
                    'market_value_usd': row['market_value_usd'] if pd.notna(row['market_value_usd']) else 0,
                    'return_1d': row['return_1d'] if pd.notna(row['return_1d']) else 0,
                    'contribution_1d': row['contribution_1d'] if pd.notna(row['contribution_1d']) else 0,
                    'open_pnl': row['open_pnl'] if pd.notna(row['open_pnl']) else 0,
                })
        
        # Build data structure for template
        data = {
            'portfolio_id': portfolio_id,
            'date': date,
            'summary': summary,
            'aggregates': agg_list,
            'holdings': holdings_list,
            'account_pnl': account_pnl_data,
            'portfolio_totals': portfolio_totals_data,
            'executive_summary': narratives.get('executive_summary') or f"Portfolio returned {(summary.get('portfolio_return_1d') or 0):+.2f}% today.",
            'key_takeaways': narratives.get('key_takeaways', []),
            'what_to_watch': narratives.get('what_to_watch', []),
            'contributors_narrative': narratives.get('contributors_narrative', ''),
            'regional_narrative': narratives.get('regional_narrative', ''),
            'sector_narrative': narratives.get('sector_narrative', ''),
            'long_short_narrative': narratives.get('long_short_narrative', ''),
            'pnl_narrative': narratives.get('pnl_narrative', ''),
            'risk_narrative': narratives.get('risk_narrative', ''),
            'market_context_narrative': narratives.get('market_context_narrative', ''),
        }
        
        result = convert_to_pdf(data, str(pdf_path))
        return result is not None
        
    except Exception as e:
        print(f"  ⚠️  PrinceXML PDF generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def markdown_to_pdf(md_path: Path, pdf_path: Path) -> bool:
    """Convert markdown to PDF using WeasyPrint (fallback)."""
    try:
        import markdown
        from weasyprint import HTML, CSS
        
        # Read markdown
        md_content = md_path.read_text()
        
        # Convert to HTML
        html_content = markdown.markdown(
            md_content,
            extensions=['tables', 'fenced_code']
        )
        
        # Wrap in HTML document with styling
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 900px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3 {{
                    color: #1a1a2e;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 8px;
                }}
                h1 {{ font-size: 24px; }}
                h2 {{ font-size: 20px; }}
                h3 {{ font-size: 16px; }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 16px 0;
                    font-size: 12px;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #1a1a2e;
                    color: white;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                blockquote {{
                    background: #f0f4f8;
                    border-left: 4px solid #1a1a2e;
                    margin: 16px 0;
                    padding: 12px 20px;
                }}
                code {{
                    background: #f4f4f4;
                    padding: 2px 6px;
                    border-radius: 3px;
                }}
                ul, ol {{
                    margin: 8px 0;
                }}
                li {{
                    margin: 4px 0;
                }}
                hr {{
                    border: none;
                    border-top: 1px solid #eee;
                    margin: 24px 0;
                }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # Generate PDF
        HTML(string=full_html).write_pdf(str(pdf_path))
        return True
        
    except Exception as e:
        print(f"  ⚠️  PDF generation failed: {e}")
        return False


def _missing_section_numbers(content: str) -> list[int]:
    """Return required section numbers missing from the generated markdown."""
    missing = []
    for section_num in sorted(REQUIRED_SECTIONS):
        pattern = rf"^\s*#{{2,3}}\s*{section_num}\."
        if not re.search(pattern, content, flags=re.MULTILINE):
            missing.append(section_num)
    return missing


def _build_continuation_prompt(data: dict, current_content: str, missing_numbers: list[int]) -> str:
    """Build a focused continuation prompt that asks only for missing sections."""
    missing_headers = [REQUIRED_SECTIONS[n] for n in missing_numbers]
    recent_context = current_content[-3000:]

    return f"""The previous draft was cut off due output limits.

Return ONLY the missing sections below, in order:
{chr(10).join(missing_headers)}

Rules:
- Start directly with the first missing section header.
- Do not repeat sections already written.
- Keep narrative-dense analysis and avoid extra tables.
- Use only data from the summary below.

CURRENT DRAFT (tail for continuity):
{recent_context}

DATA SUMMARY:
TODAY'S DATE: {data['date']}
PORTFOLIO ID: {data['portfolio_id']}
PORTFOLIO NAME: {data['portfolio_name']}

PORTFOLIO SUMMARY:
{data['portfolio_summary']}

TOP CONTRIBUTORS:
{data['top_contributors']}

TOP DETRACTORS:
{data['top_detractors']}

REGIONAL BREAKDOWN:
{data['regional_breakdown']}

TIER-1 BREAKDOWN:
{data['tier1_breakdown']}

TIER-2 BREAKDOWN:
{data['tier2_breakdown']}

SECTOR BREAKDOWN:
{data['sector_breakdown']}

HOLDINGS DETAIL:
{data['holdings_detail']}

MARKET CONTEXT:
{data['market_context']}
"""


def generate_portfolio_report(portfolio_id: str, date: str,
                              verbose: bool = True) -> dict:
    """
    Generate portfolio daily wrap report.
    
    Args:
        portfolio_id: Portfolio identifier
        date: Target date (YYYY-MM-DD)
        verbose: Print progress
        
    Returns:
        Dict with generation results
    """
    if verbose:
        print("=" * 70)
        print("PORTFOLIO REPORT GENERATION")
        print("=" * 70)
        print(f"\nPortfolio: {portfolio_id}")
        print(f"Date: {date}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load prompt template
    if verbose:
        print("\n[1] Loading prompt template...")
    system_prompt, user_template = load_prompt_template()
    
    # Prepare data
    if verbose:
        print("[2] Preparing portfolio data...")
    data = prepare_prompt_data(portfolio_id, date)
    
    # Inject data into prompt
    user_prompt = inject_data_into_prompt(user_template, data)
    
    if verbose:
        print(f"    Prompt length: {len(user_prompt):,} characters")
    
    # Generate report
    if verbose:
        print(f"[3] Generating report with {get_report_model()}...")

    result = generate_report(system_prompt, user_prompt, max_tokens=REPORT_MAX_TOKENS)
    
    if 'error' in result:
        raise Exception(f"Report generation failed: {result['error']}")
    
    raw_content = result['content']
    total_tokens_input = result.get('tokens_input', 0) or 0
    total_tokens_output = result.get('tokens_output', 0) or 0
    total_time_ms = result.get('time_ms', 0) or 0

    # If output is truncated before all required sections, request each missing
    # section individually to guarantee completion.
    missing_numbers = _missing_section_numbers(raw_content)
    if missing_numbers and verbose:
        missing_headers = ", ".join(REQUIRED_SECTIONS[n] for n in missing_numbers)
        print(f"    Continuation required for missing sections: {missing_headers}")

    section_completion_attempts = 2
    for section_num in list(missing_numbers):
        section_completed = False
        for attempt_idx in range(section_completion_attempts):
            if section_num not in _missing_section_numbers(raw_content):
                section_completed = True
                break

            if verbose:
                print(
                    f"    Filling {REQUIRED_SECTIONS[section_num]} "
                    f"(attempt {attempt_idx + 1}/{section_completion_attempts})"
                )

            continuation_prompt = _build_continuation_prompt(
                data,
                raw_content,
                [section_num],
            )
            continuation_result = generate_report(
                system_prompt,
                continuation_prompt,
                max_tokens=CONTINUATION_MAX_TOKENS,
            )

            if 'error' in continuation_result:
                raise Exception(
                    f"Continuation generation failed: {continuation_result['error']}"
                )

            raw_content = raw_content.rstrip() + "\n\n" + continuation_result['content'].lstrip()
            total_tokens_input += continuation_result.get('tokens_input', 0) or 0
            total_tokens_output += continuation_result.get('tokens_output', 0) or 0
            total_time_ms += continuation_result.get('time_ms', 0) or 0

        if section_num not in _missing_section_numbers(raw_content):
            section_completed = True

        if not section_completed:
            raise Exception(f"Could not complete required section: {REQUIRED_SECTIONS[section_num]}")

    final_missing = _missing_section_numbers(raw_content)
    if final_missing:
        missing_headers = ", ".join(REQUIRED_SECTIONS[n] for n in final_missing)
        raise Exception(f"Report incomplete after continuation passes. Missing: {missing_headers}")

    # Add title header
    content = f"# Portfolio Report\n\n**Portfolio: {portfolio_id} | Date: {date}**\n\n---\n\n" + raw_content

    # Build deterministic account-level P&L section from raw broker files and
    # the daily snapshot so the markdown/PDF always contain exact account tables.
    summary = get_portfolio_summary(portfolio_id, date)
    aggregates = get_aggregates(portfolio_id, date)
    snapshot = get_daily_snapshot(portfolio_id, date)
    portfolio_totals_data = build_portfolio_totals_data(snapshot, summary)
    portfolio_totals_markdown = build_portfolio_totals_markdown(portfolio_totals_data)
    content = inject_portfolio_totals_section(content, portfolio_totals_markdown)
    account_pnl_data = build_account_pnl_data(snapshot)
    account_pnl_markdown = build_account_pnl_markdown(account_pnl_data)
    content = inject_account_pnl_section(content, account_pnl_markdown)

    if verbose:
        print(f"    ✓ Generated {len(content):,} characters")
        if result.get('fallback_from'):
            print(f"    Model fallback: {result.get('fallback_from')} -> {result.get('model')}")
        print(f"    Model: {result.get('model', 'unknown')}")
        print(f"    Tokens: {total_tokens_input:,} in / {total_tokens_output:,} out")
        print(f"    Time: {total_time_ms:,}ms")
    
    # Create output directory
    output_dir = OUTPUT_DIR / portfolio_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save markdown
    md_filename = f"portfolio_wrap_{date}.md"
    md_path = output_dir / md_filename
    md_path.write_text(content)
    
    if verbose:
        print(f"\n[4] Saved markdown: {md_path}")
    
    # Generate PDF - try PrinceXML first, then WeasyPrint
    if verbose:
        print("[5] Generating PDF...")
    
    pdf_filename = f"portfolio_wrap_{date}.pdf"
    pdf_path = output_dir / pdf_filename
    
    # Try PrinceXML first (produces professional sell-side quality)
    pdf_success = generate_pdf_prince(
        portfolio_id,
        date,
        summary,
        aggregates,
        snapshot,
        pdf_path,
        md_content=content,
        account_pnl_data=account_pnl_data,
        portfolio_totals_data=portfolio_totals_data,
    )
    
    if not pdf_success:
        if verbose:
            print("    Falling back to WeasyPrint...")
        pdf_success = markdown_to_pdf(md_path, pdf_path)
    
    if verbose and pdf_success:
        print(f"    ✓ Saved PDF: {pdf_path}")
    
    # Save to database
    report_id = str(uuid.uuid4())[:8]
    save_report(
        report_id=report_id,
        portfolio_id=portfolio_id,
        report_date=date,
        content_md=content,
        model_name=result.get('model', 'unknown'),
        pdf_path=str(pdf_path) if pdf_success else None,
        tokens_input=total_tokens_input,
        tokens_output=total_tokens_output,
        generation_time_ms=total_time_ms,
    )
    
    if verbose:
        print(f"\n[6] Saved report to database (ID: {report_id})")
    
    # Print summary
    if verbose:
        print("\n" + "=" * 70)
        print("REPORT GENERATION COMPLETE")
        print("=" * 70)
        print(f"\nOutput files:")
        print(f"  Markdown: {md_path}")
        if pdf_success:
            print(f"  PDF: {pdf_path}")
        print(f"\nReport preview (first 500 chars):")
        print("-" * 40)
        print(content[:500])
        print("-" * 40)
    
    return {
        'report_id': report_id,
        'portfolio_id': portfolio_id,
        'date': date,
        'md_path': str(md_path),
        'pdf_path': str(pdf_path) if pdf_success else None,
        'content_length': len(content),
        'tokens_input': total_tokens_input,
        'tokens_output': total_tokens_output,
        'generation_time_ms': total_time_ms,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate portfolio daily wrap report'
    )
    parser.add_argument('--portfolio', required=True,
                        help='Portfolio ID')
    parser.add_argument('--date', required=True,
                        help='Target date (YYYY-MM-DD)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    
    args = parser.parse_args()
    
    try:
        result = generate_portfolio_report(
            portfolio_id=args.portfolio,
            date=args.date,
            verbose=not args.quiet
        )
        
        print("\n✓ Report generation successful")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
