#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: main.py
=============================================================================

INPUT FILES:
    - data/universe.xlsx        (run build_universe.py once to create)
    - data/report.db            (created/updated automatically)
    - Live Schwab + IBKR APIs   (holdings; interactive preflights)
    - Yahoo Finance             (prices; one batched download)
    - report/prompts/system.md  (LLM system prompt)

OUTPUT FILES:
    - outputs/unified/Unified_Report_<date>.pdf   (the report)
    - outputs/unified/Unified_Report_<date>.md    (canonical text)
    - outputs/unified/Unified_Report_<date>.html  (intermediate)
    - data/report.db            (prices, snapshots, report archive)
    - data/holdings.xlsx        (fresh holdings snapshot)

VERSION: 1.0
LAST UPDATED: 2026-06-09
AUTHOR: Arjun Divecha

DESCRIPTION:
    THE unified daily report pipeline - one command replaces the legacy
    Phase 0/2/4 chain:

        1. holdings   - pull Schwab + IBKR (interactive preflights,
                        approved stale fallback)
        2. prices     - one batched yfinance download (universe + holdings)
        3. analytics  - market, portfolio, and bridge computations
        4. report     - one Claude Opus call with extended thinking
        5. render     - PDF via PrinceXML, archive to report.db

USAGE:
    python3 report/main.py                  # full run
    python3 report/main.py --no-llm         # everything except LLM+PDF
    python3 report/main.py --non-interactive  # never prompt (cron mode)
    python3 report/main.py --date 2026-06-09  # analytics as-of date
=============================================================================
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PATHS, SETTINGS, BENCHMARK, TAG_VIEW_EXTRA_TICKERS, ensure_dirs
import analytics
import data as data_mod
import db
import holdings as holdings_mod
import llm as llm_mod
import names as names_mod
import pdf as pdf_mod
import prompt as prompt_mod
import tag_analytics
import tags as tags_mod


def _load_gmo() -> pd.DataFrame:
    """Load GMO holdings from GMO.xlsx into the standard holdings format."""
    gmo_path = PATHS.get("gmo_xlsx")
    if not gmo_path or not gmo_path.exists():
        return pd.DataFrame()
    raw = pd.read_excel(gmo_path)
    ticker_col = "Ticker.1" if "Ticker.1" in raw.columns else "Ticker"
    rows = []
    for _, r in raw.iterrows():
        ticker = str(r.get(ticker_col, "")).strip()
        if not ticker or ticker == "nan":
            continue
        # Strip "ISIN " prefix so "ISIN IE00BF199475" becomes "IE00BF199475"
        # Handle non-breaking spaces (\xa0) from Excel
        ticker = ticker.replace("\xa0", " ")
        if ticker.upper().startswith("ISIN "):
            ticker = ticker[5:].strip()
        qty = float(r.get("Shares", 0) or 0)
        mv = float(r.get("Value", 0) or 0)
        avg = float(r.get("AvgCost", 0) or 0)
        pnl = float(r.get("USD P&L", 0) or 0)
        px = r.get("PX_LAST", 0)
        px = float(px) if pd.notna(px) else 0.0
        is_cash = (abs(avg - 1.0) < 0.01 and abs(px - 1.0) < 0.01)
        rows.append({
            "account": "GMO", "symbol": "CASH" if is_cash else ticker,
            "quantity": qty, "avg_price": avg if avg else float("nan"),
            "market_value": mv if mv else float("nan"),
            "open_pnl": pnl,
            "broker": "GMO", "fetched_at": "",
        })
    return pd.DataFrame(rows)


def run(no_llm: bool = False, interactive: bool = True,
        asof: str = None) -> dict:
    t_start = time.time()
    print("=" * 70)
    print("UNIFIED DAILY MARKET & PORTFOLIO REPORT")
    print("=" * 70)

    ensure_dirs()
    db.init_schema()

    # ------------------------------------------------------------------ 1
    print("\n[1/5] HOLDINGS")
    holdings_df, holdings_meta = holdings_mod.get_holdings(interactive=interactive)

    # ------------------------------------------------------------------ 2
    print("\n[2/5] PRICES")
    universe = data_mod.load_universe()
    db.sync_assets(universe)
    holding_syms = holdings_mod.priceable_symbols(holdings_df)
    gmo_df = _load_gmo()
    gmo_syms = holdings_mod.priceable_symbols(gmo_df) if not gmo_df.empty else []
    aliases = SETTINGS.get("holding_price_aliases", {})
    alias_targets = set(aliases)
    alias_sources = {aliases[s] for s in holding_syms if s in aliases}
    universe_tickers = set(universe["yf_ticker"])
    # Benchmark legs (ACWI/TLT) and the VIX are pulled purely for the tier-3
    # tag views; they are never held, but the fetch must include them.
    extra = set(TAG_VIEW_EXTRA_TICKERS) if SETTINGS.get("enable_tag_views") else set()
    all_tickers = ((universe_tickers | set(holding_syms) | set(gmo_syms)
                    | alias_sources | extra)
                   - (alias_targets - universe_tickers))
    all_tickers = sorted(all_tickers)
    long_df = data_mod.fetch_prices(all_tickers)
    long_df = data_mod.apply_holding_price_aliases(
        long_df, holdings_df, aliases=aliases)
    data_mod.store_prices(long_df)
    prices = db.load_prices()
    # Drop US market-holiday rows (e.g. Juneteenth) BEFORE any analytics, so
    # returns are computed against the last REAL session, never a holiday's
    # NaNs. This is the single chokepoint - every compute_* sees clean prices.
    prices = data_mod.filter_sparse_rows(prices)
    asof = asof or data_mod.latest_trading_date(prices)
    print(f"  Analytics as-of: {asof}")

    # ------------------------------------------------------------------ 3
    print("\n[3/5] ANALYTICS")
    market = analytics.compute_market(prices, universe, asof)
    portfolio = analytics.compute_portfolio(
        holdings_df, prices, asof, holdings_stale=holdings_meta["stale"])
    bridge = analytics.compute_bridge(market, portfolio, universe=universe)

    # Sub-portfolio returns (per-account + GMO)
    gmo_df = _load_gmo()
    subportfolios = analytics.compute_subportfolios(
        holdings_df, prices, asof, gmo_holdings=gmo_df)
    n_subs = len(subportfolios)
    print(f"  Sub-portfolios: {n_subs} accounts computed")

    # Tier-3 tag views (multi-label analytics) — purely additive. Tagged from
    # the DeepSeek cache/universe; benchmark = 60/40 ACWI/TLT (tags pinned).
    tag_views = None
    if SETTINGS.get("enable_tag_views"):
        try:
            held = list(portfolio["positions"].index)
            tmap = {s: v["tags"]
                    for s, v in tags_mod.resolve_tags(held, fetch=True).items()}
            bt = tags_mod.resolve_tags([t for t, _ in BENCHMARK], fetch=True)
            bench_specs = [(w, tag_analytics._tags(bt[t]["tags"]))
                           for t, w in BENCHMARK]
            at = market["asset_table"]
            vix_level = None
            if "^VIX" in prices.columns:
                vser = prices["^VIX"].dropna()
                vix_level = float(vser.iloc[-1]) if len(vser) else None
            spy_move = (float(at.loc["SPY", "return_1d"])
                        if "SPY" in at.index
                        and pd.notna(at.loc["SPY", "return_1d"]) else None)
            tag_views = tag_analytics.build_tag_views(
                market, portfolio, tmap, bench_specs,
                vix_level=vix_level, index_move=spy_move)
            print(f"  Tag views: {len(tag_views['port_tilts'])} book tilts, "
                  f"{len(tag_views['market_tilts'])} market tag groups, "
                  f"day-type={tag_views['noise_gate']['verdict']}")
        except Exception as e:  # additive-only: never sink the daily report
            import traceback
            traceback.print_exc()
            print(f"  ⚠️  Tag views SKIPPED (report continues without them): {e}")
            tag_views = None

    s = portfolio["summary"]
    print(f"  Market: {market['data_quality']['n_priced_today']} assets priced "
          f"({market['data_quality']['coverage_pct']}%)")
    print(f"  Portfolio: {s['n_positions']} positions | "
          f"1d {s['return_1d']:+.2f}% | alpha {s['alpha_1d']:+.2f}% | "
          f"net ${s['net_exposure']:,.0f}")

    # persist today's snapshot
    pos = portfolio["positions"].reset_index()
    pos["account"] = ""
    pos["position_type"] = pos["market_value_mtm"].map(
        lambda v: "SHORT" if v < 0 else "LONG")
    pos["market_value"] = pos["market_value_mtm"]
    pos["holdings_stale"] = int(holdings_meta["stale"])
    db.save_portfolio_snapshot(asof, pos, {
        **s, "holdings_as_of": holdings_meta["as_of"],
    })
    print(f"  Snapshot saved to report.db for {asof}")

    # ------------------------------------------------------------------ 4
    print("\n[4/5] DATA PACKAGE & LLM")
    history = db.get_portfolio_history(30)
    prior = db.get_prior_summaries(asof, SETTINGS["continuity_days"])

    # Resolve full security NAMES (cached) for every ticker the report shows -
    # held positions plus today's movers - so the report uses names, not symbols.
    name_syms = (set(portfolio["positions"].index)
                 | set(market["movers_up"].index)
                 | set(market["movers_down"].index))
    name_map = names_mod.resolve_names(name_syms)

    package = prompt_mod.build_data_package(
        market, portfolio, bridge, history, prior, holdings_meta,
        subportfolios=subportfolios, name_map=name_map, tag_views=tag_views)
    pkg_path = PATHS["output_dir"] / f"Data_Package_{asof}.md"
    pkg_path.write_text(package)
    print(f"  Data package: {len(package):,} chars -> {pkg_path}")

    if no_llm:
        print("\n  --no-llm: stopping before LLM call.")
        print(f"\nDone (dry run) in {time.time() - t_start:.0f}s")
        return {"package": pkg_path}

    result = llm_mod.generate_report(package)

    # ------------------------------------------------------------------ 5
    print("\n[5/5] RENDER & ARCHIVE")
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    stale_msg = (f"using snapshot as of {holdings_meta['as_of']}"
                 if holdings_meta["stale"] else "")
    artifacts = pdf_mod.render_pdf(
        result["text"], asof, generated, result["model"],
        PATHS["output_dir"], stale=holdings_meta["stale"], stale_msg=stale_msg)

    exec_summary = llm_mod.extract_executive_summary(result["text"])
    db.save_report(asof, exec_summary, result["text"], result["model"],
                   result["tokens_input"], result["tokens_output"],
                   result["elapsed_ms"], str(artifacts["pdf"]))

    print(f"\nDone in {time.time() - t_start:.0f}s")
    print(f"  PDF:      {artifacts['pdf']}")
    print(f"  Markdown: {artifacts['md']}")
    return artifacts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified daily report")
    parser.add_argument("--no-llm", action="store_true",
                        help="Run everything except the LLM call and PDF")
    parser.add_argument("--non-interactive", action="store_true",
                        help="Never prompt; fall back to stale holdings")
    parser.add_argument("--date", default=None,
                        help="Analytics as-of date (YYYY-MM-DD)")
    args = parser.parse_args()
    run(no_llm=args.no_llm, interactive=not args.non_interactive,
        asof=args.date)
