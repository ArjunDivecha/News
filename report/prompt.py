#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: prompt.py
=============================================================================

INPUT FILES:
    (none - operates on analytics result dicts passed in by main.py)

OUTPUT FILES:
    (none - returns the user-message data package string for the LLM)

VERSION: 1.0
LAST UPDATED: 2026-06-09
AUTHOR: Arjun Divecha

DESCRIPTION:
    Serializes the computed market/portfolio/bridge analytics into ONE
    compact markdown data package for the LLM. Every number the report is
    allowed to contain must appear here - the LLM is forbidden from using
    anything else (see prompts/system.md "Data discipline").

DEPENDENCIES:
    - pandas

USAGE:
    from prompt import build_data_package
=============================================================================
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _md_table(df: pd.DataFrame, float_fmt: str = "{:.2f}") -> str:
    """Render a DataFrame as a compact markdown table."""
    if df is None or df.empty:
        return "(none)"
    show = df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(
                lambda v: float_fmt.format(v) if pd.notna(v) else "n/a")
    return show.to_markdown(index=True)


def build_data_package(market: dict, portfolio: dict, bridge: dict,
                       history: pd.DataFrame, prior_summaries: pd.DataFrame,
                       holdings_meta: dict) -> str:
    """Assemble the full user-message data package for the LLM."""
    parts = []
    asof = market["asof"]
    dq_m = market["data_quality"]
    dq_p = portfolio["data_quality"]
    s = portfolio["summary"]

    # ---------------- header & data quality ----------------
    parts.append(f"# DATA PACKAGE - {asof}\n")
    parts.append("## DATA QUALITY")
    parts.append(
        f"- Market universe: {dq_m['n_priced_today']}/{dq_m['n_universe']} "
        f"assets priced today ({dq_m['coverage_pct']}%)")
    if dq_m["missing_tickers"]:
        parts.append(f"- Unpriced universe tickers: "
                     f"{', '.join(dq_m['missing_tickers'][:20])}"
                     + (" ..." if len(dq_m["missing_tickers"]) > 20 else ""))
    if holdings_meta.get("stale"):
        parts.append(
            f"- ** HOLDINGS ARE STALE ** - live broker pull failed; using "
            f"snapshot as of {holdings_meta['as_of']}. "
            f"Failures: {'; '.join(holdings_meta.get('failures', []))}")
    else:
        parts.append(f"- Holdings: LIVE, pulled {holdings_meta.get('as_of')}")
    if dq_p["unpriced_symbols"]:
        parts.append(
            f"- Unpriced portfolio positions (excluded from today's return, "
            f"included in exposure): {', '.join(dq_p['unpriced_symbols'])} "
            f"(total value ${dq_p['unpriced_value']:,.0f})")

    # ---------------- factors ----------------
    parts.append("\n## FACTOR COMPLEX (15 factor ETFs)")
    ft = market["factor_table"].reset_index()
    ft = ft.rename(columns={ft.columns[0]: "etf"})
    ft = ft[["factor_name", "etf", "return_1d", "return_1w", "return_1m",
             "return_ytd", "vol_60d"]].sort_values("return_1d", ascending=False)
    parts.append(_md_table(ft.set_index("factor_name")))

    # factor correlation extremes (informative pairs only)
    corr = market["factor_corr"]
    if not corr.empty:
        pairs = corr.stack()
        pairs = pairs[pairs.index.get_level_values(0)
                      < pairs.index.get_level_values(1)]
        lo = pairs.nsmallest(5)
        parts.append("\nLowest 60d factor correlations (diversification "
                     "currently working):")
        for (a, b), v in lo.items():
            parts.append(f"- {a} / {b}: {v:.2f}")

    # ---------------- market tiers ----------------
    parts.append("\n## ASSET CLASS SUMMARY (tier-1, equal-weighted %, n = assets)")
    parts.append(_md_table(market["tier1_summary"]))

    t2 = market["tier2_summary"]
    t2_show = t2[t2["n"] >= 3]
    parts.append(f"\n## THEME SUMMARY (tier-2 with >=3 assets, equal-weighted %)")
    parts.append(_md_table(t2_show))

    # ---------------- movers & streaks ----------------
    cols = ["name", "tier2", "return_1d", "return_1w", "return_ytd", "pctile_1d"]
    parts.append("\n## BIGGEST MOVERS - UP (single assets, %)")
    parts.append(_md_table(market["movers_up"][cols]))
    parts.append("\n## BIGGEST MOVERS - DOWN (single assets, %)")
    parts.append(_md_table(market["movers_down"][cols]))

    if not market["streaks"].empty:
        parts.append("\n## STREAKS (>=4 consecutive up(+)/down(-) days)")
        parts.append(_md_table(market["streaks"]))

    # ---------------- portfolio ----------------
    parts.append("\n## PORTFOLIO SUMMARY")
    stale_tag = "  ** STALE HOLDINGS **" if s.get("holdings_stale") else ""
    parts.append(f"- As of: {s['asof']}{stale_tag}")
    parts.append(f"- Total value (incl. cash): ${s['total_value']:,.0f}  "
                 f"(cash ${s['cash_value']:,.0f})")
    parts.append(f"- Gross ${s['gross_exposure']:,.0f} | Net ${s['net_exposure']:,.0f} "
                 f"| Long ${s['long_value']:,.0f} | Short ${s['short_value']:,.0f}")
    parts.append(f"- Positions: {s['n_positions']} ({s['n_long']} long, "
                 f"{s['n_short']} short)")
    parts.append(f"- Return 1d: {s['return_1d']:+.2f}%  |  YTD: {s['return_ytd']:+.2f}%")
    if pd.notna(s.get("portfolio_beta")):
        parts.append(f"- Portfolio beta (vs SPY, 60d): {s['portfolio_beta']:.2f}")
        parts.append(f"- Expected 1d from beta: {s['expected_return_1d']:+.2f}%  "
                     f"=>  ALPHA today: {s['alpha_1d']:+.2f}%")
    parts.append(f"- Total open P&L: ${s['total_open_pnl']:,.0f}")

    pos = portfolio["positions"].reset_index()
    pos_cols = ["symbol", "weight", "market_value_mtm", "return_1d",
                "return_ytd", "contribution_bps", "beta_spx", "open_pnl"]
    pos_show = pos[pos_cols].copy()
    pos_show["weight"] = pos_show["weight"] * 100  # show as %
    pos_show = pos_show.rename(columns={"weight": "weight_pct",
                                        "market_value_mtm": "value_usd"})
    parts.append("\n## ALL POSITIONS (weight % of gross; contribution in bps)")
    parts.append(_md_table(pos_show.set_index("symbol")))

    parts.append("\n## PORTFOLIO FACTOR EXPOSURE (beta-weighted)")
    parts.append(_md_table(portfolio["factor_exposure"].set_index("factor")))

    # ---------------- bridge ----------------
    att = bridge["attribution"]
    if not att.empty:
        parts.append("\n## ATTRIBUTION vs PEER GROUPS "
                     "(vs_peers = position return minus its theme's mean, %)")
        att_show = att.copy()
        att_show["weight"] = att_show["weight"] * 100
        att_show = att_show.rename(columns={"weight": "weight_pct"})
        parts.append(_md_table(att_show.set_index("symbol")))

    unheld = bridge["unheld_themes"]
    if unheld is not None and not unheld.empty:
        parts.append("\n## BIGGEST-MOVING THEMES THE PORTFOLIO DOES NOT HOLD")
        parts.append(_md_table(unheld.set_index("tier2")
                               [["tier1", "n", "ret_1d", "ret_1w", "ret_ytd"]]))

    # ---------------- history & continuity ----------------
    if history is not None and not history.empty:
        h = history.tail(10)[["date", "return_1d", "return_ytd",
                              "net_exposure", "gross_exposure", "alpha_1d"]]
        parts.append("\n## PORTFOLIO HISTORY (last 10 sessions)")
        parts.append(_md_table(h.set_index("date")))

    if prior_summaries is not None and not prior_summaries.empty:
        parts.append("\n## PRIOR EXECUTIVE SUMMARIES (oldest first)")
        for _, r in prior_summaries.iterrows():
            parts.append(f"\n### {r['date']}\n{r['executive_summary']}")

    parts.append("\n---\nWrite today's report now, following your "
                 "instructions exactly.")
    return "\n".join(parts)
