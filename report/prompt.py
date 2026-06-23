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
            # NEVER print "n/a" - an em dash marks a genuinely-undefined cell
            # (the project's NO-NA rule); stale values are marked with * upstream
            show[col] = show[col].map(
                lambda v: float_fmt.format(v) if pd.notna(v) else "—")
    return show.to_markdown(index=True)


def _prose_only(text: str) -> str:
    """
    Keep only prose from a stored executive summary - drop markdown table rows,
    separators, and headings. Prior summaries are fed back for continuity; we
    want the narrative, not embedded tables or any truncation artifacts.
    """
    out = []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("|") or s.startswith("#"):
            continue                         # table row / heading
        if set(s) <= set("-|: "):
            continue                         # separator / rule
        out.append(s)
    return " ".join(out).strip()


def build_data_package(market: dict, portfolio: dict, bridge: dict,
                       history: pd.DataFrame, prior_summaries: pd.DataFrame,
                       holdings_meta: dict,
                       subportfolios: pd.DataFrame = None,
                       name_map: dict = None) -> str:
    """Assemble the full user-message data package for the LLM.

    name_map: {ticker: full security name}. Every asset is shown by NAME, not
    ticker; an unmapped symbol falls back to itself (the raw ticker/ID).
    """
    parts = []
    asof = market["asof"]
    dq_m = market["data_quality"]
    dq_p = portfolio["data_quality"]
    s = portfolio["summary"]
    name_map = name_map or {}

    def nm(sym) -> str:
        """Ticker -> full name (fallback: the ticker/ID itself)."""
        return name_map.get(str(sym), str(sym))

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
            f"included in exposure): "
            f"{', '.join(nm(x) for x in dq_p['unpriced_symbols'])} "
            f"(total value ${dq_p['unpriced_value']:,.0f})")

    # ---------------- factors ----------------
    # Factor labels (EM, Nasdaq100, Growth, ...) ARE the names; the underlying
    # ETF ticker column is dropped so the report shows no symbols.
    parts.append("\n## FACTOR COMPLEX (15 factors)")
    ft = market["factor_table"].reset_index()
    ft = ft.rename(columns={ft.columns[0]: "etf"})
    ft = ft[["factor_name", "return_1d", "return_1w", "return_1m",
             "return_ytd", "vol_60d"]].sort_values("return_1d", ascending=False)
    parts.append(_md_table(ft.set_index("factor_name")))

    # factor correlation extremes (informative pairs only) - label by factor
    # NAME, not the underlying ETF ticker (no symbols in the package).
    corr = market["factor_corr"]
    if not corr.empty:
        fac_name = market["factor_table"]["factor_name"].to_dict()
        fn = lambda t: fac_name.get(t, str(t))
        pairs = corr.stack()
        pairs = pairs[pairs.index.get_level_values(0)
                      < pairs.index.get_level_values(1)]
        lo = pairs.nsmallest(5)
        parts.append("\nLowest 60d factor correlations (diversification "
                     "currently working):")
        for (a, b), v in lo.items():
            parts.append(f"- {fn(a)} / {fn(b)}: {v:.2f}")

    # ---------------- market tiers ----------------
    parts.append("\n## ASSET CLASS SUMMARY (tier-1, equal-weighted %, n = assets)")
    parts.append(_md_table(market["tier1_summary"]))

    t2 = market["tier2_summary"]
    t2_show = t2[t2["n"] >= 3]
    parts.append(f"\n## THEME SUMMARY (tier-2 with >=3 assets, equal-weighted %)")
    parts.append(_md_table(t2_show))

    # ---------------- movers & streaks ----------------
    # Show the full NAME (from name_map), not the ticker index or the polluted
    # universe "name" column.
    def _movers_show(df):
        d = df.copy()
        d = d.reset_index()
        sym_col = d.columns[0]
        d["Name"] = d[sym_col].map(nm)
        keep = ["Name", "tier2", "return_1d", "return_1w", "return_ytd",
                "pctile_1d"]
        return d[keep].set_index("Name")

    parts.append("\n## BIGGEST MOVERS - UP (single assets, %)")
    parts.append(_md_table(_movers_show(market["movers_up"])))
    parts.append("\n## BIGGEST MOVERS - DOWN (single assets, %)")
    parts.append(_md_table(_movers_show(market["movers_down"])))

    if not market["streaks"].empty:
        st = market["streaks"].reset_index()
        sym_col = st.columns[0]
        st["Name"] = st[sym_col].map(nm)
        st = st[["Name", "streak", "return_1d"]].set_index("Name")
        parts.append("\n## STREAKS (>=4 consecutive up(+)/down(-) days)")
        parts.append(_md_table(st))

    # ---------------- portfolio ----------------
    parts.append("\n## PORTFOLIO SUMMARY (live Schwab + IBKR book; excludes GMO)")
    stale_tag = "  ** STALE HOLDINGS **" if s.get("holdings_stale") else ""
    beta = s.get("portfolio_beta")
    exp = s.get("expected_return_1d")
    alpha = s.get("alpha_1d")
    summary_rows = [
        ("As of", f"{s['asof']}{stale_tag}"),
        ("1d Return", f"{s['return_1d']:+.2f}%"),
        ("Expected (beta x S&P 500)",
         f"{exp:+.2f}%" if pd.notna(exp) else "—"),
        ("Alpha (vs single-factor S&P 500)",
         f"{alpha:+.2f}%" if pd.notna(alpha) else "—"),
        ("YTD (current-weights proxy)", f"{s['return_ytd']:+.2f}%"),
        ("Gross / Net",
         f"${s['gross_exposure']:,.0f} / ${s['net_exposure']:,.0f}"),
        ("Long / Short / Cash",
         f"${s['long_value']:,.0f} / ${s['short_value']:,.0f} / "
         f"${s['cash_value']:,.0f}"),
        ("Total value (incl. cash)", f"${s['total_value']:,.0f}"),
        ("Positions",
         f"{s['n_positions']} ({s['n_long']} long, {s['n_short']} short)"),
        ("Portfolio beta (vs S&P 500, 60d)",
         f"{beta:.2f}" if pd.notna(beta) else "—"),
        ("Total open P&L", f"${s['total_open_pnl']:,.0f}"),
    ]
    parts.append("| Metric | Value |")
    parts.append("|---|---|")
    for k, v in summary_rows:
        parts.append(f"| {k} | {v} |")

    # Denominator clarity: today's return is on the priced book only.
    if dq_p["unpriced_symbols"]:
        gross = s["gross_exposure"] or 0
        upv = dq_p["unpriced_value"]
        pct = (100.0 * abs(upv) / gross) if gross else 0.0
        parts.append(
            f"\n_1d Return is on priced positions only ({dq_p['n_priced']} of "
            f"{dq_p['n_positions']} holdings; unpriced & excluded: "
            f"{', '.join(nm(x) for x in dq_p['unpriced_symbols'])} = ${upv:,.0f}, "
            f"~{pct:.1f}% of gross). YTD is a current-weights proxy (assumes "
            f"today's holdings were held all year), not a realized figure._")
    else:
        parts.append(
            "\n_YTD is a current-weights proxy (assumes today's holdings were "
            "held all year), not a realized figure._")

    pos = portfolio["positions"].reset_index()
    pos_cols = ["symbol", "weight", "market_value_mtm", "return_1d",
                "return_ytd", "contribution_bps", "beta_spx", "open_pnl"]
    pos_show = pos[pos_cols].copy()
    pos_show["weight"] = pos_show["weight"] * 100  # show as %
    # full name, with a trailing * when the name's price is STALE (it did not
    # print on the as-of date - its 1d/return columns are last-available values)
    stale_map = (pos.set_index("symbol")["price_stale"].to_dict()
                 if "price_stale" in pos.columns else {})
    pos_show["Name"] = [nm(sym) + ("*" if stale_map.get(sym) else "")
                        for sym in pos_show["symbol"]]
    pos_show = pos_show.drop(columns=["symbol"]).rename(
        columns={"weight": "weight_pct", "market_value_mtm": "value_usd"})
    parts.append("\n## ALL POSITIONS (weight % of gross; contribution in bps)")
    parts.append(_md_table(pos_show.set_index("Name")))
    if any(stale_map.values()):
        parts.append(
            f"_* = STALE price: this name did not print on {asof}; its values "
            f"are the last available (it is excluded from today's book return)._")

    # ---------------- sub-portfolios ----------------
    if subportfolios is not None and not subportfolios.empty:
        parts.append("\n## SUB-PORTFOLIO RETURNS (per account + GMO)")
        sp = subportfolios.copy()
        sp = sp[sp["total_value"].abs() > 100]
        sp["pnl_1d"] = sp.apply(
            lambda r: r["total_value"] * r["return_1d"] / 100.0
            if pd.notna(r["return_1d"]) else float("nan"), axis=1)
        sp["pnl_1d"] = sp["pnl_1d"].map(
            lambda v: f"${v:+,.0f}" if pd.notna(v) else "—")
        sp["total_value"] = sp["total_value"].map(lambda v: f"${v:,.0f}")
        sp["return_1d"] = sp["return_1d"].map(
            lambda v: f"{v:+.2f}%" if pd.notna(v) else "—")
        sp["return_ytd"] = sp["return_ytd"].map(
            lambda v: f"{v:+.2f}%" if pd.notna(v) else "—")
        sp["name"] = sp["name"].replace({"TOTAL": "HOUSEHOLD TOTAL"})
        show = sp[["name", "total_value", "return_1d", "pnl_1d", "return_ytd"]]
        show = show.rename(columns={"total_value": "value", "return_1d": "1d %",
                                     "pnl_1d": "1d $", "return_ytd": "YTD %"})
        parts.append(show.set_index("name").to_markdown())
        parts.append(
            "_HOUSEHOLD TOTAL spans the live book PLUS the separate GMO sleeve; "
            "the PORTFOLIO SUMMARY above is the live Schwab+IBKR book only "
            "(ex-GMO), so the two value and return bases differ - do not conflate "
            "them._")

    parts.append("\n## PORTFOLIO FACTOR EXPOSURE (beta-weighted)")
    # drop the underlying ETF ticker column - factor names only, no symbols
    fe = portfolio["factor_exposure"].drop(columns=["etf"], errors="ignore")
    parts.append(_md_table(fe.set_index("factor")))
    parts.append(
        "_Betas are to overlapping factors (EM/Nasdaq/SPX/etc. co-move), so the "
        "implied per-factor contributions do NOT sum to the realized 1d return - "
        "read them as exposures, not an additive attribution._")

    # ---------------- breadth ----------------
    br = bridge.get("breadth") or {}
    if br.get("n_priced"):
        parts.append("\n## PORTFOLIO BREADTH (priced positions)")
        parts.append(
            f"- Up {br['n_up']}/{br['n_priced']} "
            f"({br['pct_up']:.0f}% of priced names rose), down {br['n_down']}")
        if br["n_with_peer"]:
            parts.append(
                f"- Beat tier-2 peers: {br['n_beating_peers']}/"
                f"{br['n_with_peer']} ({br['pct_beating_peers']:.0f}% of names "
                f"that have a >=3-name peer group) - stock-selection hit rate")
        else:
            parts.append("- Beat tier-2 peers: not applicable today "
                         "(no held name had a >=3-name peer group)")

    # ---------------- bridge ----------------
    att = bridge["attribution"]
    if not att.empty:
        parts.append("\n## ATTRIBUTION vs PEER GROUPS "
                     "(vs_peers = position return minus its theme's mean, %)")
        att_show = att.copy()
        att_show["weight"] = att_show["weight"] * 100
        att_show["Name"] = att_show["symbol"].map(nm)   # full name, not ticker
        att_show = att_show.drop(columns=["symbol"]).rename(
            columns={"weight": "weight_pct"})
        cols_order = ["Name"] + [c for c in att_show.columns if c != "Name"]
        parts.append(_md_table(att_show[cols_order].set_index("Name")))

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
        asof_ts = pd.Timestamp(asof)
        for _, r in prior_summaries.iterrows():
            try:
                days = (asof_ts - pd.Timestamp(r["date"])).days
                ago = " (today)" if days <= 0 else f" ({days} days ago)"
            except Exception:
                ago = ""
            parts.append(f"\n### {r['date']}{ago}\n"
                         f"{_prose_only(r['executive_summary'])}")

    parts.append("\n---\nWrite today's report now, following your "
                 "instructions exactly.")
    return "\n".join(parts)
