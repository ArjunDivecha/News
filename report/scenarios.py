#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: scenarios.py
=============================================================================

INPUT FILES:
    (none directly — pure functions over the household positions DataFrame,
     the holdings tag map, config.FUND_LOOKTHROUGH, and the wide price matrix
     passed in by report/main.py)

OUTPUT FILES:
    (none — returns dicts/DataFrames consumed by report/prompt.py)

VERSION: 1.0
LAST UPDATED: 2026-07-01
AUTHOR: Arjun Divecha

DESCRIPTION:
    Scenario risk engine for the daily report — answers "what happens to my
    portfolio if X" for a standing set of macro scenarios. Each scenario is a
    vector of percentage shocks per look-through bucket (asset class x region),
    calibrated to a NAMED historical episode (documented per scenario below),
    plus per-symbol overrides for concentrated or special positions (the
    Vietnam closed-end fund, GMO Beyond China, the growth short, the
    market-neutral GMO Equity Dislocation, single-country EM funds).

    Shocks are applied to the household's ACTUAL look-through slices — the
    same slicing used by the asset-allocation table — so multi-asset funds
    (GMO Benchmark-Free, Baupost policy mix) decompose into their underlying
    exposures before the shock hits. Shorts sign correctly automatically
    (a negative slice x a negative shock = a gain).

    Also computes two standing structural-risk measures:
      - CRASH BETA: the current-weights portfolio's empirical beta to the
        S&P 500 measured only on the worst-decile S&P days of the past year,
        vs its full-sample beta. Divergence = hidden crash convexity.
      - LIQUIDITY LADDER: how much of the household converts to cash in
        roughly how many days (cash / daily-NAV funds / ETFs / closed-end
        with discount risk / LP lockup).

    HONESTY CONTRACT: these are first-order estimates from episode-calibrated
    assumptions, not predictions. Every shock is visible in the output, the
    anchors are named, and the numbers live here in code where they can be
    audited and edited. Nothing is fabricated at runtime: the engine only
    multiplies stated assumptions by actual position values.

DEPENDENCIES:
    - pandas, numpy, tag_analytics (classify_holding / _tags)
=============================================================================
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tag_analytics import classify_holding, _tags  # noqa: E402

# --------------------------------------------------------------------------- #
# The scenario shock matrix.
#
# Shock units: percent return applied to the NET value of each look-through
# slice. Bucket keys: (asset_class, region); (cls, None) is the class default
# (also the catch-all for an unclassified region). "overrides" replace the
# bucket shock for EVERY slice of that symbol.
#
# Calibrations (documented so they can be challenged and edited):
#  1. US -40% deleveraging  (2008-09 GFC): SPX -40 scenario scaling of the GFC
#     cross-section — EAFE fell ~0.95x SPX, EM ~1.1x, frontier/VN worse
#     (VN-Index -66% in 2008) with CEF discounts widening; EM sovereign HY
#     (B/CCC-heavy, like the GMO EM debt sleeve) ~-30; short-duration
#     Treasuries rallied; value-vs-growth market-neutral ~flat.
#  2. Tech/growth crash     (2000-02): Nasdaq -60/SPX -30 over the episode
#     while VALUE was roughly flat-to-down-small — this book's US sleeve is
#     value-tilted and short growth, so the scenario is mild; long-value/
#     short-growth (Equity Dislocation) made large gains (spread ~ +40pp over
#     the episode; +20 used, conservative); rates were cut hard (+10 UST).
#  3. Inflation +300bp      (2022): SPX -25 full-year with growth -29 vs
#     value -7.5 (21pp spread); Agg -13 but this book's US bond sleeve is
#     ~2.1yr duration (GMO BF composition) -> -8 blended with the long muni;
#     EM USD debt -15; cash nominal 0.
#  4. Asia/EM crisis        (1997-98): Asia ex-Japan fell 50-70% in USD
#     (Korea ~-70), EMBI ~-30 into the 1998 Russia default, US/EAFE dipped
#     ~10-12% (LTCM) then recovered; flight-to-quality bid for Treasuries.
#     THE book's kill-shot scenario: no offsetting hedge, and the pain is in
#     its least liquid holdings.
#  5. China/Taiwan event    (hypothetical; calibrated between the 2022 Taiwan
#     -30% drawdown and a 1997-scale regional shock): Taiwan/Korea equity
#     -35/-40, China -30, semis-heavy US -12, broad EM -25. GMO Beyond China
#     is 30.5% Taiwan + 18.6% Korea (its own composition file) -> -35.
#  6. USD +10% spike        (2014-15 / 2022 DXY analogs): EM equity ~-12,
#     EAFE ~-8 in USD terms, EM USD-sovereign spreads widen modestly (the GMO
#     sleeve is 97% USD so FX pass-through is spread-driven, not currency),
#     VND devaluation pressure on Vietnam.
# --------------------------------------------------------------------------- #
SCENARIOS = [
    {
        "key": "us_deleveraging",
        "name": "US equities -40%",
        "anchor": "2008-09 GFC",
        "shocks": {
            ("Equities", "US"): -40, ("Equities", "International"): -38,
            ("Equities", "EM"): -45, ("Equities", None): -40,
            ("Bonds", "US"): 3, ("Bonds", "EM"): -30, ("Bonds", None): -5,
            ("Alternatives", None): -8, ("Cash", None): 0, ("Other", None): -20,
        },
        "overrides": {"VTMEF": -55, "BCHI": -45, "IE00BF199475": 0},
    },
    {
        "key": "tech_crash",
        "name": "Tech/growth crash",
        "anchor": "2000-02 dot-com",
        "shocks": {
            ("Equities", "US"): -12, ("Equities", "International"): -18,
            ("Equities", "EM"): -28, ("Equities", None): -15,
            ("Bonds", "US"): 10, ("Bonds", "EM"): -5, ("Bonds", None): 2,
            ("Alternatives", None): 3, ("Cash", None): 0, ("Other", None): -10,
        },
        "overrides": {"IWF": -40, "VTV": -8, "IE00BF199475": 20,
                      "BCHI": -25, "VTMEF": -20, "INTC": -45},
    },
    {
        "key": "inflation_shock",
        "name": "Inflation / rates +300bp",
        "anchor": "2022 hiking cycle",
        "shocks": {
            ("Equities", "US"): -20, ("Equities", "International"): -15,
            ("Equities", "EM"): -20, ("Equities", None): -18,
            ("Bonds", "US"): -8, ("Bonds", "EM"): -15, ("Bonds", None): -10,
            ("Alternatives", None): 0, ("Cash", None): 0, ("Other", None): -10,
        },
        "overrides": {"IWF": -30, "VTV": -10, "IE00BF199475": 15},
    },
    {
        "key": "asia_crisis",
        "name": "Asia/EM crisis",
        "anchor": "1997-98 Asia/LTCM",
        "shocks": {
            ("Equities", "US"): -10, ("Equities", "International"): -12,
            ("Equities", "EM"): -45, ("Equities", None): -15,
            ("Bonds", "US"): 5, ("Bonds", "EM"): -35, ("Bonds", None): 0,
            ("Alternatives", None): -5, ("Cash", None): 0, ("Other", None): -15,
        },
        "overrides": {"VTMEF": -55, "BCHI": -45, "EWY": -60,
                      "IE00BF199475": 0},
    },
    {
        "key": "taiwan_event",
        "name": "China/Taiwan event",
        "anchor": "hypothetical (2022 TW -30% scaled)",
        "shocks": {
            ("Equities", "US"): -12, ("Equities", "International"): -12,
            ("Equities", "EM"): -25, ("Equities", None): -12,
            ("Bonds", "US"): 4, ("Bonds", "EM"): -12, ("Bonds", None): 0,
            ("Alternatives", None): -3, ("Cash", None): 0, ("Other", None): -10,
        },
        "overrides": {"BCHI": -35, "EWY": -40, "VTMEF": -20, "KTEC": -40,
                      "IE00BF199475": 0},
    },
    {
        "key": "usd_spike",
        "name": "USD +10% spike",
        "anchor": "2014-15 / 2022 DXY",
        "shocks": {
            ("Equities", "US"): 0, ("Equities", "International"): -8,
            ("Equities", "EM"): -12, ("Equities", None): -5,
            ("Bonds", "US"): 1, ("Bonds", "EM"): -8, ("Bonds", None): 0,
            ("Alternatives", None): 0, ("Cash", None): 0, ("Other", None): -3,
        },
        "overrides": {"VTMEF": -15, "IE00BF199475": 0},
    },
]

# --------------------------------------------------------------------------- #
# Liquidity ladder classification
# --------------------------------------------------------------------------- #
LIQUIDITY_BUCKETS = {
    # key: (label, sort order)
    "cash":       ("Cash / same day", 0),
    "etf":        ("Exchange-traded (T+1/T+2)", 1),
    "daily_fund": ("Daily-NAV mutual fund (1-3 days)", 2),
    "cef":        ("Closed-end fund — liquid, discount risk", 3),
    "lockup":     ("LP lockup (quarterly+ redemption)", 4),
    "unpriced":   ("Unpriced / defunct lines", 5),
}
LIQUIDITY_OVERRIDES = {
    "BAUPOST": "lockup",          # Baupost LP — redemption windows
    "VTMEF": "cef",               # Vietnam Enterprise Investments (VEIL CEF)
    "GBMBX": "daily_fund", "GCCHX": "daily_fund", "GMOQX": "daily_fund",
    "IE00BF199475": "daily_fund", # GMO Equity Dislocation UCITS (daily dealing)
    "VCLAX": "daily_fund",
}


def iter_slices(positions: pd.DataFrame, tag_map: dict, cash_value: float,
                lookthrough: dict):
    """Yield (symbol, asset_class, region, net_value) look-through slices —
    the same decomposition the asset-allocation table uses."""
    LT_REGION = {"Equities": "equity_region", "Bonds": "bond_region"}
    for sym, r in positions.iterrows():
        mtm = r.get("market_value_mtm")
        if mtm is None or not np.isfinite(mtm):
            continue
        lt = (lookthrough or {}).get(sym)
        if lt:
            for cls, frac in lt.get("class", {}).items():
                if not frac:
                    continue
                regmap = lt.get(LT_REGION.get(cls, ""), {}) if cls in LT_REGION else {}
                if regmap:
                    for reg, rf in regmap.items():
                        if rf:
                            yield sym, cls, reg, mtm * frac * rf
                else:
                    yield sym, cls, None, mtm * frac
        else:
            cls, reg = classify_holding(_tags(tag_map.get(sym, "")))
            yield sym, cls, (reg if cls in ("Equities", "Bonds") else None), mtm
    if cash_value:
        yield "CASH", "Cash", None, float(cash_value)


def _shock_for(scn: dict, sym: str, cls: str, reg) -> float:
    ov = scn.get("overrides", {})
    if sym in ov:
        return float(ov[sym])
    shocks = scn["shocks"]
    if (cls, reg) in shocks:
        return float(shocks[(cls, reg)])
    return float(shocks.get((cls, None), 0.0))


def compute_scenarios(positions: pd.DataFrame, tag_map: dict,
                      cash_value: float, lookthrough: dict,
                      top_n: int = 3) -> pd.DataFrame:
    """Apply every scenario to the household. Returns one row per scenario:
    name, anchor, impact_pct (of total net value), impact_dollars, hurts
    (top_n worst symbols with $ impact), helps (top_n best), key_shocks."""
    slices = list(iter_slices(positions, tag_map, cash_value, lookthrough))
    total = sum(v for _, _, _, v in slices)
    rows = []
    for scn in SCENARIOS:
        by_sym = {}
        impact = 0.0
        for sym, cls, reg, val in slices:
            d = val * _shock_for(scn, sym, cls, reg) / 100.0
            impact += d
            by_sym[sym] = by_sym.get(sym, 0.0) + d
        ranked = sorted(by_sym.items(), key=lambda kv: kv[1])
        hurts = [(s, round(v)) for s, v in ranked[:top_n] if v <= -50_000]
        helps = [(s, round(v)) for s, v in reversed(ranked[-top_n:])
                 if v >= 50_000]
        # compact, auditable assumption string (equity legs + EM debt)
        sh = scn["shocks"]

        def g(cls, reg):
            return sh.get((cls, reg), sh.get((cls, None), 0.0))
        key_shocks = (f"US eq {g('Equities','US'):+.0f}, "
                      f"Intl {g('Equities','International'):+.0f}, "
                      f"EM eq {g('Equities','EM'):+.0f}, "
                      f"US bd {g('Bonds','US'):+.0f}, "
                      f"EM debt {g('Bonds','EM'):+.0f}")
        ov = scn.get("overrides", {})
        if ov:
            key_shocks += "; " + ", ".join(
                f"{s} {v:+.0f}" for s, v in list(ov.items())[:4])
        rows.append({
            "name": scn["name"], "anchor": scn["anchor"],
            "impact_pct": 100.0 * impact / total if total else np.nan,
            "impact_dollars": impact,
            "hurts": hurts, "helps": helps, "key_shocks": key_shocks,
        })
    df = pd.DataFrame(rows).sort_values("impact_pct").reset_index(drop=True)
    df.attrs["total_value"] = total
    return df


def compute_crash_beta(prices: pd.DataFrame, positions: pd.DataFrame,
                       factor: str = "SPY", q: float = 0.10,
                       lookback: int = 252, min_obs: int = 120) -> dict:
    """Current-weights portfolio beta to the S&P on ALL days vs on the worst-
    decile S&P days of the past year. Divergence (crash > full) = the book is
    more exposed in drawdowns than its headline beta suggests."""
    if factor not in prices.columns:
        return {"full_beta": np.nan, "crash_beta": np.nan, "coverage_pct": 0.0,
                "n_crash_days": 0}
    rets = prices.sort_index().pct_change(fill_method=None).tail(lookback)
    syms = [s for s in positions.index if s in rets.columns]
    gross = positions["market_value_mtm"].abs().sum()
    if not syms or not gross:
        return {"full_beta": np.nan, "crash_beta": np.nan, "coverage_pct": 0.0,
                "n_crash_days": 0}
    w = positions.loc[syms, "market_value_mtm"] / gross          # signed vs gross
    coverage = 100.0 * positions.loc[syms, "market_value_mtm"].abs().sum() / gross
    port = (rets[syms] * w).sum(axis=1, min_count=max(1, len(syms) // 2))
    spy = rets[factor]
    both = pd.concat([port, spy], axis=1, keys=["p", "m"]).dropna()
    if len(both) < min_obs:
        return {"full_beta": np.nan, "crash_beta": np.nan,
                "coverage_pct": round(coverage, 0), "n_crash_days": 0}
    full_beta = both["p"].cov(both["m"]) / both["m"].var()
    worst = both[both["m"] <= both["m"].quantile(q)]
    crash_beta = (worst["p"].cov(worst["m"]) / worst["m"].var()
                  if len(worst) >= 10 and worst["m"].var() > 0 else np.nan)
    return {"full_beta": round(float(full_beta), 2),
            "crash_beta": (round(float(crash_beta), 2)
                           if np.isfinite(crash_beta) else np.nan),
            "coverage_pct": round(coverage, 0),
            "n_crash_days": int(len(worst))}


def compute_liquidity_ladder(positions: pd.DataFrame, cash_value: float,
                             cash_symbols=("CASH", "SNSXX", "SNAXX")) -> pd.DataFrame:
    """Net household value per liquidity bucket, ordered fastest-to-slowest."""
    acc = {}

    def add(bucket, val):
        acc[bucket] = acc.get(bucket, 0.0) + val

    add("cash", float(cash_value or 0.0))
    for sym, r in positions.iterrows():
        mtm = r.get("market_value_mtm")
        if mtm is None or not np.isfinite(mtm) or mtm == 0:
            continue
        if str(sym) in cash_symbols:
            add("cash", mtm)
        elif sym in LIQUIDITY_OVERRIDES:
            add(LIQUIDITY_OVERRIDES[sym], mtm)
        elif not r.get("has_price", True):
            add("unpriced", mtm)
        else:
            add("etf", mtm)
    total = sum(acc.values()) or np.nan
    rows = [{"bucket": LIQUIDITY_BUCKETS[k][0], "value": v,
             "pct": 100.0 * v / total, "_ord": LIQUIDITY_BUCKETS[k][1]}
            for k, v in acc.items() if abs(v) > 1]
    return (pd.DataFrame(rows).sort_values("_ord").drop(columns="_ord")
            .reset_index(drop=True))


def compute_scenario_risk(positions: pd.DataFrame, tag_map: dict,
                          cash_value: float, lookthrough: dict,
                          prices: pd.DataFrame = None) -> dict:
    """Orchestrate the full scenario-risk block for the report package."""
    table = compute_scenarios(positions, tag_map, cash_value, lookthrough)
    out = {
        "table": table,
        "total_value": table.attrs.get("total_value"),
        "liquidity": compute_liquidity_ladder(positions, cash_value),
        "crash_beta": (compute_crash_beta(prices, positions)
                       if prices is not None else None),
    }
    return out
