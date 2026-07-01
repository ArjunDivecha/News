#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: tag_analytics.py
=============================================================================

INPUT FILES:
    (none directly - pure functions over DataFrames/dicts passed in by main.py:
     market["asset_table"] with tier-3 tags + returns, portfolio positions,
     the holdings tag map from tags.resolve_tags, and benchmark tag weights)

OUTPUT FILES:
    (none - returns dicts/DataFrames consumed by prompt.py)

VERSION: 1.0
LAST UPDATED: 2026-07-01
AUTHOR: Arjun Divecha

DESCRIPTION:
    Tier-3 (multi-label tag) analytics for the daily report. Two families:

      MARKET-side ("what kind of day was it"):
        - explode_tags / tag tilts (universe-demeaned mean return per tag, by axis)
        - style spreads (Value-Growth, Defensive-Cyclical, ...)
        - cross-sectional dispersion (macro vs stock-picker's day)
        - two-dimensional breadth (% of tags up)
        - region x sector grid
        - VIX noise gate (Rule of 16)

      PORTFOLIO-side (how the book was positioned), vs a 60/40 ACWI/TLT benchmark:
        - tag active tilts (book tag weight - benchmark tag weight, per axis)
        - the tag BRIDGE (with vs against today's dominant market tilts)
        - tag concentration (HHI / effective number of tags)

    Every function is PURE (data in -> dict/DataFrame out), honors the n>=3 peer
    floor, excludes missing data (never zero-fills), and NEVER emits "n/a".
    Tags are multi-label and correlated, so tilts are independent means-vs-
    universe and are NEVER summed across axes.

DEPENDENCIES:
    - pandas, numpy, expert_review_common (tags_to_string canonical normalizer)
=============================================================================
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
_FT = Path(__file__).resolve().parent.parent / "fine tuning" / "scripts"
sys.path.insert(0, str(_FT))
import expert_review_common as erc   # tags_to_string (canonical normalizer)

# --------------------------------------------------------------------------- #
# canonical tag -> axis map (matches ALLOWED_TIER3_TAGS in expert_review_common)
# --------------------------------------------------------------------------- #
TAG_AXIS = {
    # Asset class
    "Equity": "AssetClass", "Credit": "AssetClass", "Treasury": "AssetClass",
    "Municipal": "AssetClass", "FX": "AssetClass", "Commodity": "AssetClass",
    "Multi-Asset": "AssetClass", "Volatility": "AssetClass",
    "Alternative": "AssetClass",
    # Region
    "US": "Region", "Europe": "Region", "Asia": "Region", "EM": "Region",
    "Global": "Region", "China": "Region", "Japan": "Region", "India": "Region",
    "Canada": "Region", "UAE": "Region", "APAC": "Region", "Australia": "Region",
    "Developed": "Region", "International": "Region", "Domestic": "Region",
    # Sector
    "Tech": "Sector", "Energy": "Sector", "Financials": "Sector",
    "Healthcare": "Sector", "Industrials": "Sector", "Consumer": "Sector",
    "Infrastructure": "Sector", "Real Estate": "Sector", "Utilities": "Sector",
    "Materials": "Sector",
    # Style
    "Value": "Style", "Growth": "Style", "Momentum": "Style", "Quality": "Style",
    "Dividend": "Style", "Defensive": "Style", "Low Volatility": "Style",
    # Strategy
    "Active": "Strategy", "Passive": "Strategy", "Equal-Weight": "Strategy",
    "Thematic": "Strategy", "Quantitative": "Strategy", "Options-Based": "Strategy",
    "Factor-Based": "Strategy", "ESG": "Strategy", "Diversified": "Strategy",
    # Size
    "Small-Cap": "Size", "Mid-Cap": "Size", "Large-Cap": "Size",
    # Duration/Credit quality
    "Investment Grade": "Duration", "High Yield": "Duration",
    "Short Duration": "Duration", "Medium Duration": "Duration",
    "Long Duration": "Duration", "IG Credit": "Duration", "HY Credit": "Duration",
}

# paired spreads for the day-type classifier (long tag, short tag/None=vs-universe)
STYLE_PAIRS = [
    ("Value", "Growth", "Value vs Growth"),
    ("Defensive", "Cyclical", "Defensive vs Cyclical"),  # Cyclical derived below
    ("Quality", None, "Quality vs universe"),
    ("Momentum", None, "Momentum vs universe"),
    ("Low Volatility", None, "Low-Vol vs universe"),
    ("Dividend", None, "Dividend vs universe"),
]
REGION_PAIRS = [("EM", "Developed", "EM vs Developed")]
# "Cyclical" isn't a tag; proxy = mean of these sectors
CYCLICAL_SECTORS = {"Financials", "Energy", "Materials", "Industrials", "Tech"}


def _tags(val) -> list:
    return [t for t in erc.parse_tags(erc.tags_to_string(val)) if t]


def explode_tags(asset_table: pd.DataFrame,
                 ret_cols=("return_1d", "return_1w", "return_1m",
                           "return_ytd")) -> pd.DataFrame:
    """One row per (ticker, canonical tag); factor-proxy rows dropped; funds
    without return_1d excluded. Never zero-fills."""
    df = asset_table.copy()
    if "is_factor" in df.columns:
        df = df[df["is_factor"] != 1]
    df = df.dropna(subset=["return_1d"])
    rows = []
    for tkr, r in df.iterrows():
        for tag in _tags(r.get("tags")):
            row = {"yf_ticker": tkr, "tag": tag, "axis": TAG_AXIS.get(tag, "Other")}
            for c in ret_cols:
                if c in df.columns:
                    row[c] = r.get(c)
            rows.append(row)
    return pd.DataFrame(rows)


# =========================================================================== #
# MARKET-SIDE
# =========================================================================== #
def compute_tag_tilts(asset_table: pd.DataFrame, min_n: int = 3) -> pd.DataFrame:
    """Per tag: mean 1d/1w return of funds carrying it MINUS the universe mean
    (the tilt). Demeaning removes the market so correlated tags stay readable.
    Returns tags with n>=min_n, sorted by |tilt_1d|, with an 'axis' column.
    NEVER sum these across axes - each is an independent excess mean."""
    long = explode_tags(asset_table)
    if long.empty:
        return pd.DataFrame()
    uni_1d = long.drop_duplicates("yf_ticker")["return_1d"].mean()
    uni_1w = long.drop_duplicates("yf_ticker")["return_1w"].mean() \
        if "return_1w" in long else np.nan
    g = long.groupby("tag")
    out = g.agg(n=("return_1d", "size"),
                mean_1d=("return_1d", "mean"),
                mean_1w=("return_1w", "mean") if "return_1w" in long
                else ("return_1d", "mean")).reset_index()
    out = out[out["n"] >= min_n].copy()
    out["axis"] = out["tag"].map(lambda t: TAG_AXIS.get(t, "Other"))
    out["tilt_1d"] = out["mean_1d"] - uni_1d
    out["tilt_1w"] = out["mean_1w"] - uni_1w
    return out.sort_values("tilt_1d", key=lambda s: s.abs(),
                           ascending=False).reset_index(drop=True)


def compute_style_spreads(asset_table: pd.DataFrame, min_n: int = 3) -> pd.DataFrame:
    """Long-short style/region spreads from the ETF cross-section (equal-weight
    mean of tagged funds). Value-Growth, Defensive-Cyclical, EM-Developed, plus
    excess (vs universe) for Quality/Momentum/Low-Vol/Dividend."""
    long = explode_tags(asset_table)
    if long.empty:
        return pd.DataFrame()
    uni = long.drop_duplicates("yf_ticker")["return_1d"].mean()

    def tag_mean(tag):
        s = long[long["tag"] == tag]["return_1d"]
        return (s.mean(), len(s))

    def cyc_mean():
        # cyclical proxy = funds tagged with any cyclical sector
        sub = long[long["tag"].isin(CYCLICAL_SECTORS)].drop_duplicates("yf_ticker")
        return (sub["return_1d"].mean(), len(sub))

    rows = []
    for lng, sht, label in STYLE_PAIRS + REGION_PAIRS:
        lm, ln = tag_mean(lng)
        if sht == "Cyclical":
            sm, sn = cyc_mean()
        elif sht is None:
            sm, sn = uni, 999
        else:
            sm, sn = tag_mean(sht)
        if ln >= min_n and (sht is None or sn >= min_n) and pd.notna(lm) and pd.notna(sm):
            rows.append({"spread": label, "value": lm - sm,
                         "long_n": ln, "short_n": (None if sht is None else sn)})
    return (pd.DataFrame(rows).sort_values("value", key=lambda s: s.abs(),
            ascending=False).reset_index(drop=True) if rows else pd.DataFrame())


def compute_dispersion(asset_table: pd.DataFrame) -> dict:
    """Cross-sectional dispersion of 1d returns across funds (macro vs stock-
    picker's day). Beta-adjust by demeaning first."""
    r = asset_table.dropna(subset=["return_1d"])["return_1d"]
    if len(r) < 10:
        return {"dispersion": np.nan, "n": len(r), "regime": "insufficient data"}
    demeaned = r - r.mean()
    disp = float(demeaned.std())
    return {"dispersion": disp, "n": int(len(r)), "mean": float(r.mean())}


def compute_breadth(asset_table: pd.DataFrame) -> dict:
    """How broad was the tape: share of funds (and of tag-groups) positive."""
    r = asset_table.dropna(subset=["return_1d"])["return_1d"]
    n = len(r)
    if n == 0:
        return {"n": 0}
    tilts = compute_tag_tilts(asset_table)
    n_grp = len(tilts)
    grp_up = int((tilts["mean_1d"] > 0).sum()) if n_grp else 0
    return {
        "n": n, "n_up": int((r > 0).sum()),
        "pct_up": round(100.0 * (r > 0).sum() / n, 0),
        "n_groups": n_grp, "groups_up": grp_up,
        "pct_groups_up": round(100.0 * grp_up / n_grp, 0) if n_grp else np.nan,
        "mean": round(float(r.mean()), 3), "median": round(float(r.median()), 3),
    }


def compute_region_sector_grid(asset_table: pd.DataFrame,
                               min_n: int = 5) -> pd.DataFrame:
    """Mean 1d return for each (Region, Sector) cell with n>=min_n."""
    long = explode_tags(asset_table)
    if long.empty:
        return pd.DataFrame()
    regions = long[long["axis"] == "Region"][["yf_ticker", "tag", "return_1d"]]
    sectors = long[long["axis"] == "Sector"][["yf_ticker", "tag"]]
    m = regions.merge(sectors, on="yf_ticker", suffixes=("_region", "_sector"))
    if m.empty:
        return pd.DataFrame()
    g = m.groupby(["tag_region", "tag_sector"]).agg(
        n=("return_1d", "size"), ret_1d=("return_1d", "mean")).reset_index()
    g = g[g["n"] >= min_n]
    return g.sort_values("ret_1d", key=lambda s: s.abs(),
                         ascending=False).reset_index(drop=True)


def noise_gate(index_move_pct: float, vix_level: float,
               breadth: dict, dispersion: dict) -> dict:
    """Rule of 16: expected daily move ~ VIX/16. Classify the day's magnitude
    and flag a likely-noise day (sub-1sigma + mixed breadth)."""
    if vix_level is None or not np.isfinite(vix_level) or vix_level <= 0:
        return {"expected_move": None, "sigmas": None, "verdict": "unknown"}
    expected = vix_level / 16.0
    sig = abs(index_move_pct) / expected if expected else np.nan
    pct_up = breadth.get("pct_up", 50)
    mixed = 40 <= pct_up <= 60
    if sig < 1.0 and mixed:
        verdict = "noise"
    elif sig >= 2.0:
        verdict = "large (>2 sigma)"
    elif sig >= 1.0:
        verdict = "notable (1-2 sigma)"
    else:
        verdict = "quiet (<1 sigma)"
    return {"expected_move": round(expected, 2), "sigmas": round(sig, 2),
            "vix": round(vix_level, 1), "verdict": verdict}


# =========================================================================== #
# PORTFOLIO-SIDE  (vs a 60/40 ACWI/TLT benchmark)
# =========================================================================== #
def _weighted_tag_weights(items) -> dict:
    """items: iterable of (weight, [tags]). Returns {tag: summed weight}."""
    out = {}
    for w, tags in items:
        if not np.isfinite(w):
            continue
        for t in tags:
            out[t] = out.get(t, 0.0) + w
    return out


def benchmark_tag_weights(bench_specs) -> dict:
    """bench_specs: [(weight_frac, [tags]), ...] e.g. [(0.6, acwi_tags),
    (0.4, tlt_tags)]. Returns {tag: benchmark weight} (fractions, 0..1)."""
    return _weighted_tag_weights(bench_specs)


def compute_portfolio_tag_tilts(positions: pd.DataFrame, tag_map: dict,
                                bench_weights: dict, use_gross: bool = True) -> pd.DataFrame:
    """Book tag weight (gross) minus benchmark tag weight, per tag+axis.
    positions: index=symbol, column 'weight' (signed, vs gross). tag_map:
    {symbol: 'a, b, c'}. Returns tilts sorted by |tilt|, with axis."""
    items = []
    for sym, row in positions.iterrows():
        w = row.get("weight")
        if w is None or not np.isfinite(w):
            continue
        w = abs(w) if use_gross else w
        items.append((w, _tags(tag_map.get(sym, ""))))
    book = _weighted_tag_weights(items)
    total = sum(abs(v) for v in book.values()) or 1.0
    # normalize book tag weights to fractions of gross-tag-mass per convention:
    # keep raw book weights (already fractions of gross since 'weight' is vs gross)
    tags = sorted(set(book) | set(bench_weights))
    rows = []
    for t in tags:
        bw = book.get(t, 0.0)
        mw = bench_weights.get(t, 0.0)
        rows.append({"tag": t, "axis": TAG_AXIS.get(t, "Other"),
                     "book_wt": bw, "bench_wt": mw, "tilt": bw - mw})
    return pd.DataFrame(rows).sort_values("tilt", key=lambda s: s.abs(),
                                          ascending=False).reset_index(drop=True)


def compute_tag_bridge(port_tilts: pd.DataFrame,
                       market_tilts: pd.DataFrame, top: int = 8) -> pd.DataFrame:
    """For the book's biggest tilts, is that tag WITH or AGAINST today's tape?
    Joins portfolio tilt sign to the market tag's 1d tilt."""
    if port_tilts.empty:
        return pd.DataFrame()
    mkt = market_tilts.set_index("tag")["tilt_1d"].to_dict() if not market_tilts.empty else {}
    pt = port_tilts[port_tilts["tilt"].abs() > 0.005].head(top).copy()
    rows = []
    for _, r in pt.iterrows():
        mkt_ret = mkt.get(r["tag"])
        if mkt_ret is None or not np.isfinite(mkt_ret):
            stance = "no market read"
        else:
            # overweight a tag that outperformed -> WITH; underweight a winner -> AGAINST
            with_tape = (r["tilt"] > 0) == (mkt_ret > 0)
            stance = "WITH" if with_tape else "AGAINST"
        rows.append({"tag": r["tag"], "axis": r["axis"],
                     "tilt_pp": round(100 * r["tilt"], 1),
                     "mkt_tilt_1d": (round(mkt_ret, 2) if mkt_ret is not None
                                     and np.isfinite(mkt_ret) else None),
                     "stance": stance})
    return pd.DataFrame(rows)


# factor (regression) -> the tag it should show up as, for the
# exposure-vs-realized-beta reconciliation
FACTOR_TAG = {
    "SPX": "US", "Russell2000": "Small-Cap", "Nasdaq100": "Tech",
    "Value": "Value", "Growth": "Growth", "EAFE": "Developed", "EM": "EM",
    "HY_Credit": "HY Credit", "Treasuries": "Treasury", "TIPS": "Treasury",
    "Commodities": "Commodity", "Agriculture": "Commodity", "Crypto": "Alternative",
    "REIT_US": "Real Estate", "REIT_Global": "Real Estate",
}
# EM-ish region tags for the dispersion decomposition
EM_TAGS = {"EM", "China", "India"}


def compute_tag_pnl(positions: pd.DataFrame, tag_map: dict,
                    axes=("Region", "Sector", "Style")) -> pd.DataFrame:
    """Per-axis P&L attribution. Each position's contribution_bps is split
    EQUALLY among the tags it carries WITHIN an axis, so within any one axis the
    tag contributions sum to the day's attributed P&L (no double-counting across
    a position's multiple same-axis tags). Never summed ACROSS axes."""
    if "contribution_bps" not in positions.columns:
        return pd.DataFrame()
    rows = []
    for sym, r in positions.iterrows():
        c = r.get("contribution_bps")
        if c is None or not np.isfinite(c):
            continue
        by_axis = {}
        for t in _tags(tag_map.get(sym, "")):
            ax = TAG_AXIS.get(t, "Other")
            by_axis.setdefault(ax, []).append(t)
        for ax, ts in by_axis.items():
            if ax not in axes:
                continue
            share = c / len(ts)          # split equally within the axis
            for t in ts:
                rows.append({"axis": ax, "tag": t,
                             "contribution_bps": share, "sym": sym})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    g = df.groupby(["axis", "tag"]).agg(
        contribution_bps=("contribution_bps", "sum"),
        n=("sym", "nunique")).reset_index()
    return g.sort_values(["axis", "contribution_bps"],
                         ascending=[True, False]).reset_index(drop=True)


def compute_exposure_vs_beta(factor_exposure: pd.DataFrame,
                             tag_weights: dict) -> pd.DataFrame:
    """Reconcile the book's INTENDED exposure (tag gross weight) against its
    REALIZED exposure (regression beta to the matching factor). Flags hidden
    betas (moves with a factor it does not tag as holding) and inert tags (tags
    a theme but shows little beta to it)."""
    if factor_exposure is None or factor_exposure.empty:
        return pd.DataFrame()
    rows = []
    for _, r in factor_exposure.iterrows():
        tag = FACTOR_TAG.get(r["factor"])
        if not tag:
            continue
        beta = r.get("portfolio_beta")
        w = tag_weights.get(tag, 0.0)
        if beta is not None and np.isfinite(beta) and abs(beta) >= 0.4 and w < 0.05:
            note = "beta without tag (hidden co-movement)"
        elif w >= 0.15 and (beta is None or not np.isfinite(beta) or abs(beta) < 0.2):
            note = "tag without beta (inert)"
        else:
            note = "consistent"
        rows.append({"factor": r["factor"], "tag": tag,
                     "beta": beta, "tag_wt": w, "note": note})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # keep the interesting rows first (divergences), then by |beta|
    df["_rank"] = (df["note"] != "consistent").astype(int)
    return df.sort_values(["_rank", "beta"], key=lambda s: s if s.name == "_rank"
                          else s.abs(), ascending=[False, False]).drop(
                          columns="_rank").reset_index(drop=True)


def compute_em_dispersion(asset_table: pd.DataFrame, min_n: int = 6) -> dict:
    """Decompose today's EM cross-section: was the spread driven by COUNTRY
    selection or by STYLE? Reports eta-squared (between-group SS / total SS) for
    the Region grouping vs the Style grouping over EM-tagged funds."""
    long = explode_tags(asset_table)
    if long.empty:
        return {"n": 0, "verdict": "insufficient data"}
    em_tickers = sorted(set(long[long["tag"].isin(EM_TAGS)]["yf_ticker"]))
    if len(em_tickers) < min_n:
        return {"n": len(em_tickers), "verdict": "insufficient data"}
    sub = long[long["yf_ticker"].isin(em_tickers)]
    rets = sub.drop_duplicates("yf_ticker").set_index("yf_ticker")["return_1d"]
    allv = rets.values.astype(float)
    grand = allv.mean()
    ss_tot = float(((allv - grand) ** 2).sum())

    def eta2(axis):
        grp = {}
        for tkr in em_tickers:
            tags_ax = sub[(sub["yf_ticker"] == tkr)
                          & (sub["axis"] == axis)]["tag"].tolist()
            if tags_ax:
                grp.setdefault(tags_ax[0], []).append(float(rets[tkr]))
        if len(grp) < 2 or ss_tot <= 0:
            return None
        ss_bet = sum(len(v) * (np.mean(v) - grand) ** 2 for v in grp.values())
        return round(float(ss_bet / ss_tot), 2)

    er, es = eta2("Region"), eta2("Style")
    if er is None and es is None:
        verdict = "no groupings"
    elif (er or 0) > (es or 0):
        verdict = "country-driven"
    elif (es or 0) > (er or 0):
        verdict = "style-driven"
    else:
        verdict = "mixed"
    return {"n": len(em_tickers), "eta2_region": er, "eta2_style": es,
            "dispersion": round(float(allv.std()), 2), "verdict": verdict}


# ---- asset-allocation classification (mutually-exclusive primary buckets) ----
_BOND_TAGS = {"Credit", "Treasury", "Municipal", "IG Credit", "HY Credit",
              "Investment Grade", "High Yield"}
_EM_TAGS = {"EM", "China", "India", "UAE"}
_DEV_INTL_TAGS = {"Japan", "Developed", "Canada", "Australia", "Europe",
                  "APAC", "International"}


def classify_holding(tags, is_cash=False):
    """Assign a holding to ONE asset class and ONE region for the allocation
    report (buckets are mutually exclusive so they sum to 100%). Tag-driven;
    genuinely mis-tagged funds are corrected at source in tags.py."""
    if is_cash:
        return ("Cash", "Cash")
    ts = set(tags)
    if "Equity" in ts:
        cls = "Equities"
    elif ts & _BOND_TAGS:
        cls = "Bonds"
    elif "Commodity" in ts:
        cls = "Commodities"
    elif ts & {"Multi-Asset", "Alternative"}:
        cls = "Alternatives"
    else:
        cls = "Other"
    if ts & _EM_TAGS:
        reg = "EM"
    elif ts & {"US", "Domestic"}:
        reg = "US"
    elif ts & _DEV_INTL_TAGS:
        reg = "International"
    elif "Asia" in ts:                 # generic Asia (developed Asia is tagged
        reg = "EM"                     # Japan/Developed) -> EM-leaning
    elif "Global" in ts:
        reg = "Global"
    else:
        reg = "Unclassified"
    return (cls, reg)


def _vw(sub, col):
    """Bucket return = P&L direction over GROSS exposure: Σ(signed mtm × ret) /
    Σ|mtm|. Signed numerator keeps a short's sign correct; the gross denominator
    avoids the blow-up that a signed (net) denominator causes when longs and
    shorts nearly offset and one outlier then dominates a tiny net base."""
    s = sub.dropna(subset=[col])
    denom = s["mtm"].abs().sum()
    return float((s["mtm"] * s[col]).sum() / denom) if denom else np.nan


def compute_asset_allocation(positions: pd.DataFrame, tag_map: dict,
                             cash_value: float = 0.0) -> dict:
    """Household asset allocation. Returns:
        by_class          - Equities/Bonds/Commodities/Alternatives/Cash/Other,
                            weight % of total (incl. cash), value, 1d, YTD, n
        equity_by_region  - US/International/EM/Global for the equity sleeve
        unclassified      - equity tickers whose region could not be derived
    Weights are % of total net value; bucket returns are value-weighted."""
    rows = []
    for sym, r in positions.iterrows():
        mtm = r.get("market_value_mtm")
        if mtm is None or not np.isfinite(mtm):
            continue
        cls, reg = classify_holding(_tags(tag_map.get(sym, "")))
        rows.append({"sym": sym, "mtm": mtm, "ret_1d": r.get("return_1d"),
                     "ret_ytd": r.get("return_ytd"), "cls": cls, "reg": reg})
    if not rows:
        return {"by_class": pd.DataFrame(), "equity_by_region": pd.DataFrame(),
                "unclassified": [], "total_value": cash_value}
    df = pd.DataFrame(rows)
    total = float(df["mtm"].sum() + cash_value)

    class_rows = []
    for cls in ["Equities", "Bonds", "Commodities", "Alternatives", "Other"]:
        sub = df[df["cls"] == cls]
        if sub.empty:
            continue
        val = float(sub["mtm"].sum())
        class_rows.append({"bucket": cls,
                           "weight_pct": 100 * val / total if total else np.nan,
                           "value": val, "return_1d": _vw(sub, "ret_1d"),
                           "return_ytd": _vw(sub, "ret_ytd"), "n": len(sub)})
    if cash_value:
        class_rows.append({"bucket": "Cash",
                           "weight_pct": 100 * cash_value / total if total else np.nan,
                           "value": float(cash_value), "return_1d": 0.0,
                           "return_ytd": 0.0, "n": 0})
    by_class = pd.DataFrame(class_rows)

    eq = df[df["cls"] == "Equities"]
    eq_total = float(eq["mtm"].sum())
    reg_rows = []
    for reg in ["US", "International", "EM", "Global", "Unclassified"]:
        sub = eq[eq["reg"] == reg]
        if sub.empty:
            continue
        val = float(sub["mtm"].sum())
        reg_rows.append({"region": reg,
                         "weight_pct": 100 * val / eq_total if eq_total else np.nan,
                         "pct_of_total": 100 * val / total if total else np.nan,
                         "return_1d": _vw(sub, "ret_1d"),
                         "return_ytd": _vw(sub, "ret_ytd"), "n": len(sub)})
    equity_by_region = pd.DataFrame(reg_rows)
    return {"by_class": by_class, "equity_by_region": equity_by_region,
            "unclassified": sorted(eq[eq["reg"] == "Unclassified"]["sym"]),
            "total_value": total}


def build_tag_views(market: dict, portfolio: dict, tag_map: dict,
                    bench_specs, vix_level=None, index_move=None) -> dict:
    """Orchestrate all tier-3 views into one dict for prompt.py.
    bench_specs: [(weight_frac, [tags]), ...] e.g. [(0.6, acwi_tags), (0.4, tlt_tags)].
    """
    at = market["asset_table"]
    tilts = compute_tag_tilts(at)
    breadth = compute_breadth(at)
    disp = compute_dispersion(at)
    ptilts = compute_portfolio_tag_tilts(
        portfolio["positions"], tag_map, benchmark_tag_weights(bench_specs))
    book_tag_wt = dict(zip(ptilts["tag"], ptilts["book_wt"])) if not ptilts.empty else {}
    return {
        "market_tilts": tilts,
        "spreads": compute_style_spreads(at),
        "dispersion": disp,
        "breadth": breadth,
        "grid": compute_region_sector_grid(at),
        "em_dispersion": compute_em_dispersion(at),
        "noise_gate": noise_gate(index_move if index_move is not None else 0.0,
                                 vix_level, breadth, disp),
        "port_tilts": ptilts,
        "bridge": compute_tag_bridge(ptilts, tilts),
        "tag_pnl": compute_tag_pnl(portfolio["positions"], tag_map),
        "exposure_vs_beta": compute_exposure_vs_beta(
            portfolio.get("factor_exposure"), book_tag_wt),
        "concentration": compute_tag_concentration(portfolio["positions"], tag_map),
    }


def compute_tag_concentration(positions: pd.DataFrame, tag_map: dict) -> dict:
    """Effective number of positions (1/HHI on gross weights) + effective number
    of tags per axis + the largest single tag exposure."""
    w = positions["weight"].abs().dropna()
    w = w / w.sum() if w.sum() > 0 else w
    hhi_pos = float((w ** 2).sum()) if len(w) else np.nan
    # tag weights by axis
    axis_tag_w = {}
    for sym, row in positions.iterrows():
        ww = row.get("weight")
        if ww is None or not np.isfinite(ww):
            continue
        for t in _tags(tag_map.get(sym, "")):
            ax = TAG_AXIS.get(t, "Other")
            axis_tag_w.setdefault(ax, {})
            axis_tag_w[ax][t] = axis_tag_w[ax].get(t, 0.0) + abs(ww)
    eff_tags = {}
    for ax, d in axis_tag_w.items():
        tot = sum(d.values()) or 1.0
        fr = np.array([v / tot for v in d.values()])
        eff_tags[ax] = round(1.0 / float((fr ** 2).sum()), 1) if len(fr) else np.nan
    # biggest single tag (any axis) by gross weight
    flat = {t: v for d in axis_tag_w.values() for t, v in d.items()}
    top_tag = max(flat, key=flat.get) if flat else None
    return {
        "eff_positions": round(1.0 / hhi_pos, 1) if hhi_pos and np.isfinite(hhi_pos) else np.nan,
        "eff_tags_by_axis": eff_tags,
        "top_tag": top_tag,
        "top_tag_gross_pct": round(100 * flat[top_tag], 0) if top_tag else np.nan,
    }
