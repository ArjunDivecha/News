#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: analytics.py
=============================================================================

INPUT FILES:
    (none directly - operates on DataFrames passed in by main.py:
     wide price matrix from db.load_prices, universe from data.load_universe,
     holdings from holdings.get_holdings)

OUTPUT FILES:
    (none directly - returns dicts/DataFrames consumed by prompt.py and
     persisted to report.db by main.py)

VERSION: 1.0
LAST UPDATED: 2026-06-09
AUTHOR: Arjun Divecha

DESCRIPTION:
    ALL financial math for the unified report lives here, in pure functions
    over pandas objects, so every formula is unit-testable in isolation.

    Conventions (uniform everywhere):
      - Returns are in PERCENT (1.23 means +1.23%). Contributions are in
        BASIS POINTS. Conversions happen exactly once, here.
      - Short positions: market_value and weight are NEGATIVE; a price
        DROP therefore produces a POSITIVE contribution automatically via
        contribution = weight x return. No sign flipping anywhere else.
      - Weights are vs GROSS exposure (sum of |market_value|, cash excluded).
      - Missing prices NEVER poison aggregates: positions without prices
        are excluded from return/contribution math, included in exposure,
        and listed explicitly in data_quality.

DEPENDENCIES:
    - pandas, numpy

USAGE:
    from analytics import compute_market, compute_portfolio
=============================================================================
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import FACTORS, SETTINGS, ACCOUNT_NAMES, CASH_EQUIVALENTS

TRADING_DAYS = 252


# ===========================================================================
# Return primitives
# ===========================================================================

def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Day-over-day % returns from a wide price matrix (index=date)."""
    return prices.sort_index().pct_change(fill_method=None) * 100.0


def return_over(prices: pd.DataFrame, n_days: int) -> pd.Series:
    """% return over the last n trading days (per column)."""
    p = prices.sort_index()
    if len(p) <= n_days:
        return pd.Series(np.nan, index=p.columns)
    end = p.iloc[-1]
    start = p.iloc[-1 - n_days]
    return (end / start - 1.0) * 100.0


def ytd_return(prices: pd.DataFrame, asof: Optional[str] = None) -> pd.Series:
    """
    % return from the LAST trading day of the prior year to asof.
    Tickers with no prior-year close (new listings) return NaN - they are
    reported as missing rather than given a fake partial-year number.
    """
    p = prices.sort_index()
    if p.empty:
        return pd.Series(dtype=float)
    asof = asof or str(p.index[-1])
    year = pd.Timestamp(asof).year
    prior = p.loc[p.index < f"{year}-01-01"]
    if prior.empty:
        return pd.Series(np.nan, index=p.columns)
    base = prior.iloc[-1]
    end = p.loc[p.index <= asof].iloc[-1]
    out = (end / base - 1.0) * 100.0
    out[base.isna()] = np.nan
    return out


def annualized_vol(returns: pd.DataFrame, window: int = None) -> pd.Series:
    """Annualized volatility (%) from daily % returns over a window."""
    window = window or SETTINGS["vol_window"]
    r = returns.tail(window)
    counts = r.notna().sum()
    vol = r.std() * np.sqrt(TRADING_DAYS)
    vol[counts < window // 2] = np.nan        # not enough data to trust
    return vol


def return_percentile(returns: pd.DataFrame, window: int = None) -> pd.Series:
    """
    Percentile rank (0-100) of the latest daily return within each ticker's
    own trailing window. 100 = best day of the window. Rounded to an integer
    (a 60-day window has ~1.7% granularity, so decimals are false precision)
    and masked to NaN when there are too few observations to trust the rank
    (same rigor annualized_vol already applies).
    """
    window = window or SETTINGS["percentile_window"]
    r = returns.tail(window)
    latest = r.iloc[-1]
    counts = r.notna().sum()
    rank = (r.lt(latest, axis=1).sum() / counts.clip(lower=1)) * 100.0
    rank[latest.isna()] = np.nan
    rank[counts < window // 2] = np.nan       # not enough data to trust
    return rank.round(0)


def beta_vs(returns: pd.DataFrame, factor_returns: pd.Series,
            window: int = None) -> pd.Series:
    """
    OLS beta of each column vs a factor return series over a window.
    Requires SETTINGS['min_beta_obs'] paired observations, else NaN.
    """
    window = window or SETTINGS["beta_window"]
    r = returns.tail(window)
    f = factor_returns.reindex(r.index)
    f_dm = f - f.mean()
    var_f = (f_dm ** 2).sum()
    betas = {}
    for col in r.columns:
        pair = pd.concat([r[col], f], axis=1).dropna()
        if len(pair) < SETTINGS["min_beta_obs"] or var_f == 0:
            betas[col] = np.nan
            continue
        x = pair.iloc[:, 1] - pair.iloc[:, 1].mean()
        y = pair.iloc[:, 0] - pair.iloc[:, 0].mean()
        denom = (x ** 2).sum()
        betas[col] = (x * y).sum() / denom if denom > 0 else np.nan
    return pd.Series(betas)


def correlation_matrix(returns: pd.DataFrame, tickers: list,
                       window: int = None) -> pd.DataFrame:
    """Pairwise correlation of daily returns over a trailing window."""
    window = window or SETTINGS["beta_window"]
    cols = [t for t in tickers if t in returns.columns]
    return returns[cols].tail(window).corr()


# ===========================================================================
# Market-level analytics
# ===========================================================================

def compute_market(prices: pd.DataFrame, universe: pd.DataFrame,
                   asof: Optional[str] = None) -> dict:
    """
    Full market picture for the report.

    Returns dict with:
        asof, asset_table (per-asset metrics), tier1_summary, tier2_summary,
        factor_table, factor_corr, movers, streaks, data_quality
    """
    prices = prices.sort_index()
    asof = asof or str(prices.index[-1])
    prices = prices.loc[prices.index <= asof]

    # Market view covers UNIVERSE tickers only (the price matrix may also
    # contain portfolio-only symbols; those belong to compute_portfolio)
    uni_tickers = [t for t in universe["yf_ticker"] if t in prices.columns]
    # keep SPY available for beta even if it were not in the universe
    keep = sorted(set(uni_tickers) | ({"SPY"} & set(prices.columns)))
    prices = prices[keep]
    rets = daily_returns(prices)

    # ---- per-asset table ----
    asset = pd.DataFrame(index=pd.Index(uni_tickers, name="yf_ticker"))
    asset["return_1d"] = rets.iloc[-1]
    asset["return_1w"] = return_over(prices, 5)
    asset["return_1m"] = return_over(prices, 21)
    asset["return_ytd"] = ytd_return(prices, asof)
    asset["vol_60d"] = annualized_vol(rets)
    asset["pctile_1d"] = return_percentile(rets)

    spy = rets.get("SPY")
    if spy is not None:
        asset["beta_spx"] = beta_vs(rets, spy)
    else:
        asset["beta_spx"] = np.nan

    meta = universe.set_index("yf_ticker")[["name", "tier1", "tier2", "tags",
                                            "is_factor", "factor_name"]]
    asset = asset.join(meta, how="left")

    # ---- tier aggregates (equal-weighted means over assets WITH data) ----
    have_ret = asset.dropna(subset=["return_1d"])
    tier1 = (have_ret.groupby("tier1")
             .agg(n=("return_1d", "size"),
                  ret_1d=("return_1d", "mean"),
                  ret_1w=("return_1w", "mean"),
                  ret_1m=("return_1m", "mean"),
                  ret_ytd=("return_ytd", "mean"))
             .sort_values("ret_1d", ascending=False))
    tier2 = (have_ret.groupby(["tier1", "tier2"])
             .agg(n=("return_1d", "size"),
                  ret_1d=("return_1d", "mean"),
                  ret_1w=("return_1w", "mean"),
                  ret_ytd=("return_ytd", "mean"))
             .sort_values("ret_1d", ascending=False))

    # ---- factors ----
    factor_tickers = [t for t in FACTORS.values() if t in asset.index]
    factor_table = asset.loc[factor_tickers,
                             ["factor_name", "return_1d", "return_1w",
                              "return_1m", "return_ytd", "vol_60d"]].copy()
    factor_corr = correlation_matrix(rets, factor_tickers)

    # ---- movers (within assets that have data) ----
    movers_up = have_ret.nlargest(15, "return_1d")
    movers_down = have_ret.nsmallest(15, "return_1d")

    # ---- streaks: consecutive up/down days per asset ----
    streaks = {}
    recent = rets.tail(15)
    for col in have_ret.index:
        s = recent[col].dropna()
        if s.empty:
            continue
        sign = np.sign(s.iloc[-1])
        if sign == 0:
            continue
        count = 0
        for v in s.iloc[::-1]:
            if np.sign(v) == sign:
                count += 1
            else:
                break
        if count >= 4:
            streaks[col] = int(count * sign)
    streaks_df = (pd.Series(streaks, name="streak").to_frame()
                  .join(asset[["name", "return_1d"]])
                  .sort_values("streak"))

    # ---- data quality ----
    n_universe = len(universe)
    n_priced = int(asset["return_1d"].notna().sum())
    missing = sorted(set(universe["yf_ticker"]) - set(have_ret.index))
    data_quality = {
        "n_universe": n_universe,
        "n_priced_today": n_priced,
        "coverage_pct": round(100.0 * n_priced / max(n_universe, 1), 1),
        "missing_tickers": missing,
    }

    return {
        "asof": asof,
        "asset_table": asset,
        "tier1_summary": tier1,
        "tier2_summary": tier2,
        "factor_table": factor_table,
        "factor_corr": factor_corr,
        "movers_up": movers_up,
        "movers_down": movers_down,
        "streaks": streaks_df,
        "data_quality": data_quality,
    }


# ===========================================================================
# Portfolio analytics
# ===========================================================================

def aggregate_holdings(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse per-account broker rows into one row per symbol.
    Cash rows (symbol == 'CASH') are aggregated separately and returned
    with the flag is_cash=True. Quantities and market values are summed;
    avg_price is the |value|-weighted mean.
    """
    df = raw.copy()
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df["is_cash"] = df["symbol"].isin(CASH_EQUIVALENTS)

    def agg(group: pd.DataFrame) -> pd.Series:
        mv = group["market_value"].sum(min_count=1)
        qty = group["quantity"].sum()
        w = group["market_value"].abs()
        if w.notna().any() and w.sum() > 0:
            avg = (group["avg_price"] * w).sum() / w.sum()
        else:
            avg = group["avg_price"].mean()
        # If ANY constituent P&L is missing, mark the whole symbol missing so
        # it gets recomputed from prices rather than silently understated
        pnl = (group["open_pnl"].sum()
               if group["open_pnl"].notna().all() else np.nan)
        return pd.Series({
            "quantity": qty,
            "avg_price": avg,
            "market_value": mv,
            "open_pnl": pnl,
            "is_cash": bool(group["is_cash"].iloc[0]),
            "accounts": ", ".join(sorted(set(group["account"].astype(str)))),
        })

    out = df.groupby("symbol", sort=False).apply(agg, include_groups=False).reset_index()
    # Drop fully-closed positions that net to zero quantity AND zero value
    out = out[~((out["quantity"] == 0) & (out["market_value"].abs() < 1e-9))]
    return out


def compute_portfolio(holdings: pd.DataFrame, prices: pd.DataFrame,
                      asof: Optional[str] = None,
                      holdings_stale: bool = False) -> dict:
    """
    Portfolio analytics on aggregated holdings.

    Returns dict with:
        positions (per-symbol table), summary (totals/returns/alpha inputs),
        factor_exposure (beta-weighted), data_quality
    """
    prices = prices.sort_index()
    asof = asof or str(prices.index[-1])
    prices = prices.loc[prices.index <= asof]
    rets = daily_returns(prices)

    agg = aggregate_holdings(holdings)
    cash = agg[agg["is_cash"]]
    pos = agg[~agg["is_cash"]].copy().set_index("symbol")

    # --- price-based metrics where available ---
    last_close = prices.iloc[-1]
    pos["price"] = last_close.reindex(pos.index)
    pos["return_1d"] = rets.iloc[-1].reindex(pos.index)
    pos["return_ytd"] = ytd_return(prices, asof).reindex(pos.index)
    pos["has_price"] = pos["price"].notna()

    # Mark-to-market where we have a price; broker value otherwise
    pos["market_value_mtm"] = np.where(
        pos["has_price"], pos["quantity"] * pos["price"], pos["market_value"])

    # Open P&L: recompute from prices where the broker did not supply it
    # (IBKR positions come without P&L; avg_price is broker-reported cost)
    need_pnl = pos["open_pnl"].isna() & pos["has_price"] & pos["avg_price"].notna()
    pos.loc[need_pnl, "open_pnl"] = (
        (pos.loc[need_pnl, "price"] - pos.loc[need_pnl, "avg_price"])
        * pos.loc[need_pnl, "quantity"])

    # --- exposures (signed values; shorts are negative) ---
    long_value = pos.loc[pos["market_value_mtm"] > 0, "market_value_mtm"].sum()
    short_value = pos.loc[pos["market_value_mtm"] < 0, "market_value_mtm"].sum()
    gross = pos["market_value_mtm"].abs().sum()
    net = pos["market_value_mtm"].sum()
    cash_value = cash["market_value"].sum() if not cash.empty else 0.0

    # --- current weights vs gross (signed) - the posture NOW ---
    pos["weight"] = pos["market_value_mtm"] / gross if gross > 0 else np.nan

    # --- beginning-of-day weights - the correct base for contributions.
    # weight_bod x return_1d summed ties out EXACTLY to dollar P&L over
    # the PRICED book's BOD gross. Unpriced positions are fully excluded
    # from the return (consistent with how the report describes them),
    # not silently treated as 0%-return ballast in the denominator.
    prev_price = (prices.iloc[-2] if len(prices) >= 2
                  else pd.Series(dtype=float)).reindex(pos.index)
    pricable = pos["has_price"] & prev_price.notna()
    pos["value_bod"] = np.where(pricable, pos["quantity"] * prev_price, np.nan)
    gross_bod = pos["value_bod"].abs().sum()
    pos["weight_bod"] = pos["value_bod"] / gross_bod if gross_bod > 0 else np.nan

    # --- contributions in bps: BOD weight x return; NaN returns excluded ---
    pos["contribution_bps"] = pos["weight_bod"] * pos["return_1d"] * 100.0

    priced = pos[pos["has_price"]]
    port_ret_1d = priced["contribution_bps"].sum() / 100.0   # back to %
    priced_w_ytd = priced.dropna(subset=["return_ytd"])
    # Approximation: assumes current holdings were held all year
    port_ret_ytd = (priced_w_ytd["weight"] * priced_w_ytd["return_ytd"]).sum()

    # --- factor-implied expected return & naive alpha ---
    expected_1d = np.nan
    alpha_1d = np.nan
    if "SPY" in rets.columns:
        betas = beta_vs(rets, rets["SPY"])
        pos["beta_spx"] = betas.reindex(pos.index)
        spy_1d = rets["SPY"].iloc[-1]
        with_beta = pos.dropna(subset=["beta_spx", "weight_bod"])
        port_beta = (with_beta["weight_bod"] * with_beta["beta_spx"]).sum()
        expected_1d = port_beta * spy_1d
        alpha_1d = port_ret_1d - expected_1d
    else:
        pos["beta_spx"] = np.nan
        port_beta = np.nan

    # --- factor exposure table (weight-summed beta to each factor) ---
    factor_rows = []
    for fname, fticker in FACTORS.items():
        if fticker not in rets.columns:
            continue
        fbetas = beta_vs(rets[priced.index.tolist()], rets[fticker])
        wb = (priced["weight"] * fbetas.reindex(priced.index)).sum()
        factor_rows.append({"factor": fname, "etf": fticker,
                            "portfolio_beta": wb,
                            "factor_return_1d": rets[fticker].iloc[-1]})
    factor_exposure = pd.DataFrame(factor_rows)

    unpriced = pos[~pos["has_price"]]
    summary = {
        "asof": asof,
        "total_value": net + cash_value,
        "gross_exposure": gross,
        "net_exposure": net,
        "long_value": long_value,
        "short_value": short_value,
        "cash_value": cash_value,
        "n_positions": int(len(pos)),
        "n_long": int((pos["market_value_mtm"] > 0).sum()),
        "n_short": int((pos["market_value_mtm"] < 0).sum()),
        "return_1d": port_ret_1d,
        "return_ytd": port_ret_ytd,
        "portfolio_beta": port_beta,
        "expected_return_1d": expected_1d,
        "alpha_1d": alpha_1d,
        "total_open_pnl": pos["open_pnl"].sum(),
        "holdings_stale": int(holdings_stale),
    }

    data_quality = {
        "n_positions": len(pos),
        "n_priced": int(pos["has_price"].sum()),
        "unpriced_symbols": unpriced.index.tolist(),
        "unpriced_value": unpriced["market_value_mtm"].sum(),
    }

    return {
        "positions": pos.sort_values("market_value_mtm", ascending=False),
        "summary": summary,
        "factor_exposure": factor_exposure,
        "data_quality": data_quality,
    }


# ===========================================================================
# Sub-portfolio analytics (per-account returns)
# ===========================================================================

def compute_subportfolios(raw_holdings: pd.DataFrame, prices: pd.DataFrame,
                          asof: Optional[str] = None,
                          gmo_holdings: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Compute daily and YTD return for each sub-portfolio (broker × account),
    plus GMO if provided.

    Returns a DataFrame with one row per sub-portfolio:
        name, broker, account, n_positions, total_value, cash,
        return_1d (%), return_ytd (%), long_value, short_value
    """
    prices = prices.sort_index()
    asof = asof or str(prices.index[-1])
    prices = prices.loc[prices.index <= asof]
    rets = daily_returns(prices)
    ytd = ytd_return(prices, asof)

    last_close = prices.iloc[-1]
    prev_price = (prices.iloc[-2] if len(prices) >= 2
                  else pd.Series(dtype=float))

    results = []

    groups = raw_holdings.groupby(["broker", "account"], sort=False)
    for (broker, acct), grp in groups:
        label = ACCOUNT_NAMES.get((broker, str(acct)), f"{broker} {acct}")
        res = _subportfolio_returns(grp, rets, ytd, prev_price, last_close,
                                    label, broker, str(acct))
        results.append(res)

    if gmo_holdings is not None and not gmo_holdings.empty:
        res = _subportfolio_returns(gmo_holdings, rets, ytd, prev_price, last_close,
                                    "GMO", "GMO", "GMO")
        results.append(res)

    out = pd.DataFrame(results)

    # Add a total row (value-weighted returns)
    if not out.empty:
        total_val = out["total_value"].sum()
        total_long = out["long_value"].sum()
        total_short = out["short_value"].sum()
        total_cash = out["cash"].sum()
        total_pos = int(out["n_positions"].sum())
        # Value-weight the daily and YTD returns across sub-portfolios
        valid_1d = out.dropna(subset=["return_1d"])
        if not valid_1d.empty and valid_1d["total_value"].abs().sum() > 0:
            w = valid_1d["total_value"] / valid_1d["total_value"].sum()
            total_1d = (w * valid_1d["return_1d"]).sum()
        else:
            total_1d = np.nan
        valid_ytd = out.dropna(subset=["return_ytd"])
        if not valid_ytd.empty and valid_ytd["total_value"].abs().sum() > 0:
            w = valid_ytd["total_value"] / valid_ytd["total_value"].sum()
            total_ytd = (w * valid_ytd["return_ytd"]).sum()
        else:
            total_ytd = np.nan
        total_row = pd.DataFrame([{
            "name": "TOTAL", "broker": "", "account": "",
            "n_positions": total_pos, "total_value": total_val,
            "cash": total_cash, "long_value": total_long,
            "short_value": total_short,
            "return_1d": total_1d, "return_ytd": total_ytd,
        }])
        out = pd.concat([out, total_row], ignore_index=True)

    return out


def _subportfolio_returns(positions: pd.DataFrame, rets: pd.DataFrame,
                          ytd: pd.Series, prev_price: pd.Series,
                          last_close: pd.Series,
                          label: str, broker: str, account: str) -> dict:
    """Compute return metrics for a single sub-portfolio."""
    df = positions.copy()
    df["symbol"] = df["symbol"].astype(str).str.strip()
    is_cash = df["symbol"].isin(CASH_EQUIVALENTS)
    cash_val = df.loc[is_cash, "market_value"].sum()
    pos = df[~is_cash].copy()

    if pos.empty:
        return {
            "name": label, "broker": broker, "account": account,
            "n_positions": 0, "total_value": cash_val, "cash": cash_val,
            "long_value": 0.0, "short_value": 0.0,
            "return_1d": 0.0, "return_ytd": 0.0,
        }

    last_ret = rets.iloc[-1] if not rets.empty else pd.Series(dtype=float)
    pos["return_1d"] = pos["symbol"].map(last_ret)
    pos["return_ytd"] = pos["symbol"].map(ytd)

    # Mark-to-market: use price × quantity where available, else broker value
    pos["price"] = pos["symbol"].map(last_close)
    pos["mtm"] = np.where(pos["price"].notna(),
                          pos["quantity"] * pos["price"],
                          pos["market_value"])
    mv = pos["mtm"]
    long_val = mv[mv > 0].sum()
    short_val = mv[mv < 0].sum()
    gross = mv.abs().sum()

    prev = pos["symbol"].map(prev_price)
    pos["value_bod"] = np.where(prev.notna() & pos["return_1d"].notna(),
                                 pos["quantity"] * prev, np.nan)
    gross_bod = pos["value_bod"].abs().sum()

    if gross_bod > 0:
        pos["weight_bod"] = pos["value_bod"] / gross_bod
        pos["contrib"] = pos["weight_bod"] * pos["return_1d"] * 100.0
        ret_1d = pos["contrib"].sum() / 100.0
    else:
        ret_1d = np.nan

    if gross > 0:
        pos["weight"] = mv / gross
        priced_ytd = pos.dropna(subset=["return_ytd"])
        ret_ytd = (priced_ytd["weight"] * priced_ytd["return_ytd"]).sum()
    else:
        ret_ytd = np.nan

    return {
        "name": label, "broker": broker, "account": account,
        "n_positions": len(pos),
        "total_value": long_val + short_val + cash_val,
        "cash": cash_val,
        "long_value": long_val,
        "short_value": short_val,
        "return_1d": ret_1d,
        "return_ytd": ret_ytd,
    }


# ===========================================================================
# The bridge: market <-> portfolio connection
# ===========================================================================

def compute_bridge(market: dict, portfolio: dict,
                   universe: Optional[pd.DataFrame] = None) -> dict:
    """
    Connect today's market action to the portfolio:
      - per-position attribution vs its tier-2 peer group
      - which market themes the portfolio is exposed to / missing
      - portfolio breadth (how many positions rose / beat their peer group)

    tier-2 labels are taken from the priced asset table first, then backfilled
    from the FULL universe (so a held name that is in the universe but happened
    to be unpriced today still gets its peer group). Held names that are not in
    the universe at all are labelled "Portfolio-Specific" - no synthetic peer
    group is invented for them.
    """
    pos = portfolio["positions"]
    asset = market["asset_table"]

    # Full-universe tier-2 lookup (covers universe names missing from the
    # priced asset table). asset_table already carries tier2 for priced names.
    uni_tier2 = {}
    if universe is not None and not universe.empty \
            and "yf_ticker" in universe and "tier2" in universe:
        uni_tier2 = (universe.dropna(subset=["tier2"])
                     .set_index("yf_ticker")["tier2"].to_dict())

    rows = []
    for sym, p in pos.iterrows():
        if not p["has_price"] or pd.isna(p["return_1d"]):
            continue
        tier2 = asset["tier2"].get(sym)
        if not isinstance(tier2, str):
            tier2 = uni_tier2.get(sym)
        peer_ret = np.nan
        if isinstance(tier2, str):
            peers = asset[asset["tier2"] == tier2]["return_1d"].dropna()
            if len(peers) >= 3:
                peer_ret = peers.mean()
        else:
            tier2 = "Portfolio-Specific"      # honest label, no fake peers
        rows.append({
            "symbol": sym,
            "weight": p["weight"],
            "return_1d": p["return_1d"],
            "contribution_bps": p["contribution_bps"],
            "tier2": tier2,
            "peer_return_1d": peer_ret,
            "vs_peers": (p["return_1d"] - peer_ret) if pd.notna(peer_ret) else np.nan,
        })
    attribution = pd.DataFrame(rows)
    if not attribution.empty:
        attribution = attribution.sort_values("contribution_bps")

    # ---- breadth: how broad was the book's day, and stock-selection hit-rate
    if not attribution.empty:
        n_priced = len(attribution)
        n_up = int((attribution["return_1d"] > 0).sum())
        with_peer = attribution.dropna(subset=["vs_peers"])
        n_with_peer = int(len(with_peer))
        n_beating = int((with_peer["vs_peers"] > 0).sum())
        breadth = {
            "n_priced": n_priced,
            "n_up": n_up,
            "n_down": int((attribution["return_1d"] < 0).sum()),
            "pct_up": round(100.0 * n_up / n_priced, 1) if n_priced else np.nan,
            "n_with_peer": n_with_peer,
            "n_beating_peers": n_beating,
            "pct_beating_peers": (round(100.0 * n_beating / n_with_peer, 1)
                                  if n_with_peer else np.nan),
        }
    else:
        breadth = {"n_priced": 0, "n_up": 0, "n_down": 0, "pct_up": np.nan,
                   "n_with_peer": 0, "n_beating_peers": 0,
                   "pct_beating_peers": np.nan}

    # Themes (tier2 groups) ranked by |move| that the portfolio does NOT hold
    held_tier2 = set(t for t in pos.join(asset[["tier2"]], how="left")["tier2"]
                     if isinstance(t, str))
    t2 = market["tier2_summary"].reset_index()
    t2["held"] = t2["tier2"].isin(held_tier2)
    biggest_unheld = (t2[~t2["held"]]
                      .reindex(t2[~t2["held"]]["ret_1d"].abs()
                               .sort_values(ascending=False).index)
                      .head(10))

    return {"attribution": attribution, "unheld_themes": biggest_unheld,
            "breadth": breadth}
