#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: test_analytics.py
=============================================================================

INPUT FILES:
    (none - synthetic fixtures built in-test)

OUTPUT FILES:
    (none - pytest output only)

VERSION: 1.0
LAST UPDATED: 2026-06-09
AUTHOR: Arjun Divecha

DESCRIPTION:
    Unit tests for every financial formula in report/analytics.py.
    Locks down the exact bugs found in the legacy system review:
      - short positions must contribute POSITIVELY when prices fall
      - units: returns in %, contributions in bps (no double scaling)
      - missing prices must not poison portfolio aggregates
      - YTD anchored to last trading day of prior year
      - beta requires minimum observations

USAGE:
    python3 -m pytest tests/ -v
=============================================================================
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "report"))

import analytics
from analytics import (aggregate_holdings, annualized_vol, beta_vs,
                       compute_bridge, compute_market, compute_portfolio,
                       daily_returns, return_over, ytd_return)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_prices(days=300, seed=7) -> pd.DataFrame:
    """Synthetic price matrix spanning the year boundary, business days."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-06-09", periods=days).strftime("%Y-%m-%d")
    data = {
        "SPY": 500 * np.cumprod(1 + rng.normal(0.0004, 0.01, days)),
        "AAA": 100 * np.cumprod(1 + rng.normal(0.0008, 0.02, days)),
        "BBB": 50 * np.cumprod(1 + rng.normal(-0.0002, 0.015, days)),
    }
    return pd.DataFrame(data, index=idx)


def make_holdings() -> pd.DataFrame:
    """Two longs, one short, one cash row, one unpriceable symbol."""
    return pd.DataFrame([
        # account, symbol, quantity, avg_price, market_value, open_pnl, broker, fetched_at
        ["A1", "AAA", 1000.0, 90.0, 100000.0, 10000.0, "Schwab", "t"],
        ["A1", "BBB", 2000.0, 55.0, 100000.0, -10000.0, "Schwab", "t"],
        ["A2", "SPY", -100.0, 480.0, -50000.0, -2000.0, "IBKR", "t"],
        ["A2", "CASH", 25000.0, 1.0, 25000.0, 0.0, "IBKR", "t"],
        ["A1", "ZZZNOPRICE", 500.0, 10.0, 5000.0, 0.0, "Schwab", "t"],
    ], columns=["account", "symbol", "quantity", "avg_price", "market_value",
                "open_pnl", "broker", "fetched_at"])


# ---------------------------------------------------------------------------
# Return primitives
# ---------------------------------------------------------------------------

class TestDailyReturns:
    def test_simple_return_in_percent(self):
        p = pd.DataFrame({"X": [100.0, 102.0]}, index=["2026-01-01", "2026-01-02"])
        r = daily_returns(p)
        assert r["X"].iloc[-1] == pytest.approx(2.0)   # percent, not decimal

    def test_nan_gap_not_filled(self):
        p = pd.DataFrame({"X": [100.0, np.nan, 110.0]},
                         index=["2026-01-01", "2026-01-02", "2026-01-03"])
        r = daily_returns(p)
        assert np.isnan(r["X"].iloc[1])
        assert np.isnan(r["X"].iloc[2])   # no fill-forward across the gap


class TestReturnOver:
    def test_window_return(self):
        idx = [f"2026-01-{d:02d}" for d in range(1, 11)]
        p = pd.DataFrame({"X": np.linspace(100, 109, 10)}, index=idx)
        r5 = return_over(p, 5)
        assert r5["X"] == pytest.approx((109 / 104 - 1) * 100)

    def test_insufficient_history_is_nan(self):
        p = pd.DataFrame({"X": [100.0, 101.0]}, index=["2026-01-01", "2026-01-02"])
        assert np.isnan(return_over(p, 5)["X"])


class TestYTD:
    def test_anchored_to_prior_year_close(self):
        p = pd.DataFrame({"X": [100.0, 200.0, 110.0]},
                         index=["2025-12-30", "2025-12-31", "2026-01-05"])
        y = ytd_return(p)
        assert y["X"] == pytest.approx((110 / 200 - 1) * 100)   # vs 12-31, not 12-30

    def test_new_listing_is_nan(self):
        p = pd.DataFrame({"X": [10.0, 11.0]}, index=["2026-02-01", "2026-06-09"])
        assert np.isnan(ytd_return(p)["X"])


class TestVolAndBeta:
    def test_vol_annualization(self):
        rng = np.random.default_rng(1)
        r = pd.DataFrame({"X": rng.normal(0, 1.0, 100)})    # 1% daily vol
        v = annualized_vol(r, window=60)
        assert v["X"] == pytest.approx(np.sqrt(252), rel=0.3)

    def test_beta_of_factor_with_itself_is_one(self):
        prices = make_prices()
        rets = daily_returns(prices)
        b = beta_vs(rets, rets["SPY"])
        assert b["SPY"] == pytest.approx(1.0)

    def test_beta_insufficient_obs_is_nan(self):
        prices = make_prices(days=20)
        rets = daily_returns(prices)
        b = beta_vs(rets, rets["SPY"])     # 19 obs < min_beta_obs (30)
        assert np.isnan(b["AAA"])


# ---------------------------------------------------------------------------
# Holdings aggregation
# ---------------------------------------------------------------------------

class TestAggregateHoldings:
    def test_multi_account_summing(self):
        raw = make_holdings()
        extra = raw.iloc[[0]].copy()
        extra["account"] = "A9"
        agg = aggregate_holdings(pd.concat([raw, extra]))
        aaa = agg[agg.symbol == "AAA"].iloc[0]
        assert aaa["quantity"] == 2000.0
        assert aaa["market_value"] == 200000.0

    def test_cash_flagged(self):
        agg = aggregate_holdings(make_holdings())
        assert agg[agg.symbol == "CASH"]["is_cash"].iloc[0]

    def test_zero_position_dropped(self):
        raw = make_holdings()
        closer = raw.iloc[[0]].copy()
        closer["quantity"] = -1000.0
        closer["market_value"] = -100000.0
        agg = aggregate_holdings(pd.concat([raw, closer]))
        assert "AAA" not in agg.symbol.values


# ---------------------------------------------------------------------------
# Portfolio computation - the critical sign/unit/NaN tests
# ---------------------------------------------------------------------------

class TestPortfolio:
    def setup_method(self):
        self.prices = make_prices()
        self.result = compute_portfolio(make_holdings(), self.prices)

    def test_short_position_negative_weight(self):
        pos = self.result["positions"]
        assert pos.loc["SPY", "weight"] < 0
        assert pos.loc["SPY", "market_value_mtm"] < 0

    def test_short_contribution_sign(self):
        """If SPY fell today, the short MUST contribute positively."""
        pos = self.result["positions"]
        spy_ret = pos.loc["SPY", "return_1d"]
        contrib = pos.loc["SPY", "contribution_bps"]
        assert np.sign(contrib) == np.sign(-spy_ret) or contrib == 0

    def test_contribution_units_bps(self):
        """BOD weight * return(%) * 100 = bps. 1% move at 10% weight = 10bps."""
        pos = self.result["positions"]
        row = pos.loc["AAA"]
        expected = row["weight_bod"] * row["return_1d"] * 100.0
        assert row["contribution_bps"] == pytest.approx(expected)

    def test_contributions_tie_to_dollar_pnl(self):
        """Sum of contributions == dollar P&L / priced-book BOD gross."""
        pos = self.result["positions"]
        prices = self.prices
        p1, p0 = prices.iloc[-1], prices.iloc[-2]
        pnl = 0.0
        gross_bod = 0.0
        for sym, row in pos.iterrows():
            if row["has_price"] and pd.notna(p0.get(sym)):
                pnl += row["quantity"] * (p1[sym] - p0[sym])
                gross_bod += abs(row["quantity"] * p0[sym])
        expected_pct = pnl / gross_bod * 100.0
        assert self.result["summary"]["return_1d"] == pytest.approx(
            expected_pct, abs=1e-9)

    def test_portfolio_return_is_sum_of_contributions(self):
        pos = self.result["positions"]
        s = self.result["summary"]
        manual = pos["contribution_bps"].dropna().sum() / 100.0
        assert s["return_1d"] == pytest.approx(manual)

    def test_unpriced_position_isolated(self):
        """ZZZNOPRICE has no price: excluded from returns, kept in exposure."""
        pos = self.result["positions"]
        dq = self.result["data_quality"]
        assert "ZZZNOPRICE" in dq["unpriced_symbols"]
        assert not pos.loc["ZZZNOPRICE", "has_price"]
        # exposure includes its broker market value
        assert self.result["summary"]["gross_exposure"] >= 5000.0
        # and the portfolio return is still a finite number
        assert np.isfinite(self.result["summary"]["return_1d"])

    def test_exposures(self):
        s = self.result["summary"]
        assert s["long_value"] > 0
        assert s["short_value"] < 0
        assert s["gross_exposure"] == pytest.approx(
            s["long_value"] - s["short_value"], rel=1e-9)
        assert s["net_exposure"] == pytest.approx(
            s["long_value"] + s["short_value"], rel=1e-9)
        assert s["n_short"] == 1

    def test_cash_in_total_not_in_gross(self):
        s = self.result["summary"]
        assert s["cash_value"] == pytest.approx(25000.0)
        assert s["total_value"] == pytest.approx(s["net_exposure"] + 25000.0)

    def test_alpha_decomposition_consistency(self):
        s = self.result["summary"]
        if np.isfinite(s["alpha_1d"]):
            assert s["alpha_1d"] == pytest.approx(
                s["return_1d"] - s["expected_return_1d"])


# ---------------------------------------------------------------------------
# Market computation
# ---------------------------------------------------------------------------

class TestMarket:
    def setup_method(self):
        self.prices = make_prices()
        self.universe = pd.DataFrame({
            "yf_ticker": ["SPY", "AAA", "BBB", "MISSING1"],
            "name": ["S&P", "Asset A", "Asset B", "Ghost"],
            "tier1": ["Equities", "Equities", "Fixed Income", "Commodities"],
            "tier2": ["US Large Cap", "US Large Cap", "Credit", "Energy"],
            "tags": ["", "", "", ""],
            "is_factor": [1, 0, 0, 0],
            "factor_name": ["SPX", None, None, None],
        })
        self.m = compute_market(self.prices, self.universe)

    def test_data_quality_reports_missing(self):
        dq = self.m["data_quality"]
        assert "MISSING1" in dq["missing_tickers"]
        assert dq["n_priced_today"] == 3

    def test_tier1_means_exclude_missing(self):
        t1 = self.m["tier1_summary"]
        assert "Commodities" not in t1.index   # only asset had no prices
        assert t1.loc["Equities", "n"] == 2

    def test_factor_table_has_spy(self):
        assert "SPY" in self.m["factor_table"].index


class TestSparseHolidayRows:
    """The Juneteenth/holiday sparse-row bug: a near-empty row (US closed)
    between two full sessions must be dropped, or pct_change poisons the next
    session's returns to NaN for every US ticker."""

    def _matrix_with_holiday(self):
        idx = ["2026-06-18", "2026-06-19", "2026-06-22"]
        # 06-19 = Juneteenth: only AAA (a name that trades through) prints;
        # SPY / BBB are NaN (US markets closed).
        return pd.DataFrame({
            "SPY": [500.0, np.nan, 510.0],
            "AAA": [100.0, 100.5, 102.0],
            "BBB": [50.0, np.nan, 49.0],
        }, index=idx)

    def test_filter_drops_sparse_row(self):
        import data as data_mod
        clean = data_mod.filter_sparse_rows(self._matrix_with_holiday())
        assert list(clean.index) == ["2026-06-18", "2026-06-22"]

    def test_filter_keeps_all_full_rows(self):
        import data as data_mod
        p = make_prices()                    # continuous, no sparse row
        assert len(data_mod.filter_sparse_rows(p)) == len(p)

    def test_post_holiday_1d_spans_last_real_session(self):
        import data as data_mod
        clean = data_mod.filter_sparse_rows(self._matrix_with_holiday())
        rets = daily_returns(clean)
        # SPY's 06-22 return is now vs 06-18 (510/500-1), NOT NaN-vs-holiday
        assert rets["SPY"].iloc[-1] == pytest.approx((510 / 500 - 1) * 100)
        assert rets[["SPY", "BBB"]].iloc[-1].notna().all()

    def test_latest_trading_date_skips_holiday(self):
        import data as data_mod
        # append a trailing sparse holiday row; as-of must NOT be the holiday
        p = make_prices()
        p2 = pd.concat([p, pd.DataFrame(
            {c: [np.nan] for c in p.columns}, index=["2026-06-19"])])
        assert data_mod.latest_trading_date(p2) != "2026-06-19"


class TestSpyMandatory:
    def test_missing_spy_raises(self):
        prices = make_prices().drop(columns=["SPY"])
        uni = pd.DataFrame({
            "yf_ticker": ["AAA", "BBB"], "name": ["A", "B"],
            "tier1": ["Equities", "Fixed Income"],
            "tier2": ["US Large Cap", "Credit"], "tags": ["", ""],
            "is_factor": [0, 0], "factor_name": [None, None]})
        with pytest.raises(RuntimeError, match="SPY absent"):
            compute_market(prices, uni)


class TestBridge:
    def test_attribution_and_unheld(self):
        prices = make_prices()
        universe = pd.DataFrame({
            "yf_ticker": ["SPY", "AAA", "BBB"],
            "name": ["S&P", "A", "B"],
            "tier1": ["Equities"] * 3,
            "tier2": ["US Large Cap", "US Large Cap", "Credit"],
            "tags": [""] * 3,
            "is_factor": [1, 0, 0],
            "factor_name": ["SPX", None, None],
        })
        market = compute_market(prices, universe)
        portfolio = compute_portfolio(make_holdings(), prices)
        bridge = compute_bridge(market, portfolio)
        att = bridge["attribution"]
        assert set(att["symbol"]) == {"AAA", "BBB", "SPY"}
        # contributions sorted ascending (worst first)
        assert (att["contribution_bps"].diff().dropna() >= -1e-12).all()
