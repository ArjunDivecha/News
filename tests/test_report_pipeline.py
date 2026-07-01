#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: test_report_pipeline.py
=============================================================================

INPUT FILES:
    (none - synthetic fixtures built in-test)

OUTPUT FILES:
    (none - pytest output only)

VERSION: 1.0
LAST UPDATED: 2026-06-20
AUTHOR: Arjun Divecha

DESCRIPTION:
    Locks in the 2026-06-20 report-writing review fixes:
      - PDF table validator rejects truncated/malformed reports (the 06-18 bug)
      - data package emits a summary TABLE, proxy/scope labels, breadth, and
        a stale-strip on prior summaries
      - compute_bridge backfills tier-2 and labels off-universe names, and
        reports breadth
      - return_percentile is integer-valued and masks sparse data

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
import data as data_mod
import prompt as prompt_mod
import pdf as pdf_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _prices(days=300, seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-06-09", periods=days).strftime("%Y-%m-%d")
    cols = {t: base * np.cumprod(1 + rng.normal(0.0004, 0.018, days))
            for t, base in [("SPY", 500), ("EEM", 40), ("AAA", 100),
                            ("BBB", 50), ("CCC", 30), ("XOFF", 70)]}
    return pd.DataFrame(cols, index=idx)


def _universe():
    return pd.DataFrame({
        "yf_ticker": ["SPY", "EEM", "AAA", "BBB", "CCC"],
        "name": ["S&P", "EM", "A", "B", "C"],
        "description": [""] * 5,
        "tier1": ["Equities"] * 5,
        "tier2": ["US Large Cap", "EM", "US Large Cap", "US Large Cap",
                  "US Large Cap"],
        "tags": [""] * 5, "source": [""] * 5, "tracking_score": [0] * 5,
        "is_factor": [1, 1, 0, 0, 0],
        "factor_name": ["SPX", "EM", None, None, None],
        "proxied_tickers": [None] * 5,
    })


def _holdings():
    # XOFF is priced but NOT in the universe -> "Portfolio-Specific"
    return pd.DataFrame([
        ["A1", "AAA", 1000., 90., 100000., 10000., "Schwab", "t"],
        ["A1", "BBB", 2000., 55., 100000., -10000., "Schwab", "t"],
        ["A1", "XOFF", 100., 50., 7000., 500., "Schwab", "t"],
        ["A2", "SPY", -100., 480., -50000., -2000., "IBKR", "t"],
        ["A2", "CASH", 25000., 1., 25000., 0., "IBKR", "t"],
    ], columns=["account", "symbol", "quantity", "avg_price", "market_value",
                "open_pnl", "broker", "fetched_at"])


# ---------------------------------------------------------------------------
# Analytics-level changes
# ---------------------------------------------------------------------------

class TestBridgeBreadthAndBackfill:
    def setup_method(self):
        self.prices = _prices()
        self.universe = _universe()
        self.market = analytics.compute_market(self.prices, self.universe, "2026-06-09")
        self.portfolio = analytics.compute_portfolio(_holdings(), self.prices, "2026-06-09")
        self.bridge = analytics.compute_bridge(
            self.market, self.portfolio, universe=self.universe)

    def test_off_universe_name_labelled_portfolio_specific(self):
        att = self.bridge["attribution"]
        lab = dict(zip(att["symbol"], att["tier2"]))
        assert lab["XOFF"] == "Portfolio-Specific"
        # and it carries no fabricated peer return
        row = att[att["symbol"] == "XOFF"].iloc[0]
        assert pd.isna(row["peer_return_1d"])

    def test_breadth_present_and_consistent(self):
        br = self.bridge["breadth"]
        assert br["n_priced"] == br["n_up"] + br["n_down"] + \
            int(((self.bridge["attribution"]["return_1d"] == 0).sum()))
        assert 0 <= br["pct_up"] <= 100
        assert br["n_beating_peers"] <= br["n_with_peer"] <= br["n_priced"]

    def test_bridge_backward_compatible_without_universe(self):
        # TestBridge in test_analytics calls compute_bridge(market, portfolio)
        b = analytics.compute_bridge(self.market, self.portfolio)
        assert "attribution" in b and "breadth" in b


class TestStaleYtd:
    """A fund that has not posted its as-of NAV yet (e.g. a GMO mutual fund
    lagging the equity calendar) must still show YTD to its last real print,
    and must NOT dilute its sub-portfolio's YTD toward zero (the 2026-07-01
    GMO bug: GMO YTD read 0.24% instead of ~10%)."""

    def _prices(self):
        idx = pd.bdate_range(end="2026-07-01", periods=380).strftime("%Y-%m-%d")
        eq = pd.Series(range(380), index=idx, dtype=float).map(lambda i: 100 * 1.0005 ** i)
        mf = pd.Series(range(380), index=idx, dtype=float).map(lambda i: 50 * 1.0004 ** i)
        df = pd.DataFrame({"EQ": eq, "MF": mf})
        # MF stops printing on the last (as-of) day — NaN on 2026-07-01
        df.loc[df.index[-1], "MF"] = np.nan
        return df

    def test_ytd_uses_last_available(self):
        p = self._prices()
        ytd = analytics.ytd_return(p, "2026-07-01")
        assert pd.notna(ytd["MF"]), "stale fund must get last-available YTD"
        assert pd.notna(ytd["EQ"])

    def test_subportfolio_ytd_not_diluted(self):
        p = self._prices()
        # a sleeve that is 90% MF (stale) and 10% EQ
        holds = pd.DataFrame([
            ["GMO", "MF", 1800., 1., 90000., 0., "GMO", "t"],
            ["GMO", "EQ", 100., 1., 10000., 0., "GMO", "t"],
        ], columns=["account", "symbol", "quantity", "avg_price", "market_value",
                    "open_pnl", "broker", "fetched_at"])
        subs = analytics.compute_subportfolios(
            pd.DataFrame(columns=holds.columns), p, "2026-07-01",
            gmo_holdings=holds)
        gmo = subs[subs["name"] == "GMO"].iloc[0]
        ytd = analytics.ytd_return(p, "2026-07-01")
        # value-weighted YTD of BOTH names (MF dominates); must be near it, not ~0
        assert gmo["return_ytd"] > 5.0
        assert abs(gmo["return_ytd"] - ytd["MF"]) < abs(ytd["EQ"] - ytd["MF"])

    def test_subportfolio_1d_uses_last_available(self):
        p = self._prices()
        # give MF a distinctive -3% on its last REAL print (the 2nd-to-last row;
        # the last row is NaN because MF has not posted its as-of NAV)
        mf = p["MF"].copy()
        mf.iloc[-2] = mf.iloc[-3] * 0.97
        p["MF"] = mf
        holds = pd.DataFrame([
            ["GMO", "MF", 1800., 1., 90000., 0., "GMO", "t"],
            ["GMO", "EQ", 100., 1., 10000., 0., "GMO", "t"],
        ], columns=["account", "symbol", "quantity", "avg_price", "market_value",
                    "open_pnl", "broker", "fetched_at"])
        subs = analytics.compute_subportfolios(
            pd.DataFrame(columns=holds.columns), p, "2026-07-01",
            gmo_holdings=holds)
        gmo = subs[subs["name"] == "GMO"].iloc[0]
        # the stale fund's -3% last move must flow through (dominant weight),
        # NOT be dropped so the sleeve reads only EQ's ~+0.05%
        assert gmo["return_1d"] < -1.0


class TestPercentile:
    def test_integer_valued(self):
        rets = analytics.daily_returns(_prices())
        p = analytics.return_percentile(rets).dropna()
        assert (p == p.round(0)).all()
        assert (p.between(0, 100)).all()

    def test_sparse_window_masked(self):
        rets = analytics.daily_returns(_prices(days=10))  # < window//2 obs
        assert analytics.return_percentile(rets).isna().all()


# ---------------------------------------------------------------------------
# Prompt / data-package changes
# ---------------------------------------------------------------------------

class TestProseOnly:
    def test_strips_tables_headings_rules(self):
        dirty = ("Keep this.\n| a | b |\n|---|---|\n| 1 | 2 |\n"
                 "### Heading\n---\nAnd this.")
        clean = prompt_mod._prose_only(dirty)
        assert clean == "Keep this. And this."
        assert "|" not in clean and "#" not in clean


class TestDataPackage:
    def _pkg(self, name_map=None):
        prices = _prices()
        universe = _universe()
        market = analytics.compute_market(prices, universe, "2026-06-09")
        portfolio = analytics.compute_portfolio(_holdings(), prices, "2026-06-09")
        bridge = analytics.compute_bridge(market, portfolio, universe=universe)
        subs = analytics.compute_subportfolios(_holdings(), prices, "2026-06-09")
        prior = pd.DataFrame([{"date": "2026-06-05",
                               "executive_summary": "Prose.\n| x | y |\n|---|---|\n| 1 | 2 |"}])
        return prompt_mod.build_data_package(
            market, portfolio, bridge, pd.DataFrame(), prior,
            {"stale": False, "as_of": "2026-06-09", "failures": []},
            subportfolios=subs, name_map=name_map)

    def test_contains_new_sections_and_labels(self):
        pkg = self._pkg()
        for needle in ["| Metric | Value |", "Alpha (vs single-factor S&P 500)",
                       "YTD (current-weights proxy)", "PORTFOLIO BREADTH",
                       "HOUSEHOLD TOTAL", "days ago)"]:
            assert needle in pkg, needle
        assert "| x | y |" not in pkg   # prior-summary table stripped

    def test_no_literal_na_anywhere(self):
        # the project's NO-NA rule: the package must never contain "n/a"/"N/A"
        pkg = self._pkg(name_map={"AAA": "Asset A Corporation"})
        low = pkg.lower()
        assert "n/a" not in low and "n / a" not in low

    def test_uses_names_not_tickers(self):
        # AAA/BBB/SPY get names; XOFF is unmapped -> ticker kept (approved fallback)
        nmap = {"AAA": "Asset A Corporation", "BBB": "Asset B Corporation",
                "SPY": "S&P 500 ETF"}
        pkg = self._pkg(name_map=nmap)
        assert "Asset A Corporation" in pkg     # name rendered in tables
        assert "S&P 500 ETF" in pkg
        assert "XOFF" in pkg                     # unmapped symbol falls back
        # no bare " AAA " row label leaks into a table cell
        assert "| AAA " not in pkg and "| BBB " not in pkg
        # factor correlations use factor NAMES, not the underlying ETF tickers
        assert "EM / SPX" in pkg
        assert "EEM /" not in pkg and "/ SPY:" not in pkg
        # factor-exposure ETF ticker column is dropped, and SPY (held short,
        # in the name_map) renders as its name -> no bare "| SPY " row anywhere
        assert "S&P 500 ETF" in pkg
        assert "| SPY " not in pkg
        # the benchmark label uses S&P 500, not the SPY ticker
        assert "vs S&P 500, 60d" in pkg and "beta x S&P 500" in pkg


class TestTagCache:
    def test_empty_cache_is_not_a_hit(self):
        # a previously-cached EMPTY tag must NOT count as a cache hit, else the
        # symbol is never retried (cache poisoning). Fugu-found bug.
        import tags as tags_mod
        import db
        sym = "__EMPTYCACHETEST__"
        tags_mod.upsert_tags([{"yf_ticker": sym, "tags": "", "tier1": None,
                               "tier2": None, "name": "x", "source": "deepseek"}])
        try:
            r = tags_mod.resolve_tags([sym], fetch=False)
            assert r[sym]["source"] != "cache", "empty cache treated as a hit"
        finally:
            with db.connect() as c:
                c.execute("DELETE FROM security_tags WHERE yf_ticker=?", (sym,))


class TestNameResolver:
    def test_unknown_symbol_falls_back_to_itself(self):
        import names as names_mod
        out = names_mod.resolve_names(["___NOSUCHTICKER1", "___NOSUCHTICKER2"],
                                      fetch=False)
        assert out == {"___NOSUCHTICKER1": "___NOSUCHTICKER1",
                       "___NOSUCHTICKER2": "___NOSUCHTICKER2"}

    def test_empty_input(self):
        import names as names_mod
        assert names_mod.resolve_names([], fetch=False) == {}


class TestHoldingPriceAliases:
    def test_alias_scaled_to_broker_unit_price(self):
        prices = pd.DataFrame([
            {"date": "2026-06-29", "yf_ticker": "VEIL.L",
             "close": 10.0, "volume": 1000},
            {"date": "2026-06-30", "yf_ticker": "VEIL.L",
             "close": 11.0, "volume": 1200},
        ])
        holdings = pd.DataFrame([
            ["A1", "VTMEF", 100.0, 15.0, 2200.0, 700.0, "Schwab", "t"],
        ], columns=["account", "symbol", "quantity", "avg_price",
                    "market_value", "open_pnl", "broker", "fetched_at"])

        out = data_mod.apply_holding_price_aliases(
            prices, holdings, aliases={"VTMEF": "VEIL.L"})
        alias = out[out["yf_ticker"] == "VTMEF"].sort_values("date")

        assert alias["close"].round(6).tolist() == [20.0, 22.0]
        assert set(out["yf_ticker"]) == {"VEIL.L", "VTMEF"}


# ---------------------------------------------------------------------------
# PDF table validator (the 2026-06-18 truncation bug)
# ---------------------------------------------------------------------------

class TestTableValidator:
    def test_good_table_passes(self):
        pdf_mod._validate_report_tables(
            "## X\n\n| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n")

    def test_truncated_row_rejected(self):
        # mirrors the real 2026-06-18 failure: a report that ends mid-table at
        # a bare "| Reg" cell (header has 4 columns, the row has 1).
        bad = "## X\n\n| Unheld | n | 1d | YTD |\n|---|---|---|---|\n| Reg"
        with pytest.raises(RuntimeError, match="Malformed report table"):
            pdf_mod._validate_report_tables(bad)

    def test_render_requires_bottom_line(self, tmp_path):
        # completeness backstop: a report missing the final section (truncated
        # before the end) must be refused even if its tables are well-formed.
        md = ("## Executive Summary\n\nQuiet day.\n\n## The Tape\n\n"
              "| Factor | 1d % |\n|---|---|\n| EM | +1.0 |\n")
        with pytest.raises(RuntimeError, match="INCOMPLETE"):
            pdf_mod.render_pdf(md, "2026-06-09", "2026-06-09 12:00",
                               "claude-opus-4-8", tmp_path)

    def test_escaped_pipe_in_cell_is_valid(self):
        # a cell may legitimately contain an escaped pipe (\|); it must NOT be
        # counted as a column separator (the 2026-06-22 false-positive).
        good = ("## X\n\n| Theme | n | 1d % | YTD % |\n|---|---|---|---|\n"
                "| Sovereign Bonds \\| Corporate Credit (FI) | 1 | -0.34 | 0.08 |\n")
        pdf_mod._validate_report_tables(good)   # must not raise
