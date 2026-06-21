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
        for needle in ["| Metric | Value |", "Alpha (vs single-factor SPY)",
                       "YTD (current-weights proxy)", "PORTFOLIO BREADTH",
                       "HOUSEHOLD TOTAL", "days ago)"]:
            assert needle in pkg, needle
        assert "| x | y |" not in pkg   # prior-summary table stripped

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
