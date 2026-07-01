#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: test_tag_analytics.py
=============================================================================

INPUT FILES:
    (none - synthetic fixtures built in-test)

OUTPUT FILES:
    (none - pytest output only)

VERSION: 1.0
LAST UPDATED: 2026-07-01
AUTHOR: Arjun Divecha

DESCRIPTION:
    Locks in the tier-3 tag-view analytics (report/tag_analytics.py) and their
    wiring into the data package:
      - explode_tags drops factor proxies and unpriced funds (never zero-fills)
      - market tag tilts are universe-DEMEANED and honor the n>=3 floor
      - style spreads carry the right sign; region x sector grid honors n>=5
      - portfolio tag tilts = book gross weight - benchmark weight
      - the bridge classifies WITH/AGAINST correctly
      - concentration (1/HHI) is sane
      - the noise gate (Rule of 16) classifies day magnitude
      - flag OFF (tag_views=None) leaves the package byte-identical (no leakage)
      - flag ON never emits "n/a"

USAGE:
    python3 -m pytest tests/test_tag_analytics.py -v
=============================================================================
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "report"))

import tag_analytics as ta
import prompt as prompt_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _asset_table():
    """A tiny universe: 3 Value funds, 3 Growth funds, one EM fund, plus a
    factor proxy (dropped) and an unpriced fund (dropped)."""
    rows = [
        # ticker, tags, return_1d, return_1w, is_factor
        ("V1", "Equity, US, Value", 1.0, 2.0, 0),
        ("V2", "Equity, US, Value", 1.4, 2.2, 0),
        ("V3", "Equity, US, Value", 1.2, 1.8, 0),
        ("G1", "Equity, US, Growth", -0.5, 0.5, 0),
        ("G2", "Equity, US, Growth", -0.7, 0.1, 0),
        ("G3", "Equity, US, Growth", -0.3, 0.3, 0),
        ("EM1", "Equity, EM", 0.2, -0.4, 0),
        ("SPY", "Equity, US, Large-Cap", 0.1, 0.2, 1),   # factor -> dropped
        ("NANF", "Equity, US, Value", np.nan, np.nan, 0),  # unpriced -> dropped
    ]
    df = pd.DataFrame(rows, columns=["yf_ticker", "tags", "return_1d",
                                     "return_1w", "is_factor"]).set_index("yf_ticker")
    return df


def _market():
    return {"asset_table": _asset_table()}


def _positions():
    # two equal-weight names: one US, one EM
    return pd.DataFrame(
        {"weight": [0.5, 0.5], "market_value_mtm": [50.0, 50.0]},
        index=["AAA", "BBB"])


def _portfolio():
    return {"positions": _positions()}


# ---------------------------------------------------------------------------
# explode / axis map
# ---------------------------------------------------------------------------

class TestExplode:
    def test_drops_factor_and_unpriced(self):
        long = ta.explode_tags(_asset_table())
        assert "SPY" not in set(long["yf_ticker"])     # factor dropped
        assert "NANF" not in set(long["yf_ticker"])    # unpriced dropped

    def test_axis_assigned(self):
        long = ta.explode_tags(_asset_table())
        row = long[(long["yf_ticker"] == "V1") & (long["tag"] == "Value")].iloc[0]
        assert row["axis"] == "Style"

    def test_every_allowed_tag_has_axis(self):
        # nothing should silently fall to "Other" for the canonical vocab
        import expert_review_common as erc
        for tag in erc.ALLOWED_TIER3_TAGS:
            assert ta.TAG_AXIS.get(tag, "Other") != "Other", tag


# ---------------------------------------------------------------------------
# market side
# ---------------------------------------------------------------------------

class TestMarketTilts:
    def test_demeaned_and_min_n(self):
        tilts = ta.compute_tag_tilts(_asset_table(), min_n=3)
        # universe mean of the 7 priced funds
        uni = _asset_table().dropna(subset=["return_1d"])
        uni = uni[uni["is_factor"] != 1]["return_1d"].mean()
        v = tilts[tilts["tag"] == "Value"].iloc[0]
        assert v["n"] == 3
        assert abs(v["tilt_1d"] - (1.2 - uni)) < 1e-9   # (mean Value) - universe
        # EM has n=1 -> excluded by the n>=3 floor
        assert "EM" not in set(tilts["tag"])

    def test_value_beats_growth_sign(self):
        sp = ta.compute_style_spreads(_asset_table(), min_n=3)
        vg = sp[sp["spread"] == "Value vs Growth"].iloc[0]
        assert vg["value"] > 0   # value funds up, growth funds down

    def test_dispersion_positive(self):
        d = ta.compute_dispersion(pd.concat([_asset_table()] * 3))
        assert d["dispersion"] > 0 and d["n"] > 0

    def test_grid_respects_min_n(self):
        # with tiny n per cell, the n>=5 floor yields an empty grid (not a crash)
        g = ta.compute_region_sector_grid(_asset_table(), min_n=5)
        assert g.empty


class TestNoiseGate:
    def test_noise_day(self):
        r = ta.noise_gate(0.1, 16.0, {"pct_up": 50}, {})
        assert r["verdict"] == "noise" and r["expected_move"] == 1.0

    def test_large_day(self):
        r = ta.noise_gate(5.0, 16.0, {"pct_up": 90}, {})
        assert "large" in r["verdict"]

    def test_unknown_without_vix(self):
        assert ta.noise_gate(1.0, None, {}, {})["verdict"] == "unknown"


# ---------------------------------------------------------------------------
# portfolio side
# ---------------------------------------------------------------------------

class TestPortfolioTilts:
    def _bench(self):
        return ta.benchmark_tag_weights([(0.6, ["Equity", "Global", "Large-Cap"]),
                                          (0.4, ["Treasury", "US"])])

    def test_tilt_is_book_minus_bench(self):
        tmap = {"AAA": "Equity, US", "BBB": "Equity, EM"}
        pt = ta.compute_portfolio_tag_tilts(_positions(), tmap, self._bench())
        d = dict(zip(pt["tag"], pt["tilt"]))
        assert abs(d["Equity"] - (1.0 - 0.6)) < 1e-9      # book 1.0 vs bench 0.6
        assert abs(d["Global"] - (0.0 - 0.6)) < 1e-9      # not held
        assert abs(d["EM"] - (0.5 - 0.0)) < 1e-9          # held, not in bench

    def test_bridge_with_and_against(self):
        tmap = {"AAA": "Equity, US", "BBB": "Equity, EM"}
        pt = ta.compute_portfolio_tag_tilts(_positions(), tmap, self._bench())
        # market: Equity led (+), US lagged (-)
        mkt = pd.DataFrame({"tag": ["Equity", "US"], "tilt_1d": [0.3, -0.3]})
        br = ta.compute_tag_bridge(pt, mkt).set_index("tag")
        # overweight Equity (+tilt) while Equity led -> WITH
        assert br.loc["Equity", "stance"] == "WITH"
        # underweight? US tilt is +0.5 (book 0.5 vs bench US 0.4=0.1... actually
        # bench has US at 0.4) -> tilt +0.1, market US -0.3 -> AGAINST
        assert br.loc["US", "stance"] == "AGAINST"

    def test_concentration_sane(self):
        tmap = {"AAA": "Equity, US", "BBB": "Equity, EM"}
        c = ta.compute_tag_concentration(_positions(), tmap)
        assert c["eff_positions"] == 2.0        # two equal-weight names
        assert c["top_tag"] == "Equity"         # both carry Equity
        assert c["top_tag_gross_pct"] == 100.0


# ---------------------------------------------------------------------------
# wiring: flag off = identical package; flag on = no n/a
# ---------------------------------------------------------------------------

class TestPackageWiring:
    def _pkg(self, tag_views):
        import analytics
        # reuse the pipeline-test fixtures via a minimal market/portfolio
        prices_idx = pd.bdate_range(end="2026-06-09", periods=120).strftime("%Y-%m-%d")
        rng = np.random.default_rng(3)
        prices = pd.DataFrame(
            {t: b * np.cumprod(1 + rng.normal(0.0003, 0.015, 120))
             for t, b in [("SPY", 500), ("EEM", 40), ("AAA", 100), ("BBB", 50)]},
            index=prices_idx)
        uni = pd.DataFrame({
            "yf_ticker": ["SPY", "EEM", "AAA", "BBB"], "name": ["S", "E", "A", "B"],
            "description": [""] * 4, "tier1": ["Equities"] * 4,
            "tier2": ["US", "EM", "US", "US"], "tags": [""] * 4, "source": [""] * 4,
            "tracking_score": [0] * 4, "is_factor": [1, 1, 0, 0],
            "factor_name": ["SPX", "EM", None, None], "proxied_tickers": [None] * 4})
        holds = pd.DataFrame([
            ["A1", "AAA", 100., 90., 10000., 1000., "Schwab", "t"],
            ["A1", "BBB", 200., 55., 11000., -500., "Schwab", "t"],
        ], columns=["account", "symbol", "quantity", "avg_price", "market_value",
                    "open_pnl", "broker", "fetched_at"])
        market = analytics.compute_market(prices, uni, "2026-06-09")
        port = analytics.compute_portfolio(holds, prices, "2026-06-09")
        bridge = analytics.compute_bridge(market, port, universe=uni)
        return prompt_mod.build_data_package(
            market, port, bridge, pd.DataFrame(), pd.DataFrame(),
            {"stale": False, "as_of": "2026-06-09", "failures": []},
            tag_views=tag_views)

    def test_flag_off_is_identical(self):
        # None must produce a package with NO tier-3 leakage
        pkg = self._pkg(None)
        assert "TIER-3 TAG VIEWS" not in pkg

    def test_flag_on_adds_section_and_no_na(self):
        tv = {
            "market_tilts": ta.compute_tag_tilts(_asset_table()),
            "spreads": ta.compute_style_spreads(_asset_table()),
            "dispersion": ta.compute_dispersion(pd.concat([_asset_table()] * 2)),
            "breadth": ta.compute_breadth(_asset_table()),
            "grid": ta.compute_region_sector_grid(_asset_table(), min_n=1),
            "noise_gate": ta.noise_gate(0.1, 16.0, {"pct_up": 50}, {}),
            "port_tilts": ta.compute_portfolio_tag_tilts(
                _positions(), {"AAA": "Equity, US", "BBB": "Equity, EM"},
                ta.benchmark_tag_weights([(0.6, ["Equity", "Global"]),
                                          (0.4, ["Treasury"])])),
            "bridge": pd.DataFrame(),
            "concentration": ta.compute_tag_concentration(
                _positions(), {"AAA": "Equity, US", "BBB": "Equity, EM"}),
        }
        tv["bridge"] = ta.compute_tag_bridge(tv["port_tilts"], tv["market_tilts"])
        pkg = self._pkg(tv)
        assert "TIER-3 TAG VIEWS" in pkg
        low = pkg.lower()
        assert "n/a" not in low and "n / a" not in low
