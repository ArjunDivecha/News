#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: test_scenarios.py
=============================================================================

INPUT FILES:
    (none - synthetic fixtures built in-test)

OUTPUT FILES:
    (none - pytest output only)

VERSION: 1.0
LAST UPDATED: 2026-07-01
AUTHOR: Arjun Divecha

DESCRIPTION:
    Locks in the scenario-risk engine (report/scenarios.py):
      - bucket shocks apply to look-through slices and sum correctly
      - a SHORT position gains when its bucket is shocked down (sign safety)
      - per-symbol overrides beat bucket shocks (incl. for look-through funds)
      - cash is never shocked
      - hurts/helps ranking and the $50k noise floor
      - crash beta: a book that IS the index has beta ~1 in both regimes
      - liquidity ladder buckets and overrides

USAGE:
    python3 -m pytest tests/test_scenarios.py -v
=============================================================================
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "report"))

import scenarios as sc


def _pos(rows):
    """rows: {sym: mtm}"""
    return pd.DataFrame({"market_value_mtm": pd.Series(rows),
                         "has_price": True})


SCN = {
    "key": "t", "name": "Test", "anchor": "test",
    "shocks": {("Equities", "US"): -40, ("Equities", "EM"): -50,
               ("Equities", None): -30, ("Bonds", "US"): 10,
               ("Alternatives", None): 0, ("Cash", None): 0,
               ("Other", None): 0},
    "overrides": {"SPECIAL": -80},
}


class TestShockApplication:
    def test_bucket_shocks_and_cash(self, monkeypatch):
        monkeypatch.setattr(sc, "SCENARIOS", [SCN])
        pos = _pos({"US1": 1_000_000, "EM1": 2_000_000})
        tmap = {"US1": "Equity, US", "EM1": "Equity, EM"}
        t = sc.compute_scenarios(pos, tmap, cash_value=1_000_000, lookthrough={})
        r = t.iloc[0]
        # -40% of 1M + -50% of 2M + 0 on cash = -1.4M on a 4M base
        assert abs(r["impact_dollars"] - (-1_400_000)) < 1
        assert abs(r["impact_pct"] - (-35.0)) < 1e-6

    def test_short_gains_on_down_shock(self, monkeypatch):
        monkeypatch.setattr(sc, "SCENARIOS", [SCN])
        pos = _pos({"SHRT": -1_000_000})           # short US equity
        t = sc.compute_scenarios(pos, {"SHRT": "Equity, US"}, 0.0, {})
        assert abs(t.iloc[0]["impact_dollars"] - 400_000) < 1   # -1M x -40%
        assert t.iloc[0]["helps"][0][0] == "SHRT"

    def test_override_beats_bucket(self, monkeypatch):
        monkeypatch.setattr(sc, "SCENARIOS", [SCN])
        pos = _pos({"SPECIAL": 1_000_000})
        t = sc.compute_scenarios(pos, {"SPECIAL": "Equity, US"}, 0.0, {})
        assert abs(t.iloc[0]["impact_dollars"] - (-800_000)) < 1  # -80 not -40

    def test_lookthrough_slices_shocked_separately(self, monkeypatch):
        monkeypatch.setattr(sc, "SCENARIOS", [SCN])
        lt = {"MULTI": {"class": {"Equities": 0.5, "Bonds": 0.5},
                        "equity_region": {"US": 1.0},
                        "bond_region": {"US": 1.0}}}
        pos = _pos({"MULTI": 1_000_000})
        t = sc.compute_scenarios(pos, {"MULTI": ""}, 0.0, lookthrough=lt)
        # 500k eq x -40% + 500k bonds x +10% = -200k + 50k = -150k
        assert abs(t.iloc[0]["impact_dollars"] - (-150_000)) < 1

    def test_override_covers_all_lookthrough_slices(self, monkeypatch):
        monkeypatch.setattr(sc, "SCENARIOS", [SCN])
        lt = {"SPECIAL": {"class": {"Equities": 0.5, "Bonds": 0.5},
                          "equity_region": {"US": 1.0},
                          "bond_region": {"US": 1.0}}}
        pos = _pos({"SPECIAL": 1_000_000})
        t = sc.compute_scenarios(pos, {"SPECIAL": ""}, 0.0, lookthrough=lt)
        assert abs(t.iloc[0]["impact_dollars"] - (-800_000)) < 1  # -80 on all

    def test_noise_floor_on_hurts(self, monkeypatch):
        monkeypatch.setattr(sc, "SCENARIOS", [SCN])
        pos = _pos({"US1": 100_000, "TINY": 10_000})   # TINY loses only 4k
        t = sc.compute_scenarios(pos, {"US1": "Equity, US",
                                       "TINY": "Equity, US"}, 0.0, {})
        hurt_syms = [s for s, _ in t.iloc[0]["hurts"]]
        assert "TINY" not in hurt_syms

    def test_real_scenario_set_covers_all_buckets(self):
        # every real scenario must define the class-default keys so no slice
        # can silently fall through to an implicit 0 for a MISSING bucket
        for scn in sc.SCENARIOS:
            for cls in ["Equities", "Bonds", "Alternatives", "Cash", "Other"]:
                assert (cls, None) in scn["shocks"] or any(
                    k[0] == cls for k in scn["shocks"]), (scn["key"], cls)


class TestCrashBeta:
    def test_index_book_has_beta_one(self):
        rng = np.random.default_rng(7)
        idx = pd.bdate_range(end="2026-06-30", periods=300).strftime("%Y-%m-%d")
        spy = 500 * np.cumprod(1 + rng.normal(0.0003, 0.012, 300))
        prices = pd.DataFrame({"SPY": spy}, index=idx)
        pos = _pos({"SPY": 1_000_000})
        cb = sc.compute_crash_beta(prices, pos)
        assert abs(cb["full_beta"] - 1.0) < 0.01
        assert abs(cb["crash_beta"] - 1.0) < 0.05
        assert cb["coverage_pct"] == 100.0

    def test_missing_factor_graceful(self):
        prices = pd.DataFrame({"AAA": [1.0, 1.1]}, index=["a", "b"])
        cb = sc.compute_crash_beta(prices, _pos({"AAA": 100.0}))
        assert np.isnan(cb["full_beta"])


class TestLiquidityLadder:
    def test_buckets_and_overrides(self):
        pos = _pos({"SPY": 1_000_000, "BAUPOST": 2_000_000,
                    "VTMEF": 500_000, "GBMBX": 700_000})
        lq = sc.compute_liquidity_ladder(pos, cash_value=300_000)
        d = dict(zip(lq["bucket"], lq["value"]))
        assert d["Cash / same day"] == 300_000
        assert d["Exchange-traded (T+1/T+2)"] == 1_000_000
        assert d["LP lockup (quarterly+ redemption)"] == 2_000_000
        assert d["Closed-end fund — liquid, discount risk"] == 500_000
        assert d["Daily-NAV mutual fund (1-3 days)"] == 700_000
        assert abs(lq["pct"].sum() - 100.0) < 1e-6

    def test_unpriced_bucket(self):
        pos = pd.DataFrame({"market_value_mtm": [100.0],
                            "has_price": [False]}, index=["DEAD"])
        lq = sc.compute_liquidity_ladder(pos, cash_value=0.0)
        assert "Unpriced / defunct lines" in set(lq["bucket"])
