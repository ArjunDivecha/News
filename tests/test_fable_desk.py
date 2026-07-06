#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: test_fable_desk.py
=============================================================================

INPUT FILES:
    (none - synthetic fixtures; a temporary SQLite db per test)

OUTPUT FILES:
    (none - pytest output only)

VERSION: 1.0
LAST UPDATED: 2026-07-06
AUTHOR: Arjun Divecha

DESCRIPTION:
    Unit tests for Fable's Desk (report/fable_desk.py + the desk_ideas
    persistence layer in report/db.py) and for the policy check
    (analytics.compute_policy_check):

      - desk-json trailer parsing: happy path, missing trailer, missing
        idea fields, prose-only output
      - desk_ideas round trip: save -> open-ideas -> grade -> closed
      - idempotent re-run of a date (no duplicate ideas)
      - policy check: vol math vs a hand-computed benchmark, pro-rated
        return hurdle, breach flags, missing-benchmark failure

DEPENDENCIES:
    - pytest
    - pandas / numpy

USAGE:
    python3 -m pytest tests/test_fable_desk.py -v
=============================================================================
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "report"))

import analytics
import db
import fable_desk


# ---------------------------------------------------------------- trailer
GOOD_OUTPUT = """My opening read: the tape is rotating, not de-risking.

**Trim Vietnam into the FTSE flows.** Thesis text here.
**Conviction:** High · **Horizon:** 3 months · **Invalidation:** FTSE delays.

Scorecard: (first edition — nothing to grade).

Sources: [example](https://example.com)

```desk-json
{"ideas": [{"title": "Trim Vietnam into the FTSE flows",
            "action": "Trim Vietnam Enterprise to 12% of gross by Sep 21",
            "conviction": "High", "horizon": "3 months",
            "invalidation": "FTSE delays or waters down the phasing"}],
 "grades": [{"id": 7, "grade": "killed", "note": "invalidation hit"}]}
```"""


class TestTrailerParsing:
    def test_happy_path(self):
        md, ideas, grades = fable_desk.parse_desk_output(GOOD_OUTPUT)
        assert "desk-json" not in md
        assert md.startswith("My opening read")
        assert md.rstrip().endswith("[example](https://example.com)")
        assert len(ideas) == 1
        assert ideas[0]["conviction"] == "High"
        assert grades == [{"id": 7, "grade": "killed",
                           "note": "invalidation hit"}]

    def test_missing_trailer_raises(self):
        with pytest.raises(ValueError, match="trailer"):
            fable_desk.parse_desk_output("Just prose, no machine block.")

    def test_missing_idea_fields_raises(self):
        bad = ('Prose.\n\n```desk-json\n{"ideas": [{"title": "x"}], '
               '"grades": []}\n```')
        with pytest.raises(ValueError, match="missing fields"):
            fable_desk.parse_desk_output(bad)

    def test_trailer_only_raises(self):
        bad = '```desk-json\n{"ideas": [], "grades": []}\n```'
        with pytest.raises(ValueError, match="no prose"):
            fable_desk.parse_desk_output(bad)

    def test_zero_ideas_is_valid(self):
        ok = ('Nothing clears the bar today.\n\n```desk-json\n'
              '{"ideas": [], "grades": []}\n```')
        md, ideas, grades = fable_desk.parse_desk_output(ok)
        assert ideas == [] and grades == []
        assert md == "Nothing clears the bar today."


# ---------------------------------------------------------------- desk db
@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setitem(db.PATHS, "db", tmp_path / "test_report.db")
    db.init_schema()
    return db


IDEA = {"title": "Long yen at a 40-year extreme",
        "action": "Buy 3m JPY calls, 25bp premium",
        "conviction": "High", "horizon": "1-3 months",
        "invalidation": "Clean break above 165 with Tokyo silent"}


class TestDeskIdeasDB:
    def test_round_trip_and_grading(self, tmp_db):
        assert tmp_db.get_open_desk_ideas() == []
        tmp_db.save_desk_ideas("2026-07-06", [IDEA])
        open_ideas = tmp_db.get_open_desk_ideas()
        assert len(open_ideas) == 1
        idea_id = open_ideas[0]["id"]
        assert open_ideas[0]["title"] == IDEA["title"]

        tmp_db.apply_desk_grades(
            [{"id": idea_id, "grade": "confirmed", "note": "MoF intervened"}],
            graded_on="2026-07-10")
        assert tmp_db.get_open_desk_ideas() == []
        with tmp_db.connect() as conn:
            row = conn.execute("SELECT status, grade_note, graded_on "
                               "FROM desk_ideas WHERE id=?",
                               (idea_id,)).fetchone()
        assert row["status"] == "confirmed"
        assert row["grade_note"] == "MoF intervened"
        assert row["graded_on"] == "2026-07-10"

    def test_rerun_same_date_is_idempotent(self, tmp_db):
        tmp_db.save_desk_ideas("2026-07-06", [IDEA])
        tmp_db.save_desk_ideas("2026-07-06", [IDEA])   # re-run of the day
        assert len(tmp_db.get_open_desk_ideas()) == 1

    def test_open_grade_keeps_idea_open(self, tmp_db):
        tmp_db.save_desk_ideas("2026-07-06", [IDEA])
        idea_id = tmp_db.get_open_desk_ideas()[0]["id"]
        tmp_db.apply_desk_grades([{"id": idea_id, "grade": "open",
                                   "note": "still waiting"}])
        assert len(tmp_db.get_open_desk_ideas()) == 1


# ---------------------------------------------------------------- policy
def _prices(n=120, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-01-02", periods=n).strftime("%Y-%m-%d")
    def walk(vol_daily):
        return 100 * np.cumprod(1 + rng.normal(0, vol_daily, n))
    return pd.DataFrame({
        "ACWI": walk(0.008), "TLT": walk(0.006),
        "RISKY": walk(0.02), "SLEEPY": walk(0.004),
    }, index=dates)


POLICY = {"real_return_target_pct": 5.0, "inflation_assumption_pct": 3.0,
          "inflation_asof": "2026-07", "vol_tolerance_ratio": 1.10}
LEGS = [("ACWI", 0.60), ("TLT", 0.40)]


class TestPolicyCheck:
    def test_benchmark_vol_matches_hand_computation(self):
        prices = _prices()
        asof = prices.index[-1]
        pos = pd.DataFrame({"market_value_mtm": [500_000.0, 500_000.0]},
                           index=["RISKY", "SLEEPY"])
        pc = analytics.compute_policy_check(
            pos, prices, asof, household_ytd_pct=6.0,
            household_total_value=1_000_000.0, policy=POLICY,
            benchmark_legs=LEGS, vol_window=60)
        rets = prices.pct_change() * 100
        bench = (0.6 * rets["ACWI"] + 0.4 * rets["TLT"]).dropna().tail(60)
        assert pc["bench_vol_pct"] == pytest.approx(
            float(bench.std() * np.sqrt(252)))
        # 50/50 fully-priced book: coverage 100
        assert pc["coverage_pct"] == pytest.approx(100.0)

    def test_cash_damps_vol(self):
        prices = _prices()
        asof = prices.index[-1]
        pos = pd.DataFrame({"market_value_mtm": [500_000.0]}, index=["RISKY"])
        full = analytics.compute_policy_check(
            pos, prices, asof, 6.0, 500_000.0, POLICY, LEGS, 60)
        halved = analytics.compute_policy_check(
            pos, prices, asof, 6.0, 1_000_000.0, POLICY, LEGS, 60)
        assert halved["hh_vol_pct"] == pytest.approx(
            full["hh_vol_pct"] / 2, rel=1e-6)
        assert halved["coverage_pct"] == pytest.approx(50.0)

    def test_prorated_hurdle_and_verdicts(self):
        prices = _prices()
        asof = "2026-07-02"   # day 183 of a non-leap year
        pos = pd.DataFrame({"market_value_mtm": [1_000_000.0]},
                           index=["SLEEPY"])
        pc = analytics.compute_policy_check(
            pos, prices, asof, household_ytd_pct=8.0,
            household_total_value=1_000_000.0, policy=POLICY,
            benchmark_legs=LEGS, vol_window=60)
        assert pc["nominal_target_pct"] == pytest.approx(8.0)
        assert pc["prorated_target_pct"] == pytest.approx(
            8.0 * 183 / 365, rel=1e-3)
        assert pc["return_on_track"] is True          # 8.0 > ~4.0
        assert pc["vol_breach"] is False              # sleepy < 60/40
        behind = analytics.compute_policy_check(
            pos, prices, asof, household_ytd_pct=1.0,
            household_total_value=1_000_000.0, policy=POLICY,
            benchmark_legs=LEGS, vol_window=60)
        assert behind["return_on_track"] is False

    def test_vol_breach_flags(self):
        prices = _prices()
        asof = prices.index[-1]
        pos = pd.DataFrame({"market_value_mtm": [1_000_000.0]},
                           index=["RISKY"])
        pc = analytics.compute_policy_check(
            pos, prices, asof, 6.0, 1_000_000.0, POLICY, LEGS, 60)
        assert pc["vol_ratio"] > 1.10
        assert pc["vol_breach"] is True

    def test_missing_benchmark_leg_raises(self):
        prices = _prices().drop(columns=["TLT"])
        pos = pd.DataFrame({"market_value_mtm": [1.0]}, index=["RISKY"])
        with pytest.raises(ValueError, match="benchmark leg"):
            analytics.compute_policy_check(
                pos, prices, prices.index[-1], 6.0, 1.0, POLICY, LEGS, 60)
