"""
=============================================================================
SCRIPT NAME: test_ews.py
=============================================================================

DESCRIPTION:
    Unit tests for the MARKET REGIME (Early Warning System) section builder
    at /Users/arjundivecha/Dropbox/AAA Backup/A Working/News/report/ews.py.
    That module reads the external Early Warning project's dashboard JSON
    and renders a markdown section for the daily report's data package.
    These tests lock down its additive-only contract:
      - a good dashboard JSON renders regime, ladder, flags, trend, panel
      - a missing/corrupt file returns None (never raises, never sinks a run)
      - an old "generated" stamp triggers the loud STALE marking with '*'
        (house rule: stale-with-asterisk, never a silent drop)
      - a fresh weekly rebuild renders WITHOUT stale markers
      - prompt.build_data_package accepts the section (ews= parameter)

INPUT FILES:
    (none pre-existing — each test writes a synthetic dashboard JSON
     fixture to a pytest-managed temporary directory, e.g.
     <tmp_path>/data.json, which pytest deletes automatically. The real
     production input, NOT touched by these tests, is
     /Users/arjundivecha/Dropbox/AAA Backup/A Working/Early Warning/dashboard/data.json)

OUTPUT FILES:
    (none — pytest console output only; no persistent file I/O)

VERSION: 1.0
LAST UPDATED: 2026-07-15
AUTHOR: Arjun Divecha

DEPENDENCIES:
    - pytest

USAGE:
    python3 -m pytest tests/test_ews.py -v
=============================================================================
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "report"))

from ews import build_ews_section


def _fixture(generated="2026-07-13") -> dict:
    return {
        "generated": generated,
        "run": "run_test",
        "dates": ["2026-03", "2026-04", "2026-05"],
        "families": {"A": "Structural valuation & credit",
                     "D": "Fast stress, sentiment & positioning",
                     "S": "Shadow signals (zero weight)"},
        "state": ["OUT", "IN", "IN"],
        "ladder": ["YELLOW", "YELLOW", "YELLOW"],
        "composite": [0.69, 0.51, 0.48],
        "composite_pctile": [0.73, 0.55, 0.518],
        "diffusion": [0.23, 0.15, 0.154],
        "signals": [
            {"id": "A1", "name": "Shiller CAPE", "family": "A",
             "shadow": False, "role": "LEADER",
             "current": {"z": 1.735, "raw": 38.1, "flag": True,
                         "asof": "2026-05"}},
            {"id": "D4", "name": "Short interest", "family": "D",
             "shadow": False, "role": "confirmer",
             "current": {"z": 1.762, "raw": 20487.2, "flag": True,
                         "asof": "2026-05"}},
            {"id": "S_gdelt", "name": "GDELT news stress", "family": "S",
             "shadow": True, "role": "weak/na",
             "current": {"z": -0.18, "raw": -0.18, "flag": None,
                         "asof": "2026-05"}},
        ],
        "current": {"date": "2026-05-31", "composite": 0.483,
                    "composite_pctile": 0.518, "diffusion": 0.154,
                    "n_flags": 2, "n_avail": 13, "ladder": "YELLOW",
                    "state": "IN", "flags_on": ["A1", "D4"]},
    }


def _write(tmp_path, data) -> Path:
    p = tmp_path / "data.json"
    p.write_text(json.dumps(data))
    return p


def test_renders_current_reading(tmp_path):
    md = build_ews_section(asof="2026-07-15",
                           path=_write(tmp_path, _fixture()))
    assert md is not None
    assert "MARKET REGIME — EARLY WARNING SYSTEM" in md
    assert "**IN**" in md and "**YELLOW**" in md
    assert "2/13 weighted signals flagged" in md
    assert "A1 Shiller CAPE" in md and "D4 Short interest" in md
    assert "52nd percentile" in md
    # signal month named, taken from current.date
    assert "signal month 2026-05" in md


def test_renders_trend_and_panel(tmp_path):
    md = build_ews_section(asof="2026-07-15",
                           path=_write(tmp_path, _fixture()))
    # the 2026-03 OUT flip must be visible as trend context
    assert "| 2026-03 | OUT |" in md
    assert "REGIME TREND" in md and "SIGNAL PANEL" in md
    # shadow signal marked and rendered flagless as em dash, not "n/a"
    assert "(shadow)" in md
    assert "n/a" not in md.lower()


def test_missing_file_returns_none(tmp_path):
    assert build_ews_section(path=tmp_path / "nope.json") is None


def test_corrupt_json_returns_none(tmp_path):
    p = tmp_path / "data.json"
    p.write_text("{not json")
    assert build_ews_section(path=p) is None


def test_stale_dashboard_is_marked_not_dropped(tmp_path):
    # generated 30 days before as-of: render, but loudly starred
    md = build_ews_section(asof="2026-07-15",
                           path=_write(tmp_path, _fixture("2026-06-15")))
    assert md is not None
    assert "EWS DASHBOARD IS STALE" in md
    assert "**IN***" in md          # regime value carries the stale star


def test_fresh_dashboard_has_no_stale_marker(tmp_path):
    md = build_ews_section(asof="2026-07-15",
                           path=_write(tmp_path, _fixture("2026-07-13")))
    assert "STALE" not in md
    assert "**IN***" not in md      # no stale star on the regime value


def test_build_data_package_accepts_ews():
    import inspect
    import prompt as prompt_mod
    sig = inspect.signature(prompt_mod.build_data_package)
    assert "ews" in sig.parameters
