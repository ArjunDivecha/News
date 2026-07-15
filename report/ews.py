#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: ews.py
=============================================================================

INPUT FILES:
    - /Users/arjundivecha/Dropbox/AAA Backup/A Working/Early Warning/dashboard/data.json
      (READ-ONLY: the 12-signal market-downturn Early Warning System's
       current reading and monthly history — regime state IN/TRANSITION/OUT,
       escalation ladder, composite danger z-score, diffusion index, and
       per-signal z/flag detail. Rebuilt by that project's weekly_update.py
       every Monday; the underlying signals are MONTHLY with publication
       lags, so the latest signal month normally trails today by 1-2 months.)

OUTPUT FILES:
    (none - returns a rendered markdown section string to main.py)

VERSION: 1.0
LAST UPDATED: 2026-07-15
AUTHOR: Arjun Divecha

DESCRIPTION:
    Builds the MARKET REGIME - EARLY WARNING SYSTEM section of the daily
    report's data package. The Early Warning System (sibling project at
    "A Working/Early Warning") classifies the US equity market into
    IN / TRANSITION / OUT regimes from 12 weighted signals across four
    causal families (A structural valuation/credit, B macro/credit
    deterioration, C market internals/trend, D fast stress/sentiment),
    plus zero-weight shadow signals under evaluation. This module renders:

      1. The current reading: regime, escalation ladder color, composite
         danger z-score and expanding percentile, diffusion index, and
         which signals are flagged.
      2. The last 8 months of regime history (trend context - e.g. the
         one-month OUT flip in 2026-03).
      3. The full per-signal panel (z, flag, family, leader/confirmer role).

    Freshness has two distinct layers, both surfaced explicitly:
      - Signal month lag (BY DESIGN): monthly data with publication lags.
      - Dashboard staleness (A PROBLEM): if the weekly rebuild has not run
        for more than SETTINGS["ews_stale_days"] days, the section renders
        anyway but is loudly marked STALE with trailing '*' (house rule:
        stale-with-asterisk, never a silent drop, never "n/a").

    All reads are read-only. Any failure returns None with a loud console
    message - the report then ships without the section (additive-only),
    exactly like the ASADO snapshot pattern in asado.py.

DEPENDENCIES:
    - stdlib only (json, datetime, pathlib)

USAGE:
    from ews import build_ews_section
    section_md = build_ews_section(asof="2026-07-15")   # None on any failure
=============================================================================
"""

import json
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PATHS, SETTINGS

# Human names for the ladder rungs, in escalation order (for the caption).
LADDER_ORDER = ["GREEN", "YELLOW", "ORANGE", "RED", "BLACK"]
TREND_MONTHS = 8


def _fmt_z(v) -> str:
    return f"{v:+.2f}" if isinstance(v, (int, float)) else "—"


def _ordinal_pct(p) -> str:
    """0.518 -> '52nd percentile' (expanding-window percentile)."""
    if not isinstance(p, (int, float)):
        return "—"
    n = int(round(p * 100))
    suffix = ("th" if 10 <= n % 100 <= 20
              else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th"))
    return f"{n}{suffix} percentile"


def build_ews_section(asof: str = None, path: Path = None) -> str | None:
    """Render the Early Warning System markdown section, or None on failure."""
    data_path = path or PATHS["ews_data"]
    try:
        if not data_path.exists():
            print(f"  ⚠️  EWS section SKIPPED: {data_path} not found")
            return None
        d = json.loads(data_path.read_text())
        return _build(d, asof)
    except Exception as e:
        print(f"  ⚠️  EWS section SKIPPED ({type(e).__name__}): {e}")
        return None


def _build(d: dict, asof: str = None) -> str:
    cur = d["current"]
    generated = d.get("generated", "")

    # Dashboard staleness: the weekly rebuild has stopped running.
    asof_d = (datetime.strptime(asof, "%Y-%m-%d").date() if asof
              else date.today())
    stale = False
    age_days = None
    if generated:
        gen_d = datetime.strptime(generated, "%Y-%m-%d").date()
        age_days = (asof_d - gen_d).days
        stale = age_days > SETTINGS.get("ews_stale_days", 12)
    star = "*" if stale else ""

    sig_by_id = {s["id"]: s for s in d.get("signals", [])}

    def sig_label(sid: str) -> str:
        s = sig_by_id.get(sid)
        return f"{sid} {s['name']}" if s else sid

    lines = [
        f"\n## MARKET REGIME — EARLY WARNING SYSTEM "
        f"(monthly; signal month {cur['date'][:7]}; "
        f"dashboard generated {generated or '—'})",
    ]
    if stale:
        lines.append(
            f"** EWS DASHBOARD IS STALE ** — last rebuilt {age_days} days "
            f"before the as-of date (weekly update has not run); all values "
            f"below are last-available, marked with *.")

    # ---- 1. current reading -------------------------------------------
    flags_on = cur.get("flags_on") or []
    lines += [
        "",
        f"- Regime: **{cur['state']}{star}** | Escalation ladder: "
        f"**{cur['ladder']}{star}** "
        f"(rungs: {' < '.join(LADDER_ORDER)})",
        f"- Composite danger z: {_fmt_z(cur.get('composite'))}{star} "
        f"({_ordinal_pct(cur.get('composite_pctile'))}, expanding window)",
        f"- Diffusion: {cur.get('n_flags', 0)}/{cur.get('n_avail', 0)} "
        f"weighted signals flagged{star}",
        f"- Flags on: "
        + (", ".join(sig_label(f) for f in flags_on) if flags_on else "none"),
    ]

    # ---- 2. regime trend (last N months) ------------------------------
    dates = d.get("dates") or []
    if dates:
        n = min(TREND_MONTHS, len(dates))
        lines += [
            f"\n### REGIME TREND (last {n} signal months)",
            "",
            "| Month | Regime | Ladder | Composite z | Pctile | Diffusion |",
            "|---|---|---|---|---|---|",
        ]

        def _at(key, i):
            arr = d.get(key) or []
            return arr[i] if i < len(arr) else None

        for i in range(len(dates) - n, len(dates)):
            pct = _at("composite_pctile", i)
            dif = _at("diffusion", i)
            lines.append(
                f"| {dates[i]} | {_at('state', i) or '—'} "
                f"| {_at('ladder', i) or '—'} "
                f"| {_fmt_z(_at('composite', i))} "
                f"| {f'{pct:.0%}' if isinstance(pct, (int, float)) else '—'} "
                f"| {f'{dif:.0%}' if isinstance(dif, (int, float)) else '—'} |")

    # ---- 3. per-signal panel ------------------------------------------
    fam_names = d.get("families") or {}
    signals = d.get("signals") or []
    if signals:
        lines += [
            "\n### SIGNAL PANEL (z = expanding z-score; FLAG = past its "
            "danger threshold; shadow signals carry zero weight)",
            "",
            "| Signal | Family | Role | z | Flag |",
            "|---|---|---|---|---|",
        ]
        for s in signals:
            c = s.get("current") or {}
            fam = fam_names.get(s.get("family"), s.get("family", "—"))
            role = s.get("role", "—")
            shadow = " (shadow)" if s.get("shadow") else ""
            flag = ("FLAG" if c.get("flag")
                    else "—" if c.get("flag") is None else "ok")
            lines.append(
                f"| {s['id']} {s.get('name', '')}{shadow} | {fam} | {role} "
                f"| {_fmt_z(c.get('z'))}{star} | {flag}{star} |")

    lines.append(
        "\n_The Early Warning System is a MONTHLY 12-signal regime "
        "classifier (long-run occupancy ~70% IN / 15% TRANSITION / 15% OUT; "
        "in backtest it caught 11 of 13 post-1965 drawdowns ≥15% with a "
        "full OUT). Signals publish with lags, so the signal month trailing "
        "today by 1-2 months is normal and by design. Treat it as "
        "slow-moving risk-posture context, not a daily trading signal._")

    print(f"  Early Warning: {cur['state']} / {cur['ladder']} "
          f"(signal month {cur['date'][:7]}"
          + (f", STALE dashboard {age_days}d old" if stale else "") + ")")
    return "\n".join(lines)


if __name__ == "__main__":
    section = build_ews_section()
    print(section if section else "(no EWS section)")
