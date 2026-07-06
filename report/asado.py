#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: asado.py
=============================================================================

INPUT FILES:
    - /Users/arjundivecha/Dropbox/AAA Backup/A Working/ASADO/Data/asado.duckdb
      (READ-ONLY: the ASADO 34-country macro/equity warehouse — t2 factor
      signals, GDELT news sentiment, daily factor-portfolio returns)

OUTPUT FILES:
    (none - returns a rendered markdown snapshot string to fable_desk.py)

VERSION: 1.0
LAST UPDATED: 2026-07-06
AUTHOR: Arjun Divecha

DESCRIPTION:
    Builds the ASADO country-signal snapshot that feeds Fable's Desk (the
    judgment-based second pass of the daily report). Pulls, from the latest
    available dates in the warehouse:

      1. A 34-country composite table from t2_factors_daily: cross-sectional
         valuation composite, long-horizon momentum, RSI, currency and
         country-risk signals (all _CS z-scores; higher = more attractive
         after ASADO's universal sign convention).
      2. GDELT news tone/attention extremes from gdelt_factors_daily
         (which countries the world's news flow is loving/hating today).
      3. The best/worst t2 country-selection factors over the last 5
         sessions from factor_returns_daily (what style of country picking
         is currently being paid).

    All queries are read-only (duckdb read_only=True). Any failure returns
    None with a loud message - the Desk then runs without ASADO data and
    says so, and the deterministic report is never affected (additive-only).

DEPENDENCIES:
    - duckdb
    - pandas

USAGE:
    from asado import build_asado_snapshot
    snapshot_md = build_asado_snapshot()   # None on any failure
=============================================================================
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PATHS

# Valuation composite legs: cross-sectional z-scores that all read
# "higher = cheaper/more attractive" after ASADO's sign convention.
VALUE_LEGS = ["Earnings Yield_CS", "Shiller PE_CS", "Best PE _CS",
              "Best PBK_CS"]
MOMENTUM_VAR = "120-5DTR_CS"       # 6m-ex-1w momentum, the classic
SIGNAL_VARS = {
    "RSI14_CS": "rsi",             # short-term stretch (higher = oversold = attractive)
    "Currency_CS": "fx",
    "Bloom Country Risk_CS": "risk",
}


def _fmt(v) -> str:
    return f"{v:+.2f}" if pd.notna(v) else "—"


def build_asado_snapshot() -> str | None:
    """Render the ASADO snapshot markdown for the Desk prompt, or None."""
    db_path = PATHS["asado_db"]
    if not db_path.exists():
        print(f"  ⚠️  ASADO snapshot SKIPPED: {db_path} not found")
        return None
    try:
        import duckdb
    except ImportError:
        print("  ⚠️  ASADO snapshot SKIPPED: duckdb not installed "
              "(pip3 install duckdb)")
        return None
    try:
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            return _build(con)
        finally:
            con.close()
    except Exception as e:
        print(f"  ⚠️  ASADO snapshot SKIPPED ({type(e).__name__}): {e}")
        return None


def _build(con) -> str:
    t2_max = con.execute(
        "SELECT max(date) FROM t2_factors_daily").fetchone()[0]
    gdelt_max = con.execute(
        "SELECT max(date) FROM gdelt_factors_daily").fetchone()[0]
    fr_max = con.execute(
        "SELECT max(date) FROM factor_returns_daily").fetchone()[0]

    # ---- 1. country composite table ----------------------------------
    value_list = ", ".join(f"'{v}'" for v in VALUE_LEGS)
    sig_cases = ",\n".join(
        f"  round(avg(CASE WHEN variable='{var}' THEN value END), 2) "
        f"AS {alias}"
        for var, alias in SIGNAL_VARS.items())
    rows = con.execute(f"""
        SELECT country,
          round(avg(CASE WHEN variable IN ({value_list}) THEN value END), 2)
            AS value_cs,
          round(avg(CASE WHEN variable='{MOMENTUM_VAR}' THEN value END), 2)
            AS mom_cs,
        {sig_cases}
        FROM t2_factors_daily WHERE date = ?
        GROUP BY country ORDER BY value_cs DESC
    """, [t2_max]).fetchall()

    lines = [
        f"### ASADO COUNTRY SIGNALS (t2_factors_daily as of {t2_max}; "
        f"cross-sectional z-scores across 34 markets; HIGHER = MORE "
        f"ATTRACTIVE on every column per ASADO's sign convention; 0 can "
        f"mean 'no signal')",
        "",
        "| Country | Value (composite) | Momentum 6m-1w | RSI stretch "
        "| Currency | Country risk |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r[0]} | {_fmt(r[1])} | {_fmt(r[2])} | "
                     f"{_fmt(r[3])} | {_fmt(r[4])} | {_fmt(r[5])} |")
    lines.append(
        "_Value composite = mean of Earnings Yield / Shiller PE / Best PE / "
        "P-B cross-sectional z-scores. 'China' trades as ChinaA + ChinaH; "
        "'U.S.' splits into U.S. + NASDAQ + US SmallCap._")

    # ---- 2. GDELT news tone/attention extremes -----------------------
    gd = con.execute("""
        SELECT country,
          round(avg(CASE WHEN variable='tone_mean_CS' THEN value END), 2)
            AS tone,
          round(avg(CASE WHEN variable='sentiment_x_attention_CS'
                    THEN value END), 2) AS sent_x_attn,
          round(avg(CASE WHEN variable='attention_shock_CS' THEN value END),
                2) AS attn_shock
        FROM gdelt_factors_daily WHERE date = ?
        GROUP BY country
        HAVING tone IS NOT NULL
        ORDER BY tone
    """, [gdelt_max]).fetchall()
    if gd:
        lines += [
            "",
            f"### ASADO GDELT NEWS PULSE (as of {gdelt_max}; z-scores; "
            f"tone = how positively the world's news covers the country; "
            f"attention shock = unusual news volume)",
            "",
            "| Country | News tone | Sentiment x attention "
            "| Attention shock |",
            "|---|---|---|---|",
        ]
        worst, best = gd[:5], gd[-5:][::-1]
        for label, grp in (("(most negative)", worst),
                           ("(most positive)", best)):
            for r in grp:
                lines.append(f"| {r[0]} {label if r is grp[0] else ''} | "
                             f"{_fmt(r[1])} | {_fmt(r[2])} | {_fmt(r[3])} |")

    # ---- 3. which country-selection factors are being paid -----------
    fr = con.execute("""
        SELECT factor, round(sum(value), 2) AS ret_5d
        FROM factor_returns_daily
        WHERE source = 't2_optimizer_daily'
          AND date > (SELECT max(date) - INTERVAL 7 DAY
                      FROM factor_returns_daily)
        GROUP BY factor ORDER BY ret_5d DESC
    """).fetchall()
    if fr:
        lines += [
            "",
            f"### ASADO FACTOR P&L — last 5 sessions to {fr_max} "
            f"(long top-20%-of-countries portfolios, t2 signals; % return; "
            f"what style of country selection is currently being paid)",
            "",
            "| Best factors | 5d % | Worst factors | 5d % |",
            "|---|---|---|---|",
        ]
        top, bot = fr[:8], fr[-8:][::-1]
        for (bf, bv), (wf, wv) in zip(top, bot):
            lines.append(f"| {bf} | {bv:+.2f} | {wf} | {wv:+.2f} |")

    return "\n".join(lines)


if __name__ == "__main__":
    snap = build_asado_snapshot()
    print(snap if snap else "(no snapshot)")
