#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: build_universe.py
=============================================================================

INPUT FILES:
    - /Users/arjundivecha/Dropbox/AAA Backup/A Working/News/
        Step 2 Data Processing - Final1000/Final 1000 Asset Master List.xlsx
      The curated 970-asset universe (Bloomberg tickers + 3-tier taxonomy).
    - report/etf_map.py
      Bloomberg-index -> ETF mapping validated during the ETF migration.

OUTPUT FILES:
    - /Users/arjundivecha/Dropbox/AAA Backup/A Working/News/data/universe.xlsx
      ONE clean universe keyed by yfinance ticker. Columns:
      yf_ticker, name, description, tier1, tier2, tags, source,
      tracking_score, is_factor, factor_name, proxied_tickers.

VERSION: 1.0
LAST UPDATED: 2026-06-09
AUTHOR: Arjun Divecha

DESCRIPTION:
    One-time (re-runnable) tool that consolidates the Final 1000 list and
    the ETF migration map into a single universe file keyed by Yahoo
    Finance ticker. Where several Bloomberg indices map to the same ETF,
    they collapse into ONE row (the ETF is the asset now); the original
    Bloomberg tickers are preserved in 'proxied_tickers' for audit.
    The 15 factor ETFs are guaranteed present and flagged.

DEPENDENCIES:
    - pandas, openpyxl

USAGE:
    python report/build_universe.py
=============================================================================
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PATHS, FACTORS, ROOT_DIR, ensure_dirs
from etf_map import get_yf_ticker, get_tracking_score

FINAL_1000 = ROOT_DIR / "Step 2 Data Processing - Final1000" / "Final 1000 Asset Master List.xlsx"


def build_universe() -> pd.DataFrame:
    print("=" * 70)
    print("BUILD UNIVERSE  (Final 1000 + ETF migration map -> universe.xlsx)")
    print("=" * 70)

    if not FINAL_1000.exists():
        raise FileNotFoundError(f"Final 1000 list not found: {FINAL_1000}")

    df = pd.read_excel(FINAL_1000)
    print(f"\n[1/4] Loaded Final 1000: {len(df)} assets "
          f"({df['source'].value_counts().to_dict()})")

    # Map every Bloomberg ticker to a yfinance ticker (None = dropped)
    rows = []
    dropped = []
    for _, r in df.iterrows():
        bt = str(r["Bloomberg_Ticker"]).strip()
        yf = get_yf_ticker(bt)
        if yf is None:
            dropped.append(bt)
            continue
        rows.append({
            "yf_ticker": yf,
            "bloomberg_ticker": bt,
            "name": r["Name"],
            "description": r.get("Long_Description"),
            "tier1": r["category_tier1"],
            "tier2": r["category_tier2"],
            "tags": r.get("category_tags"),
            "source": r["source"],
            "tracking_score": get_tracking_score(bt) if "Equity" not in bt else 5,
        })
    mapped = pd.DataFrame(rows)
    print(f"[2/4] Mapped to yfinance: {len(mapped)} rows, dropped {len(dropped)} "
          f"(no ETF replacement)")

    # Collapse duplicates: the ETF is the asset. Direct ETF rows (source=='ETF')
    # win over proxy rows; among proxies keep the highest tracking score.
    mapped["is_direct"] = (mapped["source"] == "ETF").astype(int)
    mapped = mapped.sort_values(["yf_ticker", "is_direct", "tracking_score"],
                                ascending=[True, False, False])
    proxied = (mapped.groupby("yf_ticker")["bloomberg_ticker"]
               .apply(lambda s: ", ".join(s.iloc[1:]) if len(s) > 1 else ""))
    uni = mapped.drop_duplicates("yf_ticker", keep="first").set_index("yf_ticker")
    uni["proxied_tickers"] = proxied
    uni = uni.drop(columns=["is_direct"]).reset_index()
    n_collapsed = len(mapped) - len(uni)
    print(f"[3/4] Collapsed {n_collapsed} duplicate ETF mappings -> {len(uni)} unique assets")

    # Guarantee the 15 factor ETFs exist and flag them
    yf_to_factor = {v: k for k, v in FACTORS.items()}
    uni["is_factor"] = uni["yf_ticker"].map(lambda t: int(t in yf_to_factor))
    uni["factor_name"] = uni["yf_ticker"].map(yf_to_factor)
    missing_factors = [t for t in FACTORS.values() if t not in set(uni["yf_ticker"])]
    for t in missing_factors:
        uni = pd.concat([uni, pd.DataFrame([{
            "yf_ticker": t,
            "bloomberg_ticker": "",
            "name": f"{yf_to_factor[t]} factor ETF",
            "description": f"Factor proxy ETF for {yf_to_factor[t]}",
            "tier1": "Factor",
            "tier2": "Factor ETF",
            "tags": "",
            "source": "Factor",
            "tracking_score": 5,
            "proxied_tickers": "",
            "is_factor": 1,
            "factor_name": yf_to_factor[t],
        }])], ignore_index=True)
    if missing_factors:
        print(f"      Added {len(missing_factors)} factor ETFs not in Final 1000: "
              f"{missing_factors}")

    ensure_dirs()
    out = PATHS["universe"]
    uni.to_excel(out, index=False)
    print(f"\n[4/4] Wrote {len(uni)} assets -> {out}")
    print(f"      Tier-1 distribution: {uni['tier1'].value_counts().to_dict()}")
    print(f"      Factor ETFs flagged: {int(uni['is_factor'].sum())}")
    return uni


if __name__ == "__main__":
    build_universe()
