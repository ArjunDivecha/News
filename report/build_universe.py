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

VERSION: 1.2
LAST UPDATED: 2026-06-30
AUTHOR: Arjun Divecha

DESCRIPTION:
    One-time (re-runnable) tool that consolidates the Final 1000 list and
    the ETF migration map into a single universe file keyed by Yahoo
    Finance ticker. Where several Bloomberg indices map to the same ETF,
    they collapse into ONE row (the ETF is the asset now); the original
    Bloomberg tickers are preserved in 'proxied_tickers' for audit.
    The 15 factor ETFs are guaranteed present and flagged.

    v1.1 (superseded): filtered out all source != 'ETF' rows entirely.
    This was too blunt -- audit found that 60 Goldman/Bloomberg-sourced
    rows had ALREADY been proxy-mapped by etf_map.py onto real, liquid,
    Yahoo-viable ETFs (SPY, VTI, the full SPDR sector suite, MTUM, VLUE,
    IWF, VNQ, HYG, IEF, GDX, DBMF, DBV, etc.) with NO duplicate row
    elsewhere in the source file. Dropping by source would have silently
    removed all of them from the universe -- real breadth loss, not
    cleanup.

    v1.2: keep-and-relabel instead of drop. ETF_OVERRIDES below supplies
    corrected name/tier1/tier2/tags for each of these 59 tickers, hand-
    verified against what the real fund actually is (not the old GS-
    basket/Bloomberg-index concept it used to be proxied from). A few
    tier1 corrections ride along: GDX/CRAK/OIH/XME/XOP move from
    Commodities to Equities (they hold mining/energy COMPANY stocks, not
    the physical commodity/futures -- same error pattern already found
    in the direct-ETF rows for COPX/SILJ/GUNR). BITW/ETHA/IBIT are kept
    under Commodities/"Digital Assets / Alternative Commodities", matching
    the tier2 value the original IBIT row already used, consistent with
    how physical-metal trust ETFs (AAAU, PALL) are classified.
    Two tickers are dropped outright, both unrelated to source quality:
    TBT (2x inverse Treasury, per the leveraged/inverse exclusion policy)
    and GFOF (Grayscale Future of Finance, which stopped trading in 2024
    and has no viable yfinance data).

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

# Corrected classification for tickers that were only present via a Goldman/
# Bloomberg proxy row. yf_ticker is real and Yahoo-viable in every case.
# tier1/tier2/tags reflect what the real fund actually is, hand-verified.
ETF_OVERRIDES = {
    "AAXJ": ("Equities", "Country/Regional", "Equity, Asia, EM, Passive, Broad-Based"),
    "AIQ":  ("Equities", "Thematic/Factor", "Equity, Global, Tech, Thematic, Passive"),
    "ASEA": ("Equities", "Country/Regional", "Equity, Asia, EM, APAC, Passive"),
    "BITW": ("Commodities", "Digital Assets / Alternative Commodities", "Commodity, Global, Cryptocurrency, Passive"),
    "BLOK": ("Equities", "Thematic/Factor", "Equity, Global, Tech, Thematic, Active"),
    "COWZ": ("Equities", "Thematic/Factor", "Equity, US, Quality, Factor-Based, Passive"),
    "CRAK": ("Equities", "Sector Indices", "Equity, Global, Energy, Passive"),
    "DBMF": ("Alternative / Synthetic", "Quant/Style Baskets", "Multi-Asset, Global, Active, Quantitative, Trend-Following"),
    "DBV":  ("Currencies (FX)", "Majors", "FX, Global, Carry/Value Factors, Passive"),
    "DVY":  ("Equities", "Thematic/Factor", "Equity, US, Dividend, Value, Passive"),
    "DXJ":  ("Equities", "Country/Regional", "Equity, Japan, Passive, Dividend-Weighted"),
    "EMSH": ("Fixed Income", "Sovereign Bonds", "Credit, EM, Short (<2Y), Passive"),
    "EPP":  ("Equities", "Country/Regional", "Equity, APAC, Developed, Passive"),
    "ETHA": ("Commodities", "Digital Assets / Alternative Commodities", "Commodity, Global, Cryptocurrency, Passive"),
    "EUFN": ("Equities", "Sector Indices", "Equity, Europe, Financials, Passive"),
    "EWG":  ("Equities", "Country/Regional", "Equity, Europe, Germany, Passive"),
    "EWH":  ("Equities", "Country/Regional", "Equity, Asia, Hong Kong, Passive"),
    "EZU":  ("Equities", "Country/Regional", "Equity, Europe, Developed, Passive"),
    "FDN":  ("Equities", "Sector Indices", "Equity, US, Tech, Passive"),
    "GDX":  ("Equities", "Sector Indices", "Equity, Global, Materials, Passive"),
    "HYDR": ("Equities", "Thematic/Factor", "Equity, Global, Energy, Thematic, Passive"),
    "HYG":  ("Fixed Income", "Corporate Credit", "Credit, US, High Yield, Passive"),
    "IBIT": ("Commodities", "Digital Assets / Alternative Commodities", "Commodity, Global, Cryptocurrency, Passive"),
    "IEF":  ("Fixed Income", "Sovereign Bonds", "Credit, US, Medium (2-10Y), Passive"),
    "IGLB": ("Fixed Income", "Corporate Credit", "Credit, US, Long (>10Y), Passive"),
    "IGV":  ("Equities", "Sector Indices", "Equity, US, Tech, Passive"),
    "IHI":  ("Equities", "Sector Indices", "Equity, US, Healthcare, Passive"),
    "INDA": ("Equities", "Country/Regional", "Equity, EM, India, Passive"),
    "INFL": ("Equities", "Thematic/Factor", "Equity, Global, Active, Inflation-Sensitive, Thematic"),
    "ISHG": ("Fixed Income", "Sovereign Bonds", "Credit, Global, Developed, Short (<2Y), Passive"),
    "ITB":  ("Equities", "Sector Indices", "Equity, US, Industrials, Passive"),
    "IWF":  ("Equities", "Thematic/Factor", "Equity, US, Growth, Large-cap, Passive"),
    "MCHI": ("Equities", "Country/Regional", "Equity, EM, China, Passive"),
    "MTUM": ("Equities", "Thematic/Factor", "Equity, US, Momentum, Factor-Based, Passive"),
    "MUB":  ("Fixed Income", "Sovereign Bonds", "Credit, US, Municipal, Medium (2-10Y), Passive"),
    "OIH":  ("Equities", "Sector Indices", "Equity, Global, Energy, Passive"),
    "PHO":  ("Equities", "Thematic/Factor", "Equity, US, Infrastructure, Thematic, Passive"),
    "SPY":  ("Equities", "Global Indices", "Equity, US, Large-cap, Passive, Broad Market"),
    "SVXY": ("Volatility / Risk Premia", "Vol Indices", "Volatility, US, Short, Quantitative"),
    "VCIT": ("Fixed Income", "Corporate Credit", "Credit, US, Medium (2-10Y), Passive"),
    "VGK":  ("Equities", "Country/Regional", "Equity, Europe, Developed, Passive"),
    "VHT":  ("Equities", "Sector Indices", "Equity, US, Healthcare, Passive"),
    "VIXY": ("Volatility / Risk Premia", "Vol Indices", "Volatility, US, Long, Quantitative"),
    "VLUE": ("Equities", "Thematic/Factor", "Equity, US, Value, Factor-Based, Passive"),
    "VNQ":  ("Equities", "Real Estate / REITs", "Equity, US, Real Estate, Passive"),
    "VNQI": ("Equities", "Real Estate / REITs", "Equity, Global, Real Estate, Passive"),
    "VPL":  ("Equities", "Country/Regional", "Equity, APAC, Developed, Passive"),
    "VTI":  ("Equities", "Global Indices", "Equity, US, Broad Market, Passive"),
    "VTV":  ("Equities", "Thematic/Factor", "Equity, US, Value, Factor-Based, Passive"),
    "XLC":  ("Equities", "Sector Indices", "Equity, US, Communication Services, Passive"),
    "XLE":  ("Equities", "Sector Indices", "Equity, US, Energy, Passive"),
    "XLI":  ("Equities", "Sector Indices", "Equity, US, Industrials, Passive"),
    "XLP":  ("Equities", "Sector Indices", "Equity, US, Consumer, Defensive, Passive"),
    "XLU":  ("Equities", "Sector Indices", "Equity, US, Utilities, Defensive, Passive"),
    "XLV":  ("Equities", "Sector Indices", "Equity, US, Healthcare, Passive"),
    "XLY":  ("Equities", "Sector Indices", "Equity, US, Consumer, Passive"),
    "XME":  ("Equities", "Sector Indices", "Equity, US, Materials, Passive"),
    "XOP":  ("Equities", "Sector Indices", "Equity, US, Energy, Passive"),
}
# Dropped outright:
#   TBT  -- 2x inverse Treasury, per leveraged/inverse exclusion policy.
#   GFOF -- Grayscale Future of Finance: stopped trading in 2024, no viable
#           yfinance data. (Both yf_tickers are/were real; exclusion is by
#           policy / delisting, not source quality.)
DROP_TICKERS = {"TBT", "GFOF"}


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
    relabeled = []
    for _, r in df.iterrows():
        bt = str(r["Bloomberg_Ticker"]).strip()
        yf = get_yf_ticker(bt)
        if yf is None:
            dropped.append(bt)
            continue

        source = r["source"]
        name = r["Name"]
        description = r.get("Long_Description")
        tier1 = r["category_tier1"]
        tier2 = r["category_tier2"]
        tags = r.get("category_tags")

        if source != "ETF":
            if yf in DROP_TICKERS:
                dropped.append(bt)
                continue
            override = ETF_OVERRIDES.get(yf)
            if override is not None:
                tier1, tier2, tags = override
                source = "ETF"  # now correctly reflects a real, direct ETF
                description = (f"{description} [relabeled from Goldman/"
                                f"Bloomberg-proxy source -- see ETF_OVERRIDES]"
                                if isinstance(description, str) else description)
                relabeled.append(yf)
            # else: non-ETF source with no override on file -- keep as-is,
            # flagged via 'source' column for manual review later.

        rows.append({
            "yf_ticker": yf,
            "bloomberg_ticker": bt,
            "name": name,
            "description": description,
            "tier1": tier1,
            "tier2": tier2,
            "tags": tags,
            "source": source,
            "tracking_score": get_tracking_score(bt) if "Equity" not in bt else 5,
        })
    mapped = pd.DataFrame(rows)
    print(f"[2/4] Mapped to yfinance: {len(mapped)} rows, dropped {len(dropped)} "
          f"(no ETF replacement or explicit exclusion)")
    print(f"      Relabeled {len(relabeled)} Goldman/Bloomberg-proxy rows to their "
          f"real ETF identity: {sorted(relabeled)}")

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
