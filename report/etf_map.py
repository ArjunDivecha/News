#!/usr/bin/env python3
"""
=============================================================================
ETF MIGRATION MAPPING
=============================================================================

INPUT FILES:
- None (mapping is hardcoded from research)

OUTPUT FILES:
- None (used as a module by other scripts)

VERSION: 1.0.0
CREATED: 2026-05-27

PURPOSE:
Maps Bloomberg Index tickers (Bloomberg indices + Goldman Sachs baskets)
to Yahoo Finance ETF tickers. Used to eliminate the Bloomberg Terminal
dependency for daily data retrieval.

Of 297 Bloomberg-dependent tickers:
 - ~265 have ETF replacements (tracking 2-5 on a 5-point scale)
 - ~32 have no viable ETF replacement and are dropped

USAGE:
    from etf_migration_map import BLOOMBERG_TO_ETF, FACTOR_ETF_MAP, get_yf_ticker
=============================================================================
"""

from typing import Dict, Optional, Tuple

# =============================================================================
# FACTOR TICKERS (used for beta computation in the correlation engine)
# These are the 15 tickers that drive the entire factor model.
# All have liquid, high-AUM ETF proxies.
# =============================================================================

FACTOR_ETF_MAP: Dict[str, str] = {
    "SPX":         "SPY",       # S&P 500
    "Russell2000": "IWM",       # Russell 2000
    "Nasdaq100":   "QQQ",       # Nasdaq 100
    "Value":       "IWD",       # Russell 1000 Value
    "Growth":      "IWF",       # Russell 1000 Growth
    "EAFE":        "EFA",       # MSCI EAFE
    "EM":          "EEM",       # MSCI Emerging Markets
    "HY_Credit":   "HYG",       # US Corporate High Yield
    "Treasuries":  "BNDX",      # Global Aggregate Treasuries
    "TIPS":        "TIP",       # US TIPS
    "Commodities": "BCI",       # Bloomberg Commodity TR
    "Agriculture": "DBA",       # Agriculture commodities
    "Crypto":      "IBIT",      # Bitcoin (spot ETF)
    "REIT_US":     "VNQ",       # US REITs
    "REIT_Global": "VNQI",      # Global ex-US REITs
}

# =============================================================================
# BLOOMBERG INDEX → ETF MAPPING
# Maps old Bloomberg_Ticker (from Final 1000 list) to Yahoo Finance ticker.
# Format: "OLD_TICKER": ("YF_TICKER", tracking_score, "short_description")
#
# tracking_score: 1-5 (5 = near-identical index tracking)
# None entry means: DROP this ticker (no viable ETF replacement)
# =============================================================================

BLOOMBERG_TO_ETF: Dict[str, Optional[Tuple[str, int, str]]] = {

    # =========================================================================
    # BLOOMBERG INDICES (72)
    # =========================================================================

    # --- Commodities: Agriculture ---
    "BBFAT Index":    ("DBA",  2, "Ag commodities proxy for biofuels index"),
    "BCOGV10E Index": None,  # Vol-target strategy, no ETF
    "BCOMCA Index":   ("BCI",  2, "Broad commodity proxy for carbon-tilted"),
    "BCOM Index":     ("BCI",  5, "BCI directly tracks BCOM TR"),
    "BCOMTR Index":   ("BCI",  5, "BCI directly tracks BCOM TR"),

    # --- Commodities: Energy & Metals ---
    "BERY Index":     ("CERY", 5, "CERY tracks Enhanced Roll Yield"),
    "BERYTR Index":   ("CERY", 5, "CERY tracks Enhanced Roll Yield TR"),
    "BEMEWER Index":  ("COMT", 2, "Broad commodity dynamic roll proxy"),
    "BCOMF3 Index":   ("BCI",  3, "Broad commodity proxy for 3M forward"),
    "BCOMF3T Index":  ("BCI",  3, "Broad commodity proxy for 3M forward TR"),
    "BCOMRS Index":   ("CMDY", 5, "CMDY tracks Roll Select Commodity"),
    "BCOMRST Index":  ("CMDY", 5, "CMDY tracks Roll Select Commodity TR"),

    # --- Commodities: Metals ---
    "BTMAT Index":    ("REMX", 3, "Rare earth/strategic metals equity proxy"),

    # --- Commodities: Multi-Commodity ---
    "BCOMCAT Index":  ("BCI",  2, "Broad commodity proxy for carbon-tilted TR"),

    # --- Commodities: Crypto ---
    "BITCOIN Index":  ("IBIT", 5, "Spot Bitcoin ETF"),
    "ETHEREUM Index": ("ETHA", 5, "Spot Ethereum ETF"),
    "BGCI Index":     ("BITW", 4, "Multi-crypto index ETF"),

    # --- Equities: APAC Regional ---
    "APAC Index":     ("VPL",  4, "FTSE Developed Asia Pacific"),
    "APACLS Index":   ("VPL",  3, "VPL is large/mid only"),
    "APACD Index":    ("VPL",  4, "APAC developed markets"),
    "APDLS Index":    ("VPL",  3, "APAC DM large/mid/small"),
    "APACELS Index":  ("GMF",  3, "EM Asia Pacific"),
    "APACE Index":    ("GMF",  3, "APAC EM large & mid"),
    "APACXJ Index":   ("EPP",  4, "MSCI Pacific ex Japan"),
    "APDXJLS Index":  ("EPP",  3, "APAC DM ex Japan"),
    "APEJ Index":     ("AAXJ", 4, "MSCI All Country Asia ex Japan"),
    "APEJLS Index":   ("AAXJ", 3, "Asia ex Japan large/mid/small"),
    "ASIAD Index":    ("VPL",  4, "Asia developed markets"),
    "ASIADLS Index":  ("VPL",  3, "Asia DM large/mid/small"),
    "ASIAEM Index":   ("GMF",  3, "Asia EM large & mid"),
    "ASIAELS Index":  ("GMF",  3, "Asia EM large/mid/small"),
    "ASEAN Index":    ("ASEA", 4, "FTSE Southeast Asia"),
    "AE Index":       ("UAE",  5, "MSCI UAE direct match"),
    "ASIALS Index":   ("VPL",  3, "Asia broad coverage"),

    # --- Equities: Real Estate / REITs ---
    "APLSRTT Index":  ("VNQI", 3, "Global ex-US REITs, ~50-65% APAC"),
    "ADLSRTT Index":  ("VNQI", 2, "Developed intl REITs"),
    "ADREITT Index":  ("VNQI", 2, "Asia DM REITs"),
    "AEJLSRTT Index": ("VNQI", 2, "APAC ex Japan REITs"),
    "AEJREITT Index": ("VNQI", 2, "APAC ex Japan REITs"),
    "APREITT Index":  ("VNQI", 3, "APAC REITs"),
    "APDLSRTT Index": ("VNQI", 3, "APAC DM REITs"),
    "APDREITT Index": ("VNQI", 3, "APAC DM REITs"),

    # --- Equities: Global ---
    "AGGET Index":    ("VTI",  4, "US total stock market"),

    # --- Fixed Income: Corporate Credit ---
    "I00148US Index": ("LQD",  1, "IG corporate proxy for niche ABS"),
    "I00155US Index": ("IGLB", 3, "Long IG corporate bond"),

    # --- Fixed Income: Sovereign Bonds ---
    "I00762US Index": ("MUB",  2, "Broad muni bond proxy"),
    "I02526EU Index": ("ISHG", 3, "Short-term intl treasury"),

    # --- Fixed Income: Yield Curves ---
    "I00091US Index": ("TIP",  3, "Broad TIPS proxy for 5+ yr"),
    "BEM5TRUU Index": ("EMSH", 4, "Short-term EM USD bond"),
    "BTSISW3M Index": None,  # SEK cash deposit, no ETF
    "I00157US Index": ("VCIT", 2, "Intermediate corporate bond proxy"),

    # --- Currencies ---
    "I02705EU Index": None,  # Norwegian Krone, no ETF
    "BGSFXC Index":   ("DBV",  3, "G10 FX carry strategy"),
    "BGSFXV Index":   None,  # FX value strategy, no ETF
    "BGSFXT Index":   ("DBMF", 2, "Managed futures proxy for FX trend"),

    # --- Multi-Asset / Thematic ---
    "BGSXAT Index":   ("DBMF", 3, "Managed futures proxy"),
    "BGSXAV Index":   None,  # Cross-asset value, no ETF
    "BM64V20T Index": ("AOR",  2, "60/40 allocation proxy"),
    "BPRAT Index":    None,  # Proprietary thematic, no ETF
    "BINFLST Index":  ("INFL", 3, "Inflation-beneficiary equities"),
    "BWAAT Index":    ("PHO",  3, "Water resources ETF"),
    "BHYAT Index":    ("HYDR", 3, "Hydrogen ETF"),
    "BGFOFT Index":   ("GFOF", 5, "Same index tracker"),
    "BFDAT Index":    ("BLOK", 3, "Blockchain/digital infra"),
    "BBDAT Index":    None,  # Biodiversity, no US ETF

    # --- Volatility / Risk Premia ---
    "BCOEV10E Index": None,  # Energy vol-target, no ETF
    "BCOMV10E Index": None,  # Commodity vol-target, no ETF
    "BCOMV10T Index": None,  # Commodity vol-target TR, no ETF
    "BCOPV10E Index": None,  # Precious metals vol-target, no ETF
    "BGSRP06 Index":  ("DBMF", 2, "Managed futures proxy for risk premia"),
    "BGSXACV Index":  None,  # Cross-asset carry+value, no ETF

    # =========================================================================
    # GOLDMAN SACHS BASKETS (225)
    # =========================================================================

    # --- Agriculture ---
    "GSCBAGRG Index":  ("DBA",  3, "Global agriculture commodities"),
    "GSXAIAGR Index":  ("DBA",  2, "India ag via broad ag proxy"),

    # --- Carry/Value Factors ---
    "GSXAJBBS Index":  ("EWJ",  2, "Japan beta short via Japan ETF proxy"),
    "GSCMRTNL Index":  ("MTUM", 2, "RTN hedge via momentum proxy"),
    "GSIPMOML Index":  ("MTUM", 4, "Momentum factor long"),
    "GSCMCAPR Index":  ("QUAL", 2, "CPRI hedge via quality proxy"),
    "GSIPROCS Index":  ("QUAL", 2, "ROCE short via quality proxy"),
    "GSPUWFMO Index":  ("MTUM", 3, "Wolfe momentum pair"),
    "GSIPSHOS Index":  None,  # Short interest short, no ETF
    "GSIPDIVS Index":  ("DVY",  2, "Dividend yield short via dividend proxy"),
    "GSIPCROS Index":  ("QUAL", 2, "CROCI short via quality proxy"),
    "GSIPINTS Index":  None,  # Integrated factor short, no ETF
    "GSCMDTEH Index":  None,  # DTE hedge, proprietary
    "GSXUWFTS Index":  ("MTUM", 2, "Wolfe low liquid via momentum proxy"),
    "GSXEBFMS Index":  ("MTUM", 2, "EU low momentum via US momentum proxy"),

    # --- Corporate Credit ---
    "GSXEDEBT Index":  ("HYG",  2, "EU HY debt via US HY proxy"),
    "GSCMCSGN Index":  ("LQD",  1, "Credit Suisse hedge via IG proxy"),
    "GSCBHFDH Index":  ("FLOT", 4, "Floating rate debt"),
    "GSXUDFLT Index":  ("SJNK", 2, "High default prob via short-term HY"),
    "GSCBIGEQ Index":  ("LQD",  4, "Investment grade corporate"),
    "GSCBODBT Index":  ("HYG",  2, "HY debt proxy"),
    "GSXECLFF Index":  ("HYG",  2, "EU debt refinancing needs via HY"),
    "GSXUHYGE Index":  ("HYG",  4, "High yield debt levered"),

    # --- Country/Regional ---
    "GSXATWHN Index":  ("EWT",  3, "Taiwan high dividends"),
    "GSXAKPBP Index":  ("EWY",  4, "Korea low PB high profit"),
    "GSCBJLAM Index":  ("EWJ",  3, "Japan low AGM"),
    "GSXEDEIN Index":  ("EWG",  4, "German international exposure"),
    "GSXAKRIN Index":  ("EWY",  3, "Korea weak won"),
    "GSXATXTM Index":  ("EWT",  2, "Taiwan ex TSMC"),
    "GSXAJPIN Index":  ("DXJ",  3, "Japan weak yen"),
    "GSCBRUS1 Index":  ("VGK",  1, "EU Russia exposure via Europe ETF"),
    "GSCBJRTL Index":  ("EWJ",  2, "Japan custom retail"),
    "GSXACHHU Index":  ("FXI",  4, "China A/H H-shares"),

    # --- Credit Spreads ---
    "GSQIHYLL Index":  ("HYG",  2, "HY credit spreads long"),
    "GSPQSPRD Index":  ("HYG",  2, "QI HY spreads"),
    "GSQIHYLS Index":  None,  # HY credit spreads short, no ETF

    # --- Cross-Asset Indices ---
    "GSCBJXAB Index":  ("EWJ",  2, "Japan corp gov trifecta"),
    "GSCBUKPD Index":  ("EWU",  2, "UK peace deal via UK ETF"),
    "GSPRUCIT Index":  ("KWEB", 2, "China vs US internet"),
    "GSPUMOXX Index":  ("MTUM", 2, "Cross-asset momentum proxy"),
    "GSCBMRGC Index":  ("SPY",  1, "US emergency via broad US"),
    "GSXACTRA Index":  ("FXI",  2, "China trade insurance"),
    "GSPUSTAG Index":  None,  # Stagflation pair, no ETF
    "GSCBJPAR Index":  ("EWJ",  2, "Japan approval rating"),
    "GSSZMFOW Index":  ("AAXJ", 2, "Asia ex Japan mutual fund OW"),
    "GSPUWGRO Index":  ("IWF",  2, "Growth pair proxy"),
    "GSCMAALH Index":  ("EWU",  2, "Anglo hedge via UK ETF"),
    "GSPUCPIP Index":  ("TIP",  2, "US inflation pair via TIPS"),
    "GSQIU10S Index":  ("TBT",  2, "US 10Y short via inverse treasury"),
    "GSCMFMEH Index":  None,  # FME hedge, proprietary
    "GSCMNWGH Index":  None,  # Natwest hedge, proprietary
    "GSCBCNSR Index":  ("MCHI", 2, "Custom China basket"),
    "GSCBHAIH Index":  ("FXI",  2, "China hedge"),
    "GS24TRFS Index":  None,  # Tariff risk, no ETF
    "GSXU1970 Index":  ("INFL", 2, "US inflation comeback"),

    # --- Custom/Proprietary ---
    "GSCMORFP Index":  None,  # L'Oreal custom hedge
    "GSCMMICH Index":  None,  # Magnum hedge
    "GSCMNEST Index":  None,  # Neste custom hedge

    # --- EM FX ---
    "GSQICFXS Index":  None,  # USD CNH short, no ETF
    "GSCMCNHH Index":  None,  # CNH hedge, no ETF
    "GSXFBZFX Index":  ("EWZ", 2, "Brazil FX sensitivity via Brazil ETF"),
    "GSQICFXL Index":  None,  # USD CNH long, no ETF
    "GSCMCNHI Index":  None,  # Custom CNH hedge, no ETF
    "GSDESKYN Index":  None,  # DESK CNY, no ETF
    "GSCMIDRH Index":  None,  # IDR hedge, no ETF

    # --- Energy ---
    "GSSBOILE Index":  ("XOP",  5, "Oil & gas E&P"),
    "GSSBOILS Index":  ("OIH",  5, "Oil services"),
    "GSENOSAN Index":  ("XLE",  2, "Canada oil sands via energy sector"),
    "GSCMHBRL Index":  ("XLE",  1, "Harbour Energy hedge via energy"),
    "GSXEBOIL Index":  ("XLE",  3, "EU big oil via energy sector"),
    "GSSBOILR Index":  ("CRAK", 5, "Oil refiners"),
    "GSXAOILX Index":  ("XLE",  2, "APAC oil input cost"),
    "GSXGNUCL Index":  ("URA",  4, "Global uranium/nuclear"),
    "GSCBGURA Index":  ("URNM", 5, "Custom uranium miners"),
    "GSCBGCOA Index":  ("XME",  1, "Coal proxy via metals/mining"),
    "GSXFBZCM Index":  ("EWZ",  3, "Brazil commodity levered"),
    "GSXAURAN Index":  ("URA",  4, "Asia uranium"),

    # --- Energy/Metals/Agriculture ---
    "GSXAFCFC Index":  ("XME",  2, "Asia FCF commods via metals/mining"),
    "GSCBAFCF Index":  ("XME",  2, "Asia high FCF commods"),
    "GSXUCOMO Index":  ("BCI",  3, "US commodity basket"),

    # --- Hedge / Downside Protection ---
    "GSCMSWON Index":  None,  # SoftwareOne hedge, proprietary

    # --- Low Volatility Factors ---
    "GSXELVOW Index":  ("SPLV", 2, "EU Wolfe low vol via US low vol"),

    # --- Majors (FX) ---
    "GSQIEFXL Index":  ("FXE",  4, "EUR/USD long"),
    "GSXFBRFX Index":  ("EWZ",  2, "Brazil FX sensitivity"),
    "GSCBEJPY Index":  ("DXJ",  2, "EU strong JPY via hedged Japan"),
    "GSQIEFXS Index":  None,  # EUR/USD short, no ETF

    # --- Metals ---
    "GSCMFRES Index":  ("GDX",  2, "Custom FRES hedge via gold miners"),
    "GSCMAMSJ Index":  ("GDX",  2, "Anglo Platinum hedge"),
    "GSCMSAIO Index":  ("PICK", 2, "SA iron ore hedge"),
    "GSIDLMET Index":  ("XME",  4, "Metals & mining ex gold"),
    "GSXFBRCO Index":  ("EWZ",  2, "Brazil commodity levered"),
    "GSCBLITH Index":  ("LIT",  4, "Lithium miners"),
    "GSCBGLLI Index":  ("LIT",  4, "Global lithium"),
    "GSXGOLDM Index":  ("GDX",  5, "Global gold miners"),
    "GSCMGLEG Index":  ("XME",  2, "Glencore hedge via metals/mining"),
    "GSCBIROR Index":  ("PICK", 3, "Global iron ore"),
    "GSXAJCRR Index":  ("REMX", 3, "Japan critical resources"),
    "GSXGCOPP Index":  ("COPX", 5, "Global copper exposed"),
    "GSXGCUMA Index":  ("COPX", 3, "Copper M&A"),
    "GSXGRARE Index":  ("REMX", 5, "Global rare earths"),
    "GSPWMURA Index":  ("URA",  4, "Uranium"),
    "GSXAGOLD Index":  ("GDX",  3, "Asia gold miners"),
    "GSCMPOLM Index":  ("GDX",  1, "Polymetal custom hedge"),
    "GSXACRAR Index":  ("REMX", 4, "China rare earths"),

    # --- Options-Based ---
    "GSCMITMH Index":  None,  # ITM hedge, proprietary

    # --- Quant/Style Baskets ---
    "GSPUBGPF Index":  ("QUAL", 2, "GEMLT profitability pair"),
    "GSFIMOMO Index":  ("MTUM", 3, "US financials momentum"),
    "GSPUWREV Index":  ("MTUM", 2, "Wolfe revisions pair"),
    "GSPUSENT Index":  ("MTUM", 2, "Barra sentiment"),
    "GSPUMFMO Index":  ("MTUM", 3, "Axioma momentum"),

    # --- Real Estate / REITs ---
    "GSSBGRES Index":  ("VNQ",  2, "German residential via US REITs"),
    "GSFINWCO Index":  ("VNQ",  2, "West Coast office via US REITs"),

    # --- Sector Indices ---
    "GSSBDDEF Index":  ("XLP",  3, "Domestic defensives via staples"),
    "GSSBOPEX Index":  ("XLI",  3, "OPEX industrials"),
    "GSTMTSFT Index":  ("IGV",  4, "Software basket"),
    "GSSIDIST Index":  ("XLY",  2, "Distributors via consumer disc"),
    "GSSICMEQ Index":  ("XLC",  2, "Communications equipment"),
    "GSFINCRB Index":  ("XLF",  3, "Banks with CRE exposure"),
    "GSBRCONS Index":  ("EWZ",  2, "Brazil consumer via Brazil ETF"),
    "GSCMSFWR Index":  ("XLI",  1, "Smurfit WestRock hedge"),
    "GSSBPHAR Index":  ("XLV",  3, "Pharma via healthcare"),
    "GSHLCHSP Index":  ("VHT",  4, "Hospitals & providers"),
    "GSFINMNY Index":  ("XLF",  3, "Money-center banks"),
    "GSXACOMO Index":  ("GUNR", 2, "Asia commodities via global resources"),
    "GSXERATE Index":  ("EUFN", 3, "EU rate-sensitive financials"),
    "GSSBUTUR Index":  ("XLU",  4, "Unregulated utilities"),
    "GSXULOCN Index":  ("XLP",  2, "Low income consumer via staples"),
    "GSSIINCR Index":  ("FDN",  3, "Internet retail"),
    "GSSIELUT Index":  ("XLU",  4, "Electric utilities"),
    "GSXEMEDT Index":  ("IHI",  5, "EU medical technology via US medtech"),
    "GSCBMSIN Index":  ("XLI",  1, "Industrials most short (long-only proxy)"),
    "GSHLCMDT Index":  ("IHI",  4, "Medtech tools"),
    "GSSIHOME Index":  ("ITB",  5, "Homebuilders"),
    "GSXUIBIO Index":  ("XBI",  5, "US biotechnology"),
    "GSXUENRG Index":  ("XLE",  4, "US energy"),
    "GSXUHLTH Index":  ("XLV",  4, "US healthcare"),
    "GSENLVRD Index":  ("XLE",  2, "Levered energy basket"),
    "GSSIDVFS Index":  ("XLF",  2, "Diversified financial services"),

    # --- Sovereign Bonds ---
    "GSXAHK7S Index":  ("ISHG", 2, "HK 7s via short intl treasury"),
    "GSCBXAB2 Index":  ("EWJ",  1, "Japan gov trifecta 2 via Japan ETF"),

    # --- Thematic/Factor ---
    "GSXACHSB Index":  ("MCHI", 3, "China domestic stimulus"),
    "GSXACHGG Index":  ("MCHI", 3, "China going global"),
    "GSPUERNY Index":  ("VLUE", 2, "Barra earnings yield pair"),
    "GSCBHKBM Index":  ("KWEB", 3, "HK tech ex Baba Mei"),
    "GSXUBFTL Index":  ("MTUM", 2, "Barra high LTR"),
    "GSXUWFGS Index":  ("IWF",  2, "Wolfe low growth via growth"),
    "GSXABIOT Index":  ("XBI",  3, "Asia biotech via US biotech"),
    "GSFIBFVL Index":  ("XLF",  3, "US financials value long"),
    "GSCMRYAH Index":  None,  # Ryanair hedge, single stock
    "GSHLCOMB Index":  ("XBI",  3, "US commercial biotech"),
    "GSXACHSS Index":  ("MCHI", 2, "China size short"),
    "GSIPBALS Index":  None,  # Balance sheet short, no ETF
    "GSEMCHNA Index":  ("MCHI", 3, "EM China sales exposure"),
    "GS24DEML Index":  ("SPY",  1, "Dem policy outperform via broad US"),
    "GSCBJHPQ Index":  ("QUAL", 3, "Japan high PB quality"),
    "GSPUFCFP Index":  ("COWZ", 3, "High vs low FCF via cash flow ETF"),
    "GSCNGRWT Index":  ("XLY",  3, "US consumer growth"),
    "GSIPBALS Index":  None,  # Balance sheet short, no ETF (duplicate key handled)
    "GSPUHLCN Index":  ("XLV",  2, "US HC pair"),
    "GSFIBFVL Index":  ("XLF",  3, "US fin value long"),  # duplicate ok
    "GSSZSERV Index":  ("AAXJ", 2, "AeJ strong earning revision"),
    "GSXAJPHD Index":  ("DXJ",  2, "Japan high dividends"),
    "GSRHOILX Index":  ("XLE",  2, "Oil input cost exposure"),
    "GSSZLDUR Index":  ("AAXJ", 2, "APJ long duration"),
    "GSCBCHSF Index":  ("CQQQ", 4, "China software"),
    "GSINLMOL Index":  ("MTUM", 4, "US momentum long"),
    "GSTHMFOW Index":  ("SPY",  2, "Mutual fund OW US"),
    "GSTMTVLS Index":  ("IGV",  2, "TMT US value short via software"),
    "GSXACHAI Index":  ("AIQ",  3, "China AI"),
    "GSXELMOW Index":  ("MTUM", 2, "EU Wolfe low momo"),
    "GSXULVSG Index":  ("VTV",  2, "US high value vs growth"),
    "GSXAAIGC Index":  ("AIQ",  3, "China AI gen content"),
    "GSCMVKHG Index":  None,  # Vallourec hedge, single stock
    "GSCMDISC Index":  ("VGK",  3, "EU at a discount"),
    "GSCBSMQB Index":  ("SPHQ", 4, "Small cap quality"),
    "GSIPPERL Index":  ("RPV",  2, "PE long via pure value"),
    "GSXAJBGL Index":  ("EWJ",  2, "Japan Barra growth long"),
    "GSXEMEGA Index":  ("EZU",  3, "Make EU great again"),
    "GSCBAIP3 Index":  ("AIQ",  3, "GIR AI phase 3"),
    "GSXUHVAL Index":  ("RPV",  3, "US high beta cheap value"),
    "GSXUBGLL Index":  None,  # GEMLT high leverage, no ETF
    "GSXELPRF Index":  ("VGK",  2, "EU low profitability"),
    "GSPUCHAT Index":  ("AIQ",  3, "US AI vs nonprof tech"),
    "GSXAJBQS Index":  ("EWJ",  2, "Japan Barra quality short"),
    "GSCMVODH Index":  None,  # Vodafone hedge, single stock
    "GSXAJDSC Index":  ("EWJ",  2, "Japan domestic sec consol"),
    "GSWDADAP Index":  ("SPY",  1, "New cycle adaptors"),
    "GSPUHLCM Index":  ("XLV",  2, "US HC momo pair"),
    "GSXUINFH Index":  ("PAVE", 4, "Infrastructure hedge"),
    "GSINMCMB Index":  ("INDA", 3, "India midcap multibagger"),
    "GSIPEVES Index":  ("RPV",  2, "EV/EBITDA short via value"),
    "GSCNLEVG Index":  ("XLY",  2, "US consumer leverage"),
    "GSCNDVLS Index":  ("XLY",  2, "CND US value short"),
    "GSPUARTI Index":  ("AIQ",  2, "AI pair"),
    "GSP24TRF Index":  None,  # Tariff risk pair, no ETF
    "GSPUGRVA Index":  ("VTV",  2, "Growth vs value pair"),
    "GSPUINFS Index":  ("PAVE", 3, "Infrastructure pair"),
    "GSPUSHOR Index":  ("RSHO", 4, "Onshoring pair"),
    "GSPRMUTL Index":  ("SPY",  1, "MF OW vs UW"),
    "GSCBCNRO Index":  ("MCHI", 3, "Custom CN reopening"),
    "GSP24REP Index":  ("SPY",  1, "Republican pair"),

    # --- Vol Indices ---
    "GSXEMFWL Index":  ("VIXY", 2, "EU vol long via VIX proxy"),
    "GSFIRSVL Index":  ("XLF",  2, "US financials res vol"),
    "GSCMDSVH Index":  None,  # DSV hedge, proprietary
    "GSPUBGRV Index":  ("SPLV", 2, "GEMLT residual vol pair"),
    "GSPUMFVO Index":  ("SPLV", 2, "US Axioma vol"),
    "GSXUMFWL Index":  ("VIXY", 2, "US volatility long"),
    "GSCBSGRL Index":  ("IWF",  2, "Vol opt sec growth"),
    "GSXEMFWS Index":  ("SVXY", 2, "EU vol short via inverse VIX"),
    "GSXELRSV Index":  ("SPLV", 2, "EU low residual vol"),
    "GSXAMFWS Index":  ("EWJ",  2, "Japan vol short"),
    "GSXEHVOW Index":  ("VGK",  2, "EU Wolfe high vol"),
    "GSTMBFRL Index":  ("IGV",  2, "US TMT res vol long"),
    "GSTMRSVL Index":  ("IGV",  2, "US TMT residual volatility"),
    "GSXAHBDS Index":  ("EWH",  2, "HK delta-beta short via HK ETF"),

    # --- Yield Curves ---
    "GSPU10YR Index":  ("IEF",  4, "US 10Y rate sensitive"),
    "GSCBJYC1 Index":  None,  # Japan YCC, proprietary
    "GSXFDILG Index":  None,  # Brazil DI corr long, no ETF
    "GSQI5YIL Index":  ("TIP",  3, "US 5Y inflation expectations"),
    "GSBZRATE Index":  None,  # Brazil rate sensitive, no ETF
    "GSCBLDUR Index":  None,  # EU long duration custom, no ETF
    "GSXELDUR Index":  None,  # EU long duration, no ETF
    "GSCBSDUR Index":  None,  # EU short duration custom, no ETF
    "GSCB30YR Index":  ("TLT",  3, "EU 30Y bond sensitive via US long treasury"),
    "GSXESDUR Index":  None,  # EU short duration, no ETF
    "GSPUDURA Index":  None,  # Duration pair, no ETF
    "GSBRDILG Index":  None,  # Brazil DI corr long, no ETF
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_yf_ticker(bloomberg_ticker: str) -> Optional[str]:
    """
    Convert a Bloomberg ticker to a Yahoo Finance ticker.

    For ETFs (ending in 'Equity'), strips the suffix.
    For Index tickers, looks up the mapping.

    Returns None if the ticker should be dropped.
    """
    ticker = bloomberg_ticker.strip()

    # ETF tickers: "QAI US Equity" → "QAI"
    if "Equity" in ticker:
        parts = ticker.split()
        yf = parts[0]
        # Handle Canadian tickers: "ETHH/B CN Equity" → "ETHH-B.TO"
        if len(parts) >= 3 and parts[1] == "CN":
            yf = parts[0].replace("/", "-") + ".TO"
        return yf

    # Index tickers: look up mapping
    entry = BLOOMBERG_TO_ETF.get(ticker)
    if entry is None:
        return None  # Drop this ticker
    return entry[0]


def get_tracking_score(bloomberg_ticker: str) -> int:
    """Return the tracking confidence score (1-5) for a mapped ticker."""
    entry = BLOOMBERG_TO_ETF.get(bloomberg_ticker.strip())
    if entry is None:
        return 0
    return entry[1]


def build_full_yf_ticker_list(bloomberg_tickers: list) -> dict:
    """
    Convert a list of Bloomberg tickers to Yahoo Finance tickers.

    Returns:
        Dict mapping bloomberg_ticker → yf_ticker (None entries are dropped)
    """
    mapping = {}
    for bt in bloomberg_tickers:
        yf = get_yf_ticker(bt)
        if yf is not None:
            mapping[bt] = yf
    return mapping


def get_dropped_tickers() -> list:
    """Return list of Bloomberg tickers that have no ETF replacement."""
    return [k for k, v in BLOOMBERG_TO_ETF.items() if v is None]


def migration_summary() -> dict:
    """Return a summary of the migration mapping."""
    total = len(BLOOMBERG_TO_ETF)
    mapped = sum(1 for v in BLOOMBERG_TO_ETF.values() if v is not None)
    dropped = total - mapped

    by_score = {}
    for entry in BLOOMBERG_TO_ETF.values():
        if entry is not None:
            score = entry[1]
            by_score[score] = by_score.get(score, 0) + 1

    return {
        "total_index_tickers": total,
        "mapped_to_etf": mapped,
        "dropped_no_replacement": dropped,
        "by_tracking_score": by_score,
        "factor_tickers": len(FACTOR_ETF_MAP),
    }


if __name__ == "__main__":
    summary = migration_summary()
    print("=" * 60)
    print("ETF MIGRATION MAPPING SUMMARY")
    print("=" * 60)
    print(f"\nTotal Bloomberg/GS Index tickers: {summary['total_index_tickers']}")
    print(f"Mapped to ETF:                    {summary['mapped_to_etf']}")
    print(f"Dropped (no replacement):         {summary['dropped_no_replacement']}")
    print(f"\nFactor tickers:                   {summary['factor_tickers']}")
    print(f"\nTracking score distribution:")
    for score in sorted(summary['by_tracking_score'].keys()):
        count = summary['by_tracking_score'][score]
        label = {1: "Weak proxy", 2: "Partial", 3: "Reasonable",
                 4: "Good", 5: "Excellent"}[score]
        print(f"  Score {score} ({label}): {count}")

    print(f"\nDropped tickers:")
    for t in sorted(get_dropped_tickers()):
        print(f"  {t}")
