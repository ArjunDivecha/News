#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: tags.py
=============================================================================

INPUT FILES:
    - /Users/arjundivecha/Dropbox/AAA Backup/A Working/News/data/universe.xlsx
      (universe tags, normalized to the canonical vocab)
    - /Users/arjundivecha/Dropbox/AAA Backup/A Working/News/data/report.db
      (security_tags cache + latest portfolio snapshot of live holdings)
    - /Users/arjundivecha/Dropbox/AAA Backup/A Working/News/GMO.xlsx
      (the separate GMO sleeve holdings)
    - Yahoo Finance (full name + longBusinessSummary description, via yfinance)
    - Arjun Model Gateway (local LaunchAgent, http://127.0.0.1:8765/v1) — BOTH
      the classifier (routing profile "cheap") and the web_search fallback for
      junk names (OpenRouter web_search server tool). The gateway picks the
      actual model dynamically from the live catalog, bills exact cost, and
      LEARNS from the outcomes this script reports back. No provider API keys
      are needed in this repo for tagging.
    - /Users/arjundivecha/Dropbox/AAA Backup/A Working/News/fine tuning/scripts/expert_review_common.py
      (SYSTEM_PROMPT, build_user_prompt, parse_model_json, tags_to_string,
       ALLOWED_TIER3_TAGS — the canonical vocabulary + normalizer)

OUTPUT FILES:
    - /Users/arjundivecha/Dropbox/AAA Backup/A Working/News/data/report.db
      (security_tags table: yf_ticker -> canonical tags, cached)

VERSION: 3.0
LAST UPDATED: 2026-07-11
AUTHOR: Arjun Divecha

DESCRIPTION:
    Workstream-0 dynamic holding tagger. Resolves every holding to canonical
    tier-3 tags. Quality pipeline (v2):

      1. MANUAL_OVERRIDES  - user-authoritative corrections win over everything
                             (you know your own holdings; web sources can be wrong).
      2. universe tags     - authoritative for clean universe rows (unless forced).
      3. report.db cache   - previously resolved.
      4. classify          - fetch REAL FACTS first (Yahoo longName + business
                             summary; for junk names a gateway web_search lookup),
                             then hand the name+description to the gateway
                             classifier ("cheap" profile). The classifier model
                             has NO web access itself, so it must be fed facts -
                             name-only starves it and it hallucinates (that was
                             the v1 failure). v3: every classification call's
                             outcome (non-empty canonical tags or not) is posted
                             to the gateway's /v1/outcomes, so model routing
                             improves from this pipeline's real results.

    Holdings are force-reclassified fresh (universe tags are currently GS-basket
    contaminated for ~41 tickers, so the book must not inherit them).

    Run directly to tag the whole book and print the result:
        python3 report/tags.py

DEPENDENCIES:
    - pandas, yfinance, openai
=============================================================================
"""

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))            # report/
_FT = Path(__file__).resolve().parent.parent / "fine tuning" / "scripts"
sys.path.insert(0, str(_FT))                                        # reuse tagger

from config import PATHS, CASH_EQUIVALENTS                          # loads .env
import db
import names as names_mod
import expert_review_common as erc                                  # the tagger

# All LLM calls go through the local Arjun Model Gateway (Divecha Router).
# "Models" here are ROUTING PROFILES — the gateway resolves the actual model
# from the live OpenRouter catalog, so nothing in this file goes stale when
# new models ship. FAIL IS FAIL: if the gateway is down, tagging fails loudly
# (the per-symbol handler records "none" exactly as before).
GATEWAY_BASE_URL = "http://127.0.0.1:8765/v1"
GATEWAY_CLASSIFY_PROFILE = "cheap"
GATEWAY_SEARCH_PROFILE = "cheap"  # + openrouter:web_search server tool

# User-authoritative overrides (you know your own holdings). name/description
# feed the classifier; "tags" (optional) pins the canonical output directly.
MANUAL_OVERRIDES = {
    # Private single-company stakes (config.MANUAL_HOLDINGS) — illiquid,
    # non-public equity. Classified as Alternatives (private equity/venture),
    # not public Equities, matching how these are treated everywhere else in
    # the report (no daily price, no public-market co-movement).
    "ANTHROPIC_PVT": {"name": "Anthropic (private stake)",
                      "tags": "Alternative, US, Active"},
    "PERPLEXITY_PVT": {"name": "Perplexity (private stake)",
                       "tags": "Alternative, US, Active"},
    # User-confirmed:
    "GMOQX": {
        "name": "GMO Emerging Country Debt Fund",
        "tags": "Credit, EM, Active, High Yield",
    },
    # Single-country EM equity ETFs the tagger under-/mis-classified (needed for
    # a correct household asset-allocation report):
    "EWU": {"name": "iShares MSCI United Kingdom ETF", "tags": "Equity, Europe"},
    "EWZ": {"name": "iShares MSCI Brazil ETF", "tags": "Equity, EM"},
    "EZA": {"name": "iShares MSCI South Africa ETF", "tags": "Equity, EM"},
    "ECH": {"name": "iShares MSCI Chile ETF", "tags": "Equity, EM"},
    # Benchmark components (60/40 ACWI/TLT) - must be exactly right:
    "TLT": {  # iShares 20+ Year Treasury - NOT credit, NOT Europe
        "name": "iShares 20+ Year Treasury Bond ETF",
        "tags": "Treasury, US, Long Duration, Passive",
    },
    "ACWI": {  # iShares MSCI ACWI (global large-cap equity)
        "name": "iShares MSCI ACWI ETF",
        "tags": "Equity, Global, Large-Cap, Passive",
    },
    # Research-confirmed (HIGH confidence — issuer/SEC/CUSIP-index sourced):
    "IE00BF199475": {  # GMO Equity Dislocation: long value / short growth, beta-neutral
        "name": "GMO Equity Dislocation Fund",
        "tags": "Alternative, Global, Value, Active, Quantitative, Factor-Based",
    },
    "EDD": {
        "name": "Morgan Stanley Emerging Markets Domestic Debt Fund",
        "tags": "Credit, EM, HY Credit, Medium Duration, Active",
    },
    "92189F403": {  # VanEck Russia ETF (RSX) — wound down / frozen legacy position
        "name": "VanEck Russia ETF (RSX)",
        "tags": "Equity, EM, Passive, Diversified",
    },
    "12464X101": {  # BZAM Ltd (ex-TGOD) — Canadian cannabis, delisted 2024 (stale)
        "name": "BZAM Ltd",
        "tags": "Equity, Canada, Healthcare",
    },
    "ETM": {  # Entercom/Audacy — former ticker, delisted/private (stale)
        "name": "Audacy, Inc. (formerly Entercom)",
        "tags": "Equity, US, Consumer",
    },
    # LOWER confidence — pinned to best identification; CONFIRM with custodian:
    "IES": {  # likely IES Holdings (US Industrials/Infrastructure); bare ticker ambiguous
        "name": "IES Holdings, Inc.",
        "tags": "Equity, US, Industrials, Infrastructure, Mid-Cap",
    },
    "SES+": {  # SES S.A. (Luxembourg satellite); '+' is a broker suffix
        "name": "SES S.A.",
        "tags": "Equity, Europe, Tech, Infrastructure",
    },
    # Specific GMO/structured holdings the classifier left empty:
    "GBMBX": {  # GMO Benchmark-Free Allocation — flagship multi-asset absolute return
        "name": "GMO Benchmark-Free Allocation Fund",
        "tags": "Multi-Asset, Global, Active, Diversified",
    },
    "BCHI": {  # GMO Beyond China ETF — thematic, companies benefiting ex-China
        "name": "GMO Beyond China ETF",
        "tags": "Equity, Global, Thematic, Active",
    },
    "CORD": {  # T-REX 2X Inverse — leveraged/inverse single-stock product
        "name": "T-REX 2X Inverse CRWV Daily Target ETF",
        "tags": "Alternative, US, Options-Based",
    },
    "VCLAX": {  # Vanguard California Long-Term Tax-Exempt — muni bonds
        "name": "Vanguard California Long-Term Tax-Exempt Fund",
        "tags": "Municipal, US, Long Duration, Active",
    },
    "GCCHX": {  # GMO Climate Change — persistent classifier whiff
        "name": "GMO Climate Change Fund",
        "tags": "Equity, Global, ESG, Thematic, Active",
    },
    # Persistent classifier whiffs (pinned; correct + deterministic):
    "EWS": {"name": "iShares MSCI Singapore ETF",
            "tags": "Equity, Asia, Developed, Passive"},
    "CEE": {"name": "The Central and Eastern Europe Fund",
            "tags": "Equity, Europe, EM, Active"},
    "LGO": {"name": "Largo Inc.",
            "tags": "Equity, Canada, Materials"},
    "UMICY": {"name": "Umicore SA",
              "tags": "Equity, Europe, Materials"},
}


# --------------------------------------------------------------------------- #
# report.db cache (security_tags)
# --------------------------------------------------------------------------- #
def _ensure_cache() -> None:
    with db.connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS security_tags (
                yf_ticker  TEXT PRIMARY KEY,
                tags       TEXT,
                tier1      TEXT,
                tier2      TEXT,
                name       TEXT,
                source     TEXT,
                fetched_at TEXT DEFAULT (datetime('now'))
            )""")
        # migrate a v1 table that predates the `name` column
        cols = {r[1] for r in conn.execute("PRAGMA table_info(security_tags)")}
        if "name" not in cols:
            conn.execute("ALTER TABLE security_tags ADD COLUMN name TEXT")


def get_cached_tags(symbols) -> dict:
    _ensure_cache()
    syms = [str(s).strip() for s in symbols if str(s).strip()]
    out = {}
    if not syms:
        return out
    with db.connect() as conn:
        for i in range(0, len(syms), 400):
            chunk = syms[i:i + 400]
            q = ("SELECT yf_ticker, tags FROM security_tags "
                 "WHERE yf_ticker IN (%s)" % ",".join("?" * len(chunk)))
            for r in conn.execute(q, chunk).fetchall():
                out[r["yf_ticker"]] = r["tags"] or ""
    return out


def _persist_overrides(out: dict) -> None:
    """Cache any override-sourced entries so a no-fetch pass still writes them."""
    rows = [{"yf_ticker": s, "tags": v["tags"], "tier1": None, "tier2": None,
             "name": v.get("name"), "source": "override"}
            for s, v in out.items()
            if v.get("source") == "override" and v.get("tags")]
    upsert_tags(rows)


def upsert_tags(rows: list) -> int:
    if not rows:
        return 0
    _ensure_cache()
    with db.connect() as conn:
        conn.executemany("""
            INSERT INTO security_tags (yf_ticker, tags, tier1, tier2, name, source, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(yf_ticker) DO UPDATE SET
                tags=excluded.tags, tier1=excluded.tier1, tier2=excluded.tier2,
                name=excluded.name, source=excluded.source, fetched_at=datetime('now')
        """, [(r["yf_ticker"], r["tags"], r.get("tier1"), r.get("tier2"),
               r.get("name"), r.get("source", "gateway")) for r in rows])
    return len(rows)


# --------------------------------------------------------------------------- #
# fact sources: universe tags, Yahoo info, OpenAI web_search fallback
# --------------------------------------------------------------------------- #
def _universe_tags() -> dict:
    u = pd.read_excel(PATHS["universe"])
    return {str(t): erc.tags_to_string(tg)
            for t, tg in zip(u["yf_ticker"], u["tags"])}


def _yahoo_info(sym: str) -> tuple:
    """(name, description) from yfinance; best-effort, ('','') on failure."""
    try:
        import yfinance as yf
        info = yf.Ticker(sym).get_info()
        name = (info.get("longName") or info.get("shortName") or "").strip()
        bits = [info.get("longBusinessSummary") or "",
                f"Category: {info.get('category')}" if info.get("category") else "",
                f"Type: {info.get('quoteType')}" if info.get("quoteType") else "",
                f"Sector: {info.get('sector')}" if info.get("sector") else ""]
        desc = " ".join(b for b in bits if b).strip()
        return name, desc[:1200]
    except Exception:
        return "", ""


def _is_junk_name(name: str, sym: str) -> bool:
    """A name we can't classify on: empty, equal to the ticker, numeric, CUSIP."""
    n = (name or "").strip()
    if not n or n == sym:
        return True
    if re.fullmatch(r"[0-9]+", n):                      # e.g. "397991"
        return True
    if re.fullmatch(r"[0-9A-Z]{9}", sym) and n == sym:  # bare CUSIP
        return True
    return False


def _websearch_identify(sym: str) -> tuple:
    """Gateway web_search FALLBACK for junk names -> (name, description).

    The classifier model can't browse; this is how a genuinely-unknown
    ticker/CUSIP gets real facts. Runs through the gateway with OpenRouter's
    web_search server tool (tool_choice=required so the search always runs).
    Low-confidence by nature - flagged for review by the caller.
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key="local", base_url=GATEWAY_BASE_URL)
        r = client.chat.completions.create(
            model=GATEWAY_SEARCH_PROFILE,
            messages=[{"role": "user", "content": (
                f"Identify the security with ticker or CUSIP '{sym}'. "
                f"FIRST LINE of your answer: ONLY the security's official name, "
                f"nothing else. Then one or two sentences on what it is and what "
                f"it invests in (asset class, region, sector, strategy). If you "
                f"cannot identify it confidently, the first line must be "
                f"UNIDENTIFIED.")}],
            tools=[{"type": "openrouter:web_search"}],
            tool_choice="required",
            extra_body={"metadata": {"repo": "News", "script": "report/tags.py",
                                     "call_site_id": "tags._websearch_identify",
                                     "routing_profile": GATEWAY_SEARCH_PROFILE}})
        txt = (r.choices[0].message.content or "").strip()
        # first line = official name (prompt-enforced); whole thing ~ description
        first = txt.split("\n", 1)[0].strip(" *#-")
        if not first or first.upper().startswith("UNIDENTIFIED"):
            return "", ""
        return first[:120], txt[:1200]
    except Exception as e:
        print(f"    web_search failed for {sym}: {e}")
        return "", ""


_CAP_TAGS = {"Small-Cap", "Mid-Cap", "Large-Cap"}


def _drop_multicap(tags_str: str) -> str:
    """A broad/all-cap fund often gets tagged Small+Mid+Large — meaningless.
    If 2+ size tags are present, drop them all (no defining size tilt)."""
    tags = [t.strip() for t in tags_str.split(",") if t.strip()]
    if len([t for t in tags if t in _CAP_TAGS]) >= 2:
        tags = [t for t in tags if t not in _CAP_TAGS]
    return ", ".join(tags)


def _gateway_classify(client, ticker: str, name: str, description: str) -> tuple:
    """One classification call through the gateway -> (parsed, amg_request_id).

    The amg_request_id is the key for reporting this call's verified outcome
    back to the gateway (its learning signal)."""
    row = pd.Series({"yf_ticker": ticker, "name": name,
                     "description": description})
    resp = client.chat.completions.create(
        model=GATEWAY_CLASSIFY_PROFILE,
        messages=[{"role": "system", "content": erc.SYSTEM_PROMPT},
                  {"role": "user", "content": erc.build_user_prompt(row)}],
        temperature=0, max_tokens=256,
        extra_body={"metadata": {"repo": "News", "script": "report/tags.py",
                                 "call_site_id": "tags._gateway_classify",
                                 "routing_profile": GATEWAY_CLASSIFY_PROFILE}})
    request_id = getattr(resp, "amg_request_id", None)
    return erc.parse_model_json(resp.choices[0].message.content), request_id


def _report_outcome(request_id, success: bool, detail: str) -> None:
    """Post a verified outcome to the gateway. Best-effort: outcome reporting
    must never break the report pipeline."""
    if not request_id:
        return
    try:
        import httpx
        httpx.post(f"{GATEWAY_BASE_URL}/outcomes",
                   json={"request_id": request_id, "success": bool(success),
                         "verifier": "tests", "detail": detail},
                   timeout=10)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# resolve
# --------------------------------------------------------------------------- #
def resolve_tags(symbols, name_hints=None, force=None, fetch=True) -> dict:
    """
    Return {symbol: {"tags", "source", "name"}}.
      source: override | universe | cache | gateway | gateway+websearch | none
              (rows cached before 2026-07-11 may carry the old "deepseek" labels)
      force:  symbols to (re)classify fresh, bypassing universe + cache.
    """
    name_hints = name_hints or {}
    force = set(force or [])
    syms = sorted({str(s).strip() for s in symbols if str(s).strip()})
    uni = _universe_tags()
    cached = get_cached_tags(syms)

    out, to_tag = {}, []
    for s in syms:
        ov = MANUAL_OVERRIDES.get(s)
        if ov and ov.get("tags") is not None:
            out[s] = {"tags": ov["tags"], "source": "override",
                      "name": ov.get("name", name_hints.get(s, s))}
        elif s not in force and uni.get(s):
            out[s] = {"tags": uni[s], "source": "universe",
                      "name": name_hints.get(s, s)}
        elif s not in force and cached.get(s):   # non-empty cache hit only;
            out[s] = {"tags": cached[s], "source": "cache",  # empty -> re-tag
                      "name": name_hints.get(s, s)}
        else:
            to_tag.append(s)

    if not (fetch and to_tag):
        for s in to_tag:
            out[s] = {"tags": "", "source": "none", "name": name_hints.get(s, s)}
        _persist_overrides(out)
        return out

    from openai import OpenAI
    gw = OpenAI(api_key="local", base_url=GATEWAY_BASE_URL)
    new_rows = []
    for s in to_tag:
        ov = MANUAL_OVERRIDES.get(s, {})
        # ---- assemble the best available (name, description) ----
        yf_name, yf_desc = _yahoo_info(s)
        name = ov.get("name") or (name_hints.get(s) if not
                                   _is_junk_name(name_hints.get(s, ""), s) else "") \
            or (yf_name if not _is_junk_name(yf_name, s) else "") or ""
        desc = ov.get("description") or yf_desc or ""
        src = "gateway"
        if _is_junk_name(name, s):
            ws_name, ws_desc = _websearch_identify(s)
            if ws_name:
                name, desc, src = ws_name, ws_desc or desc, "gateway+websearch"
        if _is_junk_name(name, s):
            out[s] = {"tags": "", "source": "none", "name": name or s}
            print(f"  UNRESOLVED {s:14s} (no name from Yahoo or web)")
            continue
        # ---- classify from the facts (best-of-3: cheap classifier models are
        #      non-deterministic even at temp 0 and some calls return
        #      empty/thin). Each call's outcome is reported to the gateway:
        #      that is how the router learns which models actually work here.
        def _ntags(t):
            return len([x for x in t.split(",") if x.strip()])
        try:
            best_tags, best_p = "", None
            for _ in range(3):
                p, req_id = _gateway_classify(gw, s, name, desc)
                cand = _drop_multicap(erc.tags_to_string(p["tier3_tags"]))
                _report_outcome(req_id, bool(cand),
                                f"canonical tags {'non-empty' if cand else 'EMPTY'} for {s}")
                if _ntags(cand) > _ntags(best_tags):
                    best_tags, best_p = cand, p
                if _ntags(best_tags) >= 3:
                    break
            tags = best_tags
            out[s] = {"tags": tags, "source": src, "name": name}
            # only cache a NON-empty result - caching "" would be a cache-hit
            # next run and the symbol would never be retried (cache poisoning)
            if tags:
                new_rows.append({"yf_ticker": s, "tags": tags,
                                 "tier1": best_p["tier1"] if best_p else None,
                                 "tier2": best_p["tier2"] if best_p else None,
                                 "name": name, "source": src})
            flag = " [web]" if src.endswith("websearch") else ""
            print(f"  {src.split('+')[0]:8s}{flag} {s:13s} {name[:40]:40s} -> {tags or '(empty!)'}")
        except Exception as e:
            out[s] = {"tags": "", "source": "none", "name": name}
            print(f"  FAIL  {s:14s} {e}")

    upsert_tags(new_rows)
    _persist_overrides(out)
    return out


# --------------------------------------------------------------------------- #
# collect the whole book (live snapshot + GMO sleeve)
# --------------------------------------------------------------------------- #
def _live_holdings() -> list:
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT DISTINCT symbol FROM portfolio_snapshots "
            "WHERE date=(SELECT MAX(date) FROM portfolio_snapshots)").fetchall()
    return [r[0] for r in rows if r[0] not in CASH_EQUIVALENTS]


def _gmo_holdings() -> tuple:
    path = PATHS.get("gmo_xlsx")
    if not path or not path.exists():
        return [], {}
    g = pd.read_excel(path)
    tcol = "Ticker.1" if "Ticker.1" in g.columns else "Ticker"
    syms, names = [], {}
    for _, r in g.iterrows():
        t = str(r.get(tcol, "")).replace("\xa0", " ").strip()
        if t.upper().startswith("ISIN "):
            t = t[5:].strip()
        if not t or t == "nan":
            continue
        syms.append(t)
        nm = str(r.get("Ticker", "")).strip()
        if nm and nm != "nan" and not nm.upper().startswith("ISIN"):
            names[t] = nm
    return syms, names


if __name__ == "__main__":
    live = _live_holdings()
    gmo, gmo_names = _gmo_holdings()
    all_syms = sorted(set(live) | set(gmo))
    print(f"Tagging {len(all_syms)} holdings "
          f"({len(live)} live + {len(gmo)} GMO), force-fresh...\n")

    name_hints = names_mod.resolve_names(all_syms)
    name_hints.update(gmo_names)

    # Force-reclassify only holdings that are NOT in the universe or whose
    # universe row is GS-basket contaminated; trust clean universe tags for
    # the rest (faster, lower-risk).
    uni = pd.read_excel(PATHS["universe"])
    uni_name = dict(zip(uni["yf_ticker"].astype(str), uni["name"].astype(str)))
    force = {s for s in all_syms
             if s not in uni_name or uni_name[s].startswith("GS ")}
    print(f"force-reclassifying {len(force)} contaminated/missing "
          f"(trusting universe for {len(all_syms) - len(force)} clean)\n")
    result = resolve_tags(all_syms, name_hints=name_hints, force=force)

    print("\n" + "=" * 96)
    print(f"{'SYMBOL':15s} {'SOURCE':18s} {'NAME':32s} TAGS")
    print("=" * 96)
    for s in all_syms:
        r = result[s]
        print(f"{s:15s} {r['source']:18s} {(r.get('name') or s)[:31]:32s} "
              f"{r['tags'] or '(none)'}")

    from collections import Counter
    print("\nby source:", dict(Counter(r["source"] for r in result.values())))
    empty = [s for s in all_syms if not result[s]["tags"]]
    if empty:
        print("STILL EMPTY (need manual override):", empty)
