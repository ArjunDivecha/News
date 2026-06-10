#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: MemeFinder.py
=============================================================================

DESCRIPTION:
    This script discovers trending meme stocks by searching the web and social
    media for stock discussions. It uses the Exa API to run multiple search
    queries (WSB trending stocks, short squeeze candidates, StockTwits
    mentions), extracts ticker candidates from search results using regex
    patterns ($TICKER, exchange:TICKER, /stock/TICKER paths, bare all-caps
    tokens with financial context), aggregates candidates across queries and
    source domains, validates each candidate against Yahoo Finance as a real
    equity with sufficient trading history, scores them by mention breadth,
    source diversity, volume spike, and price change, and optionally
    cross-references against the curated Final 1000 universe. Results are
    written to a JSON output file with full run diagnostics.

INPUT FILES:
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/News/.env
        Environment file containing the EXA_API_KEY required for Exa web
        search API access. Loaded via load_dotenv().
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/News/Step 2 Data Processing - Final1000/Final 1000 Asset Master List.xlsx
        (Optional) Excel workbook of curated asset metadata (Bloomberg_Ticker,
        Name, category_tier1, category_tier2, category_tags). Used to tag
        whether detected meme stock symbols overlap with this known universe.
        Read via pd.read_excel().

OUTPUT FILES:
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/News/Step 3 Data Analysis/search_based_meme_stocks.json
        JSON file containing detected meme stocks plus run diagnostics:
        timestamp, search queries executed, run statistics (candidates
        extracted, analyzed, qualified, rejected), full meme stock objects
        with scoring metadata, and top rejected candidates. Written via
        Path.write_text(json.dumps(...)).

VERSION: 1.0
LAST UPDATED: 2026-06-05
AUTHOR: Arjun Divecha

DEPENDENCIES:
    - argparse
    - json
    - os
    - re
    - collections (defaultdict)
    - dataclasses
    - datetime
    - pathlib
    - typing
    - urllib.parse
    - pandas
    - yfinance
    - python-dotenv
    - exa_py

USAGE:
    python MemeFinder.py [--max-candidates N] [--output PATH]

NOTES:
    - Requires a valid EXA_API_KEY in the .env file at the project root.
    - Requires internet access for Exa web search and Yahoo Finance market data.
    - The Final 1000 asset master list is optional; the script runs without it.
    - Yahoo Finance fetching is rate-limited; a larger --max-candidates value
      increases runtime proportionally.
=============================================================================
"""

import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from exa_py import Exa


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("search_based_meme_stocks.json")
FINAL_1000_PATH = PROJECT_ROOT / "Step 2 Data Processing - Final1000" / "Final 1000 Asset Master List.xlsx"

SEARCH_QUERIES: List[Tuple[str, List[str]]] = [
    ("wallstreetbets trending stocks today", ["apewisdom.io", "quiverquant.com", "swaggystocks.com"]),
    ("WSB YOLO plays this week", ["apewisdom.io", "quiverquant.com"]),
    ("meme stocks trending today", []),
    ("reddit short squeeze stocks today", []),
    ("trending stocks stocktwits", ["stocktwits.com"]),
    ("short squeeze stocks today", []),
]

FINANCE_CONTEXT_KEYWORDS = {
    "stock",
    "stocks",
    "shares",
    "ticker",
    "tickers",
    "call",
    "calls",
    "put",
    "puts",
    "squeeze",
    "short",
    "volume",
    "bullish",
    "bearish",
    "yolo",
    "reddit",
    "meme",
    "squeeze",
    "float",
    "gamma",
    "wallstreetbets",
    "stocktwits",
}

GENERIC_EXCLUDES = {
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HER",
    "WSB", "CEO", "USA", "API", "ETF", "IPO", "FAQ", "PDF", "URL", "SEC",
    "IT", "GO", "AM", "SO", "UP", "AT", "IN", "ON", "TO", "OF", "OR", "BY",
    "GET", "NEW", "NOW", "DAY", "SEE", "USE", "TOP", "HOT", "BIG", "LOW",
    "HIGH", "LONG", "PUT", "CALL", "BUY", "SELL", "INC", "LLC", "LTD",
    "PLC", "CORP", "CO", "GROUP", "HOLD", "HOLDING", "HOLDINGS", "BANK",
    "NYSE", "NASDAQ", "JAMES", "JOHN", "JULIE", "CHASE", "HOYLE",
}

MAJOR_TICKER_EXCLUDES = {
    "SPY", "QQQ", "VOO", "IWM", "DIA", "VIX", "TQQQ", "SQQQ",
    "NVDA", "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "TSLA", "TSM",
}

AMBIGUOUS_WORD_SYMBOLS = {
    "AI", "AN", "ANY", "AS", "BACK", "BE", "CASH", "COM", "FREE", "HAS",
    "HERE", "MOST", "MORE", "NEXT", "OUT", "REAL", "SOME", "TIME", "TWO",
    "VIEW", "YEAR",
}

ALLOWED_QUOTE_TYPES = {"EQUITY"}

DOLLAR_TICKER_RE = re.compile(r"\$([A-Z][A-Z\.-]{0,4})\b")
EXCHANGE_TICKER_RE = re.compile(
    r"\b(?:NASDAQ|NYSE|NYSEARCA|NYSEAMERICAN|AMEX|OTC|TSX|LSE)\s*[:\-]\s*([A-Z][A-Z\.-]{0,4})\b",
    re.IGNORECASE,
)
SITE_PATH_RE = re.compile(r"/(?:stock|stocks)/([A-Z][A-Z\.-]{0,4})(?:[/?#]|\b)", re.IGNORECASE)
BARE_TICKER_RE = re.compile(r"\b([A-Z]{2,5})\b")


@dataclass
class Candidate:
    symbol: str
    mentions: int = 0
    sources: Set[str] = field(default_factory=set)
    queries: Set[str] = field(default_factory=set)
    evidence_types: Set[str] = field(default_factory=set)
    contexts: List[str] = field(default_factory=list)

    @property
    def source_count(self) -> int:
        return len(self.sources)

    @property
    def query_count(self) -> int:
        return len(self.queries)


def normalize_symbol(raw_symbol: str) -> str:
    """Normalize raw ticker strings into Yahoo-compatible symbols."""
    symbol = raw_symbol.upper().strip().strip("$")
    symbol = symbol.replace(".", "-")
    return symbol


def clean_scalar(value: Any) -> Any:
    """Convert pandas/NaN scalar values into JSON-safe values."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if pd.isna(value):
        return None
    return value


def is_blocked_symbol(symbol: str) -> bool:
    """Return True for tokens that are clearly not useful ticker candidates."""
    return (
        len(symbol) < 2
        or len(symbol) > 5
        or symbol in GENERIC_EXCLUDES
        or symbol in MAJOR_TICKER_EXCLUDES
    )


def extract_context(text: str, start: int, end: int, width: int = 100) -> str:
    """Extract a compact context window around a candidate symbol."""
    snippet = text[max(0, start - width):min(len(text), end + width)]
    return " ".join(snippet.split())


def bare_symbol_has_context(symbol: str, text: str, start: int, end: int) -> bool:
    """Require stronger context for bare all-caps tokens."""
    window = text[max(0, start - 90):min(len(text), end + 90)]
    window_lower = window.lower()
    keyword_hits = sum(1 for keyword in FINANCE_CONTEXT_KEYWORDS if keyword in window_lower)
    has_path_hint = f"/stock/{symbol.lower()}" in window_lower or f"/stocks/{symbol.lower()}" in window_lower
    has_explicit_hint = any(token in window_lower for token in ["$", "ticker", "stock", "shares"])
    return has_path_hint or (has_explicit_hint and keyword_hits >= 1) or keyword_hits >= 2


def add_local_candidate(
    local_hits: Dict[str, Dict[str, Any]],
    symbol: str,
    evidence_type: str,
    text: str,
    start: int,
    end: int,
) -> None:
    """Add a candidate hit for a single search result."""
    symbol = normalize_symbol(symbol)
    if is_blocked_symbol(symbol):
        return

    if symbol not in local_hits:
        local_hits[symbol] = {
            "evidence_types": set(),
            "contexts": [],
        }

    local_hits[symbol]["evidence_types"].add(evidence_type)
    context = extract_context(text, start, end)
    if context and context not in local_hits[symbol]["contexts"]:
        local_hits[symbol]["contexts"].append(context)


def extract_candidates_from_text(text: str) -> Dict[str, Dict[str, Any]]:
    """Extract ticker candidates from a single search result body."""
    local_hits: Dict[str, Dict[str, Any]] = {}

    for match in DOLLAR_TICKER_RE.finditer(text):
        add_local_candidate(local_hits, match.group(1), "dollar", text, match.start(1), match.end(1))

    for match in EXCHANGE_TICKER_RE.finditer(text):
        add_local_candidate(local_hits, match.group(1), "exchange", text, match.start(1), match.end(1))

    for match in SITE_PATH_RE.finditer(text):
        add_local_candidate(local_hits, match.group(1), "site_path", text, match.start(1), match.end(1))

    for match in BARE_TICKER_RE.finditer(text):
        symbol = normalize_symbol(match.group(1))
        if is_blocked_symbol(symbol):
            continue
        if symbol in AMBIGUOUS_WORD_SYMBOLS:
            continue
        if bare_symbol_has_context(symbol, text, match.start(1), match.end(1)):
            add_local_candidate(local_hits, symbol, "bare", text, match.start(1), match.end(1))

    return local_hits


def candidate_priority(candidate: Candidate) -> float:
    """Rank candidates before Yahoo validation."""
    evidence_bonus = 0.0
    if "dollar" in candidate.evidence_types:
        evidence_bonus += 2.0
    if "exchange" in candidate.evidence_types:
        evidence_bonus += 1.5
    if "site_path" in candidate.evidence_types:
        evidence_bonus += 1.0
    if "bare" in candidate.evidence_types:
        evidence_bonus += 0.5
    return (
        candidate.mentions * 3.0
        + candidate.source_count * 2.0
        + candidate.query_count * 1.5
        + evidence_bonus
    )


def load_curated_universe() -> Dict[str, Dict[str, Any]]:
    """Load ticker metadata from the Final 1000 workbook when available."""
    if not FINAL_1000_PATH.exists():
        return {}

    df = pd.read_excel(FINAL_1000_PATH)
    universe: Dict[str, Dict[str, Any]] = {}

    for _, row in df.iterrows():
        raw_ticker = str(row.get("Bloomberg_Ticker", "")).strip()
        if not raw_ticker or raw_ticker == "nan":
            continue
        symbol = normalize_symbol(raw_ticker.split()[0])
        universe[symbol] = {
            "name": clean_scalar(row.get("Name")),
            "tier1": clean_scalar(row.get("category_tier1")),
            "tier2": clean_scalar(row.get("category_tier2")),
            "tags": clean_scalar(row.get("category_tags")),
        }

    return universe


def get_domain(url: str) -> str:
    """Extract a normalized domain from a URL."""
    try:
        return urlparse(url).netloc or url
    except ValueError:
        return url


def get_quote_type(info: Dict[str, Any]) -> str:
    """Get the Yahoo quote type in upper-case form."""
    return str(info.get("quoteType") or "").upper().strip()


def fetch_market_profile(symbol: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Fetch Yahoo Finance data and validate the symbol as a tradable equity."""
    stock = yf.Ticker(symbol)

    try:
        hist = stock.history(period="3mo", interval="1d", auto_adjust=False)
    except Exception as exc:
        return None, f"history_error: {exc}"

    if hist.empty or len(hist) < 15:
        return None, "insufficient_history"

    info: Dict[str, Any] = {}
    try:
        info = stock.info or {}
    except Exception:
        info = {}

    quote_type = get_quote_type(info)
    if quote_type and quote_type not in ALLOWED_QUOTE_TYPES:
        return None, f"quote_type:{quote_type.lower()}"

    if quote_type == "" and symbol in GENERIC_EXCLUDES:
        return None, "generic_without_quote_type"

    current_volume = float(hist["Volume"].tail(3).mean())
    avg_volume = float(hist["Volume"].iloc[:-3].mean()) if len(hist) > 3 else float(hist["Volume"].mean())
    if avg_volume <= 0:
        return None, "zero_average_volume"

    current_price = float(hist["Close"].iloc[-1])
    price_change_1d = float(((current_price - hist["Close"].iloc[-2]) / hist["Close"].iloc[-2]) * 100)
    price_change_5d = float(((current_price - hist["Close"].iloc[-6]) / hist["Close"].iloc[-6]) * 100)
    volume_spike = float(current_volume / avg_volume)

    market_cap = info.get("marketCap") or 0
    market_cap_b = round(float(market_cap) / 1e9, 2) if market_cap else 0
    name = info.get("longName") or info.get("shortName") or symbol
    exchange = info.get("exchange") or info.get("fullExchangeName")

    return {
        "name": name,
        "quote_type": quote_type or "UNKNOWN",
        "exchange": exchange,
        "market_cap_b": market_cap_b,
        "current_volume": int(current_volume),
        "avg_volume": int(avg_volume),
        "volume_spike": round(volume_spike, 2),
        "price_change_1d": round(price_change_1d, 2),
        "price_change_5d": round(price_change_5d, 2),
    }, None


def candidate_reasons(candidate: Candidate, profile: Dict[str, Any]) -> List[str]:
    """Build the list of reasons a symbol qualifies as meme-like."""
    reasons: List[str] = []
    has_breadth = candidate.mentions >= 2 and (candidate.source_count >= 2 or candidate.query_count >= 2)
    has_explicit_evidence = bool(candidate.evidence_types.intersection({"dollar", "exchange", "site_path"}))

    if not has_breadth:
        return reasons
    if not has_explicit_evidence and candidate.mentions < 3:
        return reasons
    if profile["market_cap_b"] >= 100 and not (
        candidate.mentions >= 5 and candidate.source_count >= 3 and has_explicit_evidence
    ):
        return reasons

    if (
        candidate.mentions >= 5
        and candidate.source_count >= 3
        and has_explicit_evidence
    ):
        reasons.append(f"{candidate.mentions} explicit mentions across {candidate.source_count} sources")
    if profile["volume_spike"] >= 1.5:
        reasons.append(f"{profile['volume_spike']:.1f}x volume")
    if abs(profile["price_change_1d"]) >= 8:
        reasons.append(f"{profile['price_change_1d']:+.1f}% (1d)")
    if abs(profile["price_change_5d"]) >= 12:
        reasons.append(f"{profile['price_change_5d']:+.1f}% (5d)")

    return reasons


def compute_meme_score(candidate: Candidate, profile: Dict[str, Any]) -> float:
    """Score a symbol using breadth of mentions and market behavior."""
    return round(
        (
            candidate.mentions * 2.5
            + candidate.source_count * 1.5
            + candidate.query_count
        )
        * max(profile["volume_spike"], 1.0)
        * (1 + abs(profile["price_change_1d"]) / 100 + abs(profile["price_change_5d"]) / 200),
        2,
    )


def search_candidates(exa: Exa) -> Dict[str, Candidate]:
    """Run Exa searches and aggregate ticker candidates across results."""
    all_candidates: Dict[str, Candidate] = {}

    print("Searching for meme-stock discussion across multiple sources...")
    print("=" * 70)
    print()

    for query, domains in SEARCH_QUERIES:
        print(f"Query: {query}")
        search_params: Dict[str, Any] = {
            "query": query,
            "text": True,
            "num_results": 10,
            "type": "auto",
        }
        if domains:
            search_params["include_domains"] = domains

        try:
            results = exa.search_and_contents(**search_params)
        except Exception as exc:
            print(f"  Error: {exc}")
            continue

        query_symbols: Set[str] = set()
        print(f"  Results fetched: {len(results.results)}")

        for result in results.results:
            text_parts = [
                getattr(result, "title", "") or "",
                getattr(result, "text", "") or "",
            ]
            text = "\n".join(part for part in text_parts if part)
            if not text:
                continue

            domain = get_domain(getattr(result, "url", ""))
            local_hits = extract_candidates_from_text(text)
            query_symbols.update(local_hits.keys())

            for symbol, hit_data in local_hits.items():
                candidate = all_candidates.setdefault(symbol, Candidate(symbol=symbol))
                candidate.mentions += 1
                candidate.sources.add(domain)
                candidate.queries.add(query)
                candidate.evidence_types.update(hit_data["evidence_types"])
                for context in hit_data["contexts"][:2]:
                    if context not in candidate.contexts:
                        candidate.contexts.append(context)

        print(f"  Unique symbols extracted from query: {len(query_symbols)}")

    print()
    print("=" * 70)
    print("Aggregating and validating candidates...")
    print("=" * 70)
    print()

    return all_candidates


def analyze_candidates(
    candidates: Dict[str, Candidate],
    curated_universe: Dict[str, Dict[str, Any]],
    max_candidates: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """Validate top candidates and return meme hits plus rejected diagnostics."""
    ranked_candidates = sorted(candidates.values(), key=candidate_priority, reverse=True)
    meme_stocks: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    analyzed_count = 0

    for candidate in ranked_candidates[:max_candidates]:
        analyzed_count += 1
        print(
            f"Analyzing ${candidate.symbol} "
            f"(mentions={candidate.mentions}, sources={candidate.source_count}, queries={candidate.query_count})...",
            end=" ",
        )

        profile, rejection_reason = fetch_market_profile(candidate.symbol)
        if rejection_reason:
            print(f"rejected [{rejection_reason}]")
            rejected.append(
                {
                    "symbol": candidate.symbol,
                    "mentions": candidate.mentions,
                    "sources": candidate.source_count,
                    "queries": candidate.query_count,
                    "reason": rejection_reason,
                }
            )
            continue

        reasons = candidate_reasons(candidate, profile)
        if not reasons:
            print(
                f"no trigger (vol={profile['volume_spike']:.1f}x, "
                f"1d={profile['price_change_1d']:+.1f}%, 5d={profile['price_change_5d']:+.1f}%)"
            )
            rejected.append(
                {
                    "symbol": candidate.symbol,
                    "mentions": candidate.mentions,
                    "sources": candidate.source_count,
                    "queries": candidate.query_count,
                    "reason": "no_meme_trigger",
                }
            )
            continue

        curated = curated_universe.get(candidate.symbol, {})
        meme_score = compute_meme_score(candidate, profile)
        meme_stocks.append(
            {
                "symbol": candidate.symbol,
                "name": profile["name"],
                "mentions": candidate.mentions,
                "source_count": candidate.source_count,
                "query_count": candidate.query_count,
                "sources": sorted(candidate.sources),
                "queries": sorted(candidate.queries),
                "evidence_types": sorted(candidate.evidence_types),
                "volume_spike": profile["volume_spike"],
                "price_change_1d": profile["price_change_1d"],
                "price_change_5d": profile["price_change_5d"],
                "current_volume": profile["current_volume"],
                "avg_volume": profile["avg_volume"],
                "market_cap_b": profile["market_cap_b"],
                "exchange": profile["exchange"],
                "quote_type": profile["quote_type"],
                "reasons": reasons,
                "sample_context": candidate.contexts[:3],
                "meme_score": meme_score,
                "in_curated_universe": bool(curated),
                "curated_tier1": curated.get("tier1"),
                "curated_tier2": curated.get("tier2"),
                "curated_tags": curated.get("tags"),
            }
        )
        print(f"qualified [{', '.join(reasons)}]")

    meme_stocks.sort(key=lambda stock: stock["meme_score"], reverse=True)
    rejected.sort(key=lambda row: (-row["mentions"], -row["sources"], row["symbol"]))
    return meme_stocks, rejected, analyzed_count


def build_output(
    candidates: Dict[str, Candidate],
    meme_stocks: List[Dict[str, Any]],
    rejected: List[Dict[str, Any]],
    analyzed_count: int,
) -> Dict[str, Any]:
    """Assemble the JSON payload."""
    in_universe = sum(1 for stock in meme_stocks if stock.get("in_curated_universe"))

    return {
        "timestamp": datetime.now().isoformat(),
        "search_queries": [query for query, _ in SEARCH_QUERIES],
        "run_stats": {
            "candidates_extracted": len(candidates),
            "candidates_analyzed": analyzed_count,
            "qualified_count": len(meme_stocks),
            "rejected_count": len(rejected),
            "curated_universe_matches": in_universe,
        },
        "total_found": len(meme_stocks),
        "meme_stocks": meme_stocks,
        "top_rejected_candidates": rejected[:15],
    }


def print_summary(meme_stocks: List[Dict[str, Any]]) -> None:
    """Print a compact run summary."""
    print()
    print("=" * 70)
    print("MEME STOCK RESULTS")
    print("=" * 70)
    print()

    if not meme_stocks:
        print("No meme stocks found with the current criteria.")
        return

    for idx, stock in enumerate(meme_stocks, 1):
        print(f"{idx}. ${stock['symbol']} - {stock['name']}")
        print(
            f"   Mentions: {stock['mentions']} across {stock['source_count']} sources "
            f"and {stock['query_count']} queries"
        )
        print(
            f"   Price/Volume: {stock['price_change_1d']:+.2f}% (1d), "
            f"{stock['price_change_5d']:+.2f}% (5d), {stock['volume_spike']:.2f}x volume"
        )
        if stock["market_cap_b"] > 0:
            print(f"   Market Cap: ${stock['market_cap_b']:.2f}B")
        if stock.get("in_curated_universe"):
            print(
                f"   Curated Universe: Yes ({stock.get('curated_tier1')} / "
                f"{stock.get('curated_tier2')})"
            )
        else:
            print("   Curated Universe: No")
        print(f"   Reasons: {', '.join(stock['reasons'])}")
        print(f"   Meme Score: {stock['meme_score']:.2f}")
        if stock["sample_context"]:
            print(f"   Context: ...{stock['sample_context'][0][:120]}...")
        print()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Discover trending meme stocks from web/social search")
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=80,
        help="Maximum ranked candidates to validate against Yahoo Finance",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to write the JSON output",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    exa_api_key = os.getenv("EXA_API_KEY")
    if not exa_api_key:
        raise RuntimeError("EXA_API_KEY is missing from the environment.")

    exa = Exa(api_key=exa_api_key)
    curated_universe = load_curated_universe()

    candidates = search_candidates(exa)
    if not candidates:
        print("No ticker candidates were extracted from search results.")
        output = build_output({}, [], [], 0)
        Path(args.output).write_text(json.dumps(output, indent=2))
        return 0

    print(f"Candidates extracted before market validation: {len(candidates)}")
    print(f"Curated universe loaded: {len(curated_universe)} tickers")
    print()

    meme_stocks, rejected, analyzed_count = analyze_candidates(
        candidates,
        curated_universe,
        args.max_candidates,
    )
    output = build_output(candidates, meme_stocks, rejected, analyzed_count)

    output_path = Path(args.output).expanduser()
    output_path.write_text(json.dumps(output, indent=2))

    print_summary(meme_stocks)
    print(f"Results saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
