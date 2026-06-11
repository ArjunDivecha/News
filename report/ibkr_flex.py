#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: ibkr_flex.py
=============================================================================

INPUT FILES:
    (none - makes two HTTPS GET calls to IBKR Flex Web Service)

OUTPUT FILES:
    (none - returns a normalized DataFrame matching the holdings format)

VERSION: 1.0
LAST UPDATED: 2026-06-10
AUTHOR: Arjun Divecha

DESCRIPTION:
    Fetch IBKR portfolio positions via the Flex Web Service — a token-based
    HTTP API that requires NO TWS login. One-time setup in Client Portal:
      1. Performance & Reports → Flex Queries → create a query with
         "Open Positions" and "Cash Transactions" sections
      2. Flex Web Service Configuration → Generate Token → note the token
         and the query ID
      3. Put IBKR_FLEX_TOKEN and IBKR_FLEX_QUERY_ID in .env

    Every run after: two HTTPS GETs → XML → normalized DataFrame.

    FLEX DATA NOTES:
      - Data is delayed (not real-time); positions reflect the most recent
        settlement. Fine for end-of-day reporting.
      - The XML does NOT contain live market value — the OpenPositions
        section has markPrice, so we compute: market_value = position × markPrice.
      - Futures positions are tagged as .FUT so they are never priced like stocks.
      - Cash is read from the CashTransactions section (or defaults to
        $0 when that section is not present in the query result).

DEPENDENCIES:
    - requests (stdlib fallback: urllib)
    - xml.etree.ElementTree (stdlib)

USAGE:
    from ibkr_flex import fetch_positions
    df = fetch_positions(token="...", query_id="12345")
=============================================================================
"""

import json
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

# IBKR Flex Web Service endpoints (version 3)
_SEND_URL = (
    "https://gdcdyn.interactivebrokers.com/Universal/servlet/"
    "FlexStatementService.SendRequest"
)
_FETCH_URL = (
    "https://gdcdyn.interactivebrokers.com/Universal/servlet/"
    "FlexStatementService.GetStatement"
)
_USER_AGENT = "Python-ibkr_flex/1.0"

# How many times to poll for the statement before giving up
_MAX_POLLS = 6
_POLL_WAIT_S = 5.0

SEC_TYPE_FUT = "FUT"
SEC_TYPE_OPT = "OPT"
SEC_TYPE_FOP = "FOP"
SEC_TYPE_CASH = "CASH"


def _http_get(url: str, timeout: int = 30) -> str:
    """GET an HTTPS URL and return the response body as a string."""
    req = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def _request_report(token: str, query_id: str) -> str:
    """Send a Flex request; extract and return the reference code."""
    url = f"{_SEND_URL}?t={token}&q={query_id}&v=3"
    body = _http_get(url)
    root = ET.fromstring(body)

    status = root.findtext("Status", "")
    if status and status != "Success":
        code = root.findtext("ErrorCode", "")
        msg = root.findtext("ErrorMessage", "")
        raise RuntimeError(
            f"Flex SendRequest failed: {status} "
            f"(error {code}: {msg})")

    ref = root.findtext("ReferenceCode") or root.findtext("referenceCode")
    if not ref:
        raise RuntimeError(
            f"Flex SendRequest returned no ReferenceCode. "
            f"Root element: {root.tag}, children: "
            f"{[c.tag for c in root[:5]]}")

    return ref.strip()


def _fetch_statement(token: str, ref_code: str, max_polls: int = _MAX_POLLS,
                     wait_s: float = _POLL_WAIT_S) -> str:
    """Poll the GetStatement endpoint until the report is ready."""
    url_base = f"{_FETCH_URL}?t={token}&q={ref_code}&v=3"
    for attempt in range(1, max_polls + 1):
        body = _http_get(url_base)
        root = ET.fromstring(body)

        status = root.findtext("Status", "")
        if status == "Warn":
            # Statement not ready yet — the XML will have an error message
            err = root.findtext("ErrorMessage", "not ready")
            if attempt < max_polls:
                print(f"    Flex statement not ready yet ({err}); "
                      f"retry {attempt}/{max_polls} in {wait_s}s...")
                time.sleep(wait_s)
                continue
            raise RuntimeError(
                f"Flex statement never became ready after {max_polls} "
                f"polls: {err}")
        if status and status != "Success":
            code = root.findtext("ErrorCode", "")
            msg = root.findtext("ErrorMessage", "")
            raise RuntimeError(
                f"Flex GetStatement failed: {status} (error {code}: {msg})")
        return body

    raise RuntimeError(
        f"Flex statement not ready after {max_polls} poll cycles")


# ---------------------------------------------------------------------------
# XML → DataFrame
# ---------------------------------------------------------------------------

def _parse_positions(xml_str: str) -> pd.DataFrame:
    """Parse OpenPositions (and optionally CashTransactions) from Flex XML."""
    root = ET.fromstring(xml_str)

    # The XML is namespaced; IBKR may or may not use explicit prefixes.
    # Find FlexStatements → FlexStatement → OpenPositions → OpenPosition
    ns = {}
    tag_elem = root.tag
    if "}" in tag_elem:
        ns = {"ns": tag_elem.split("}")[0].lstrip("{")}

    def _find_all(parent, tag):
        if ns:
            return parent.findall(f"ns:{tag}", ns)
        return parent.findall(tag)

    def _attrib(el, tag, default=""):
        """Read an XML attribute (case-insensitive). IBKR Flex uses attributes
        on elements (e.g. <OpenPosition symbol='IWF'>), not child text nodes."""
        val = el.get(tag)
        if val is not None:
            return val
        # Try lowercase variant
        val = el.get(tag.lower())
        if val is not None:
            return val
        return default

    # Locate the statement(s) and position lists.
    # Structure: FlexQueryResponse → FlexStatements → FlexStatement → OpenPositions → OpenPosition
    flex_statements = _find_all(root, "FlexStatements")
    if not flex_statements:
        # Some responses omit the FlexStatements wrapper
        flex_statements = [root]

    fetched_at = datetime.now().isoformat(timespec="seconds")
    rows = []
    for fs in flex_statements:
        # Some formats wrap each account in FlexStatement, others don't.
        # Try to find FlexStatement children; if none, treat fs itself as
        # the statement (flat format).
        stmts = _find_all(fs, "FlexStatement") or [fs]
        for stmt in stmts:
            acct_id = _attrib(stmt, "accountId") or _attrib(stmt, "AccountId")

            # --- Positions ---
            for pos_container in _find_all(stmt, "OpenPositions"):
                for pos in _find_all(pos_container, "OpenPosition"):
                    symbol = _attrib(pos, "symbol")
                    if not symbol:
                        continue
                    qty = float(_attrib(pos, "position", "0"))
                    if qty == 0:
                        continue
                    cost = float(_attrib(pos, "costBasisPrice", "0"))
                    mark = float(_attrib(pos, "markPrice", "0"))
                    mv = qty * mark if mark else None
                    pnl_str = _attrib(pos, "fifoPnlUnrealized") or _attrib(pos, "unrealizedPnl")
                    pnl = float(pnl_str) if pnl_str else None
                    asset_cat = _attrib(pos, "assetCategory", "STK")

                    rows.append({
                        "account": acct_id or "",
                        "symbol": symbol,
                        "quantity": qty,
                        "avg_price": cost,
                        "market_value": mv,
                        "open_pnl": pnl,
                        "broker": "IBKR",
                        "fetched_at": fetched_at,
                        "_asset_category": asset_cat,
                    })

            # ---- Cash transactions (the statement-level cash balance) ----
            for ct_section in _find_all(stmt, "CashTransactions"):
                for ct in _find_all(ct_section, "CashTransaction"):
                    ccy = _attrib(ct, "currency", "")
                    if ccy and ccy != "USD":
                        continue
                    amt_str = _attrib(ct, "endingCash") or _attrib(ct, "amount")
                    if amt_str:
                        rows.append({
                            "account": acct_id or "",
                            "symbol": "CASH",
                            "quantity": float(amt_str),
                            "avg_price": 1.0,
                            "market_value": float(amt_str),
                            "open_pnl": 0.0,
                            "broker": "IBKR",
                            "fetched_at": fetched_at,
                            "_asset_category": SEC_TYPE_CASH,
                        })
                        break  # one cash row per account

    if not rows:
        raise RuntimeError(
            "Flex report contained no positions. Check that your query "
            "includes an 'Open Positions' section.")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fetch_positions(token: str, query_id: str) -> pd.DataFrame:
    """
    Fetch and parse IBKR positions via Flex Web Service.

    Args:
        token: Flex Web Service token from Client Portal.
        query_id: Numeric query ID from Client Portal.

    Returns:
        Normalized DataFrame with columns:
        account, symbol, quantity, avg_price, market_value, open_pnl,
        broker='IBKR', fetched_at (ISO timestamp), _asset_category.
    """
    print(f"    IBKR Flex: sending request...")
    ref = _request_report(token, query_id)
    print(f"    IBKR Flex: fetching statement (ref {ref})...")
    xml_str = _fetch_statement(token, ref)
    df = _parse_positions(xml_str)

    # Tag non-equity symbols (.FUT, .OPT, .FOP) so they are never confused
    # with equity tickers in the pricing step
    non_stock = df["_asset_category"].isin([SEC_TYPE_FUT, SEC_TYPE_OPT, SEC_TYPE_FOP])
    df.loc[non_stock, "symbol"] = (
        df.loc[non_stock, "symbol"].astype(str)
        + "." + df.loc[non_stock, "_asset_category"])

    df = df.drop(columns=["_asset_category"])
    print(f"    IBKR: {len(df)} positions via Flex")
    return df
