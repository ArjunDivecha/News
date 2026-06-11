#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: holdings.py
=============================================================================

INPUT FILES:
    - ~/.schwabdev/tokens.db          (Schwab OAuth token store, via schwabdev)
    - IBKR Flex Web Service            (primary: HTTPS API, token-based, no TWS)
    - IBKR TWS/Gateway on 127.0.0.1   (fallback: via report/ibkr_fetch.py subprocess)
    - data/holdings.xlsx              (previous snapshot, stale fallback)
    - Client.xlsx                     (legacy snapshot, first-run fallback)

OUTPUT FILES:
    - data/holdings.xlsx
      Normalized holdings snapshot. Columns: account, symbol, quantity,
      avg_price, market_value, open_pnl, broker, fetched_at.

VERSION: 1.0
LAST UPDATED: 2026-06-09
AUTHOR: Arjun Divecha

DESCRIPTION:
    Pulls live holdings from Schwab (schwabdev) and IBKR. For IBKR, the
    PRIMARY path is now the Flex Web Service — a token-based HTTPS API that
    requires NO TWS login. Set IBKR_FLEX_TOKEN + IBKR_FLEX_QUERY_ID in .env
    and TWS is never needed. The old TWS subprocess path is kept as a
    fallback that activates only when Flex env vars are absent.

    PREFLIGHTS (user-in-the-loop, per design decision):
      - Schwab: reads refresh_token_issued from ~/.schwabdev/tokens.db.
        If the 7-day refresh token is expired/near expiry, prompts the user.
      - IBKR (Flex): NO preflight — the token is stateless, no login needed.
        The run just works.
      - IBKR (TWS fallback, only when Flex env is unset): probes the TWS
        port; if closed, launches TWS and prompts for login.

    FALLBACK (explicitly approved): if a broker still fails after the
    prompts, the run continues on the LAST SAVED snapshot, loudly stamped
    as stale with its as-of date.

DEPENDENCIES:
    - schwabdev, pandas, openpyxl

USAGE:
    from holdings import get_holdings
    positions, meta = get_holdings(interactive=True)
=============================================================================
"""

import json
import os
import sqlite3
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PATHS, SETTINGS, REPORT_DIR

SCHWAB_TOKENS_DB = Path.home() / ".schwabdev" / "tokens.db"
REFRESH_TOKEN_LIFETIME_S = 7 * 24 * 3600          # Schwab refresh tokens last 7 days
REFRESH_TOKEN_WARN_S = int(6.5 * 24 * 3600)       # warn at 6.5 days

HOLDINGS_COLUMNS = ["account", "symbol", "quantity", "avg_price",
                    "market_value", "open_pnl", "broker", "fetched_at"]


class BrokerError(RuntimeError):
    """Raised when a broker pull fails after all prompts."""


# ---------------------------------------------------------------------------
# Preflights
# ---------------------------------------------------------------------------

def schwab_token_age_s() -> Optional[float]:
    """Age of the Schwab refresh token in seconds, or None if no token."""
    if not SCHWAB_TOKENS_DB.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{SCHWAB_TOKENS_DB}?mode=ro", uri=True)
        row = conn.execute(
            "SELECT refresh_token_issued FROM schwabdev LIMIT 1").fetchone()
        conn.close()
    except Exception:
        return None
    if not row or not row[0]:
        return None
    issued = pd.Timestamp(row[0])
    if issued.tzinfo is not None:
        now = pd.Timestamp.now(tz=issued.tzinfo)
    else:
        now = pd.Timestamp.now()
    return (now - issued).total_seconds()


def preflight_schwab(interactive: bool) -> None:
    """
    Verify the Schwab refresh token is usable; trigger interactive re-auth
    with the user's consent when it is expired or near expiry.
    """
    age = schwab_token_age_s()
    if age is None:
        msg = "Schwab token store missing/unreadable (~/.schwabdev/tokens.db)"
        if not interactive:
            raise BrokerError(msg + " and run is non-interactive")
        print(f"\n  !! {msg}")
        print("  schwabdev will now start its interactive authorization "
              "(browser window + paste-the-redirect-URL).")
        input("  Press Enter to begin Schwab re-authorization... ")
        return  # Client construction below performs the actual flow

    days = age / 86400
    if age < REFRESH_TOKEN_WARN_S:
        print(f"  Schwab refresh token OK (issued {days:.1f} days ago)")
        return

    msg = (f"Schwab refresh token is {days:.1f} days old "
           f"(hard expiry at 7 days)")
    if not interactive:
        raise BrokerError(msg + " and run is non-interactive")
    print(f"\n  !! {msg}")
    print("  schwabdev will start its interactive re-authorization "
          "(browser + paste redirect URL).")
    input("  Press Enter to begin Schwab re-authorization... ")


def ibkr_port_open(port: int, timeout: float = 2.0) -> bool:
    """True if something is listening on the IBKR API port."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def _try_launch_tws() -> bool:
    """Try to launch TWS/IB Gateway on macOS. Returns True if launched."""
    candidates = ["Trader Workstation", "IB Gateway", "Trader Workstation 10"]
    for app in candidates:
        result = subprocess.run(["open", "-a", app], capture_output=True)
        if result.returncode == 0:
            print(f"  Launched '{app}' - waiting for login...")
            return True
    return False


def preflight_ibkr(interactive: bool, port: int) -> None:
    """
    Verify TWS/Gateway is reachable; launch it and prompt the user to log
    in when it is not.
    """
    if ibkr_port_open(port):
        print(f"  IBKR API port {port} reachable")
        return

    msg = f"IBKR TWS/Gateway not reachable on 127.0.0.1:{port}"
    if not interactive:
        raise BrokerError(msg + " and run is non-interactive")

    print(f"\n  !! {msg}")
    launched = _try_launch_tws()
    if not launched:
        print("  Could not auto-launch TWS. Please start TWS or IB Gateway "
              "manually and log in.")

    for attempt in range(1, 11):
        answer = input(
            f"  [{attempt}/10] Press Enter once TWS is logged in "
            f"(API port {port}), or type 'skip' to fall back to stale "
            f"holdings: ").strip().lower()
        if answer == "skip":
            raise BrokerError("User chose to skip IBKR connection")
        # TWS needs a few seconds after login before the API socket opens
        for _ in range(10):
            if ibkr_port_open(port):
                print("  IBKR API port is now reachable")
                return
            time.sleep(2)
        print("  Port still closed - is the API enabled in TWS "
              "(File > Global Configuration > API > Settings)?")
    raise BrokerError("IBKR port never became reachable")


# ---------------------------------------------------------------------------
# Broker pulls
# ---------------------------------------------------------------------------

def fetch_schwab() -> pd.DataFrame:
    """Pull all Schwab positions + cash. Returns normalized DataFrame."""
    import schwabdev

    app_key = os.getenv(SETTINGS["schwab_app_key_env"])
    app_secret = os.getenv(SETTINGS["schwab_app_secret_env"])
    if not app_key or not app_secret:
        raise BrokerError(
            "SCHWAB_APP_KEY / SCHWAB_APP_SECRET not set in .env")

    client = schwabdev.Client(app_key, app_secret)
    linked = client.linked_accounts().json()
    if not isinstance(linked, list) or not linked:
        raise BrokerError(f"Schwab linked_accounts returned: {linked}")

    fetched_at = datetime.now().isoformat(timespec="seconds")
    rows = []
    for account in linked:
        acct_hash = account.get("hashValue")
        acct_num = str(account.get("accountNumber", ""))
        details = client.account_details(acct_hash, fields="positions").json()
        sec_acct = details.get("securitiesAccount", {})

        for pos in sec_acct.get("positions", []):
            long_q = float(pos.get("longQuantity", 0.0))
            short_q = float(pos.get("shortQuantity", 0.0))
            qty = long_q - short_q                      # signed quantity
            if qty == 0:
                continue
            pnl = (float(pos.get("longOpenProfitLoss", 0.0))
                   + float(pos.get("shortOpenProfitLoss", 0.0)))
            rows.append({
                "account": acct_num,
                "symbol": str(pos.get("instrument", {}).get("symbol", "")).strip(),
                "quantity": qty,
                "avg_price": float(pos.get("averagePrice", 0.0)),
                "market_value": float(pos.get("marketValue", 0.0)),
                "open_pnl": pnl,
                "broker": "Schwab",
                "fetched_at": fetched_at,
            })

        cash = (sec_acct.get("currentBalances", {}) or {}).get("cashBalance")
        if cash is None:
            cash = (sec_acct.get("initialBalances", {}) or {}).get("cashBalance", 0.0)
        if cash:
            rows.append({
                "account": acct_num, "symbol": "CASH",
                "quantity": float(cash), "avg_price": 1.0,
                "market_value": float(cash), "open_pnl": 0.0,
                "broker": "Schwab", "fetched_at": fetched_at,
            })

    if not rows:
        raise BrokerError("Schwab returned zero positions across all accounts")

    df = pd.DataFrame(rows)
    # Schwab sometimes returns junk instrument records with no symbol.
    # Negligible ones (<$100) are dropped with a note; material ones are
    # kept and surfaced as UNKNOWN so they are never silently lost.
    blank = df["symbol"].astype(str).str.strip().isin(["", "nan", "None"])
    negligible = blank & (df["market_value"].abs() < 100)
    if negligible.any():
        print(f"  Schwab: dropped {int(negligible.sum())} symbol-less junk "
              f"rows (total value ${df.loc[negligible, 'market_value'].sum():,.2f})")
        df = df[~negligible]
    still_blank = df["symbol"].astype(str).str.strip().isin(["", "nan", "None"])
    df.loc[still_blank, "symbol"] = "UNKNOWN"

    print(f"  Schwab: {len(df)} rows across {len(linked)} accounts")
    return df


def fetch_ibkr(port: int, flex_token: Optional[str] = None,
               flex_query_id: Optional[str] = None) -> pd.DataFrame:
    """
    Pull IBKR positions: Flex Web Service (primary) or TWS subprocess (fallback).

    Flex is tried first when a token + query ID are provided. If Flex
    succeeds, TWS is never touched. If Flex env vars are not set, the
    TWS subprocess is used as before. If Flex is configured but fails
    (e.g. expired token), we DO NOT fall back to TWS — the token should work,
    and a single failing path means a stale-fallback per the design rule.
    """
    use_flex = bool(flex_token and flex_query_id)
    if not use_flex:
        return _fetch_ibkr_tws(port)

    from ibkr_flex import fetch_positions
    try:
        return fetch_positions(flex_token, flex_query_id)
    except Exception as e:
        raise BrokerError(f"IBKR Flex Web Service failed: {e}")


def _fetch_ibkr_tws(port: int) -> pd.DataFrame:
    """Pull all IBKR positions + cash via the 3.12-venv subprocess."""
    py = PATHS["ibkr_python"]
    if not py.exists():
        raise BrokerError(f"IBKR Python interpreter not found: {py}")

    result = subprocess.run(
        [str(py), str(REPORT_DIR / "ibkr_fetch.py"),
         "--port", str(port), "--client-id", str(SETTINGS["ibkr_client_id"])],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise BrokerError(
            f"IBKR fetch failed (exit {result.returncode}): "
            f"{result.stderr.strip()[-500:]}")

    data = json.loads(result.stdout)
    fetched_at = datetime.now().isoformat(timespec="seconds")
    rows = []
    for item in data:
        if item["sec_type"] == "CASH" and item["symbol"] == "CASH":
            rows.append({
                "account": item["account"], "symbol": "CASH",
                "quantity": item["quantity"], "avg_price": 1.0,
                "market_value": item["market_value"], "open_pnl": 0.0,
                "broker": "IBKR", "fetched_at": fetched_at,
            })
            continue
        symbol = str(item["symbol"]).strip()
        # Tag non-stock instruments (futures, options) so they are never
        # confused with - or priced as - an equity ticker
        if item["sec_type"] != "STK":
            symbol = f"{symbol}.{item['sec_type']}"
        rows.append({
            "account": item["account"],
            "symbol": symbol,
            "quantity": item["quantity"],
            "avg_price": item["avg_price"],
            "market_value": item["market_value"],
            "open_pnl": item["open_pnl"],
            "broker": "IBKR",
            "fetched_at": fetched_at,
        })
    if not rows:
        raise BrokerError("IBKR returned zero positions")
    print(f"  IBKR: {len(rows)} rows via TWS subprocess")
    return pd.DataFrame(rows)


def priceable_symbols(holdings_df: pd.DataFrame) -> list:
    """
    Symbols worth sending to yfinance: excludes cash, derivative-tagged
    symbols (SYM.FUT / SYM.OPT etc.) and CUSIP-like identifiers (9-char
    alphanumerics containing digits, e.g. escrow shares).
    """
    import re
    out = []
    for s in holdings_df["symbol"].astype(str).str.strip().unique():
        if not s or s in ("CASH", "UNKNOWN", "nan", "None"):
            continue
        if re.search(r"\.(FUT|OPT|FOP|BOND|WAR|CASH)$", s):
            continue
        if re.fullmatch(r"[0-9A-Z]{9}", s) and re.search(r"\d", s):
            continue
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# Fallback + orchestration
# ---------------------------------------------------------------------------

def load_stale_holdings() -> Tuple[pd.DataFrame, str]:
    """
    Load the most recent saved snapshot. Order of preference:
      1. data/holdings.xlsx (written by a previous successful run)
      2. Client.xlsx (legacy format, converted on the fly)
    Returns (DataFrame, as_of_description).
    """
    snap = PATHS["holdings"]
    if snap.exists():
        df = pd.read_excel(snap)
        as_of = str(df["fetched_at"].max()) if "fetched_at" in df.columns else "unknown"
        return df[HOLDINGS_COLUMNS], as_of

    legacy = PATHS["legacy_client_xlsx"]
    if legacy.exists():
        c = pd.read_excel(legacy)
        mtime = datetime.fromtimestamp(legacy.stat().st_mtime).isoformat(timespec="seconds")
        df = pd.DataFrame({
            "account": "",
            "symbol": c["Symbol"].astype(str).str.strip(),
            "quantity": c["Long Quantity"],
            "avg_price": c["Average Price"],
            "market_value": c["Market Value"],
            "open_pnl": c["Long Open Profit/Loss"],
            "broker": "legacy",
            "fetched_at": mtime,
        })
        return df, mtime

    raise BrokerError(
        "No live brokers AND no saved snapshot (data/holdings.xlsx or "
        "Client.xlsx) - cannot produce holdings at all")


def get_holdings(interactive: bool = True) -> Tuple[pd.DataFrame, dict]:
    """
    Main entry point. Returns (positions_df, meta).

    meta = {stale: bool, as_of: str, failures: [str, ...]}
    """
    print("\n[holdings] Pulling live holdings (Schwab + IBKR)...")
    if interactive and not sys.stdin.isatty():
        print("  (stdin is not a TTY - switching to non-interactive mode)")
        interactive = False

    port = SETTINGS["ibkr_port"]
    failures = []
    frames = []

    # --- Schwab ---
    try:
        preflight_schwab(interactive)
        frames.append(fetch_schwab())
    except (BrokerError, Exception) as e:
        failures.append(f"Schwab: {e}")
        print(f"  !! Schwab pull failed: {e}")

    # --- IBKR ---
    flex_token = os.getenv(SETTINGS["ibkr_flex_token_env"])
    flex_query_id = os.getenv(SETTINGS["ibkr_flex_query_id_env"])
    use_flex = bool(flex_token and flex_query_id)

    try:
        if use_flex:
            print(f"  IBKR: Flex Web Service configured "
                  f"(token {flex_token[:4]}..., query {flex_query_id})")
            frames.append(fetch_ibkr(
                port, flex_token=flex_token, flex_query_id=flex_query_id))
        else:
            print("  IBKR: Flex not configured, falling back to TWS...")
            preflight_ibkr(interactive, port)
            frames.append(fetch_ibkr(port))
    except (BrokerError, Exception) as e:
        failures.append(f"IBKR: {e}")
        print(f"  !! IBKR pull failed: {e}")

    if failures:
        # Approved fallback: consistent stale snapshot, loudly stamped
        print("\n  " + "!" * 60)
        print("  !! FALLING BACK TO STALE HOLDINGS - broker pull incomplete")
        for f in failures:
            print(f"  !!   {f}")
        df, as_of = load_stale_holdings()
        print(f"  !! Using snapshot as of: {as_of}")
        print("  " + "!" * 60)
        return df, {"stale": True, "as_of": as_of, "failures": failures}

    df = pd.concat(frames, ignore_index=True)[HOLDINGS_COLUMNS]
    as_of = datetime.now().isoformat(timespec="seconds")

    # Persist the fresh snapshot for future stale-fallbacks
    PATHS["holdings"].parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(PATHS["holdings"], index=False)
    print(f"  Saved fresh snapshot: {PATHS['holdings']} "
          f"({len(df)} rows, as of {as_of})")
    return df, {"stale": False, "as_of": as_of, "failures": []}


if __name__ == "__main__":
    positions, meta = get_holdings(interactive=True)
    print(f"\nstale={meta['stale']}  as_of={meta['as_of']}")
    print(positions.groupby("broker")["market_value"].agg(["count", "sum"]))
