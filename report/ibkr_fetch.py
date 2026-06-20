#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: ibkr_fetch.py
=============================================================================

INPUT FILES:
    (none - connects to live IBKR API on 127.0.0.1)

OUTPUT FILES:
    (stdout - JSON array of position dicts)

VERSION: 1.0
LAST UPDATED: 2026-06-09
AUTHOR: Arjun Divecha

DESCRIPTION:
    Fetches all IBKR portfolio positions and prints them as JSON to stdout.
    This script is executed BY holdings.py as a subprocess under the
    .venv-ibkr312 interpreter, because ib_insync requires Python 3.12.
    Everything except the JSON payload goes to stderr.

    Exit codes: 0 = success, 2 = connection failed, 3 = no data.

DEPENDENCIES:
    - ib_insync, nest_asyncio (Python 3.12 venv only)

USAGE:
    .venv-ibkr312/bin/python3 report/ibkr_fetch.py [--port 7496] [--client-id 103]
=============================================================================
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=4002)
    parser.add_argument("--client-id", type=int, default=103)
    parser.add_argument("--wait", type=float, default=3.0,
                        help="Seconds to allow portfolio updates to settle")
    args = parser.parse_args()

    from ib_insync import IB

    ib = IB()
    try:
        # Plain synchronous API - reqAccountUpdates is itself blocking and
        # deadlocks when wrapped in asyncio, so we avoid async entirely.
        ib.connect("127.0.0.1", args.port,
                   clientId=args.client_id, readonly=True, timeout=20)
    except Exception as e:
        print(f"IBKR connection failed: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        accounts = ib.managedAccounts()
        print(f"IBKR accounts: {accounts}", file=sys.stderr)

        # One-shot positions request: returns positions for ALL accounts.
        # (Per-account reqAccountUpdates subscriptions hang on multi-account
        # setups, so we use reqPositions + accountSummary instead. Market
        # value and open P&L are computed downstream from our own prices.)
        positions = ib.reqPositions()
        print(f"IBKR positions: {len(positions)}", file=sys.stderr)

        rows = []
        for p in positions:
            if not p.position:
                continue
            rows.append({
                "account": p.account,
                "symbol": p.contract.symbol,
                "sec_type": p.contract.secType,
                "currency": p.contract.currency,
                "quantity": float(p.position),
                "market_value": None,          # computed downstream (qty x price)
                "avg_price": float(p.avgCost),
                "open_pnl": None,              # computed downstream
            })

        # Cash balances per account via one-shot account summary
        summary = ib.accountSummary()
        for av in summary:
            if av.tag == "TotalCashValue" and av.currency == "USD":
                rows.append({
                    "account": av.account,
                    "symbol": "CASH",
                    "sec_type": "CASH",
                    "currency": "USD",
                    "quantity": float(av.value),
                    "market_value": float(av.value),
                    "avg_price": 1.0,
                    "open_pnl": 0.0,
                })

        if not rows:
            print("IBKR returned no positions", file=sys.stderr)
            sys.exit(3)

        print(json.dumps(rows))
    finally:
        if ib.isConnected():
            ib.disconnect()


if __name__ == "__main__":
    main()
