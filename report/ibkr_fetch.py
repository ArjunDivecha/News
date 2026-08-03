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


def _valid(px):
    """
    True if px is a usable price.

    IBKR does NOT signal 'no data' with None — it returns sentinels: -1.0 when
    a tick is unavailable, and 0.0/-100.0 out of marketPrice() when there is no
    live subscription and the market is closed. Anything <= 0 is therefore junk,
    never a real quote. NaN is the other no-data form.
    """
    return px is not None and px == px and px > 0


def fx_to_usd(ib, currency, _cache={}):
    """
    USD per 1 unit of `currency`, from IBKR's own FX close. None if unavailable.

    IB quotes some pairs as CCYUSD (GBP.USD) and others as USDCCY (USD.JPY),
    so try the direct pair first and invert the reciprocal one. Returning None
    (rather than falling back to 1.0) is deliberate: silently treating AUD as
    USD would misstate a position by ~30%.
    """
    currency = (currency or "USD").upper()
    if currency == "USD":
        return 1.0
    if currency in _cache:
        return _cache[currency]

    from ib_insync import Forex

    rate = None
    for pair, invert in ((f"{currency}USD", False), (f"USD{currency}", True)):
        try:
            fx = Forex(pair)
            if not ib.qualifyContracts(fx):
                continue
            t = ib.reqTickers(fx)[0]
            px = next((v for v in (t.close, t.last, t.marketPrice())
                       if _valid(v)), None)
            if px is not None:
                rate = (1.0 / px) if invert else px
                break
        except Exception:
            continue

    if rate is None:
        print(f"IBKR: no FX rate for {currency}/USD - positions in "
              f"{currency} left unpriced", file=sys.stderr)
    _cache[currency] = rate
    return rate


def snapshot_prices(ib, positions):
    """
    Closing mark per position contract, straight from IBKR, in USD.

    Returns {id(contract): (usd_price_or_None, multiplier)}. Keyed by object
    identity because one symbol can appear under several contracts.

    Price preference is close -> last -> marketPrice(): the daily run fires
    just after the US close, so the official close is the correct mark and
    marketPrice() is the least trustworthy (see _valid).

    Two corrections that are easy to miss and wrong by orders of magnitude:
      - priceMagnifier: LSE and similar venues quote in minor units (pence).
        IBKR reports the magnifier in ContractDetails; the raw tick must be
        divided by it. Invinity (IES) quotes at 21.7 pence, not GBP 21.70 —
        a 100x error on a real position if ignored.
      - currency: this book holds ASX (AUD) and LSE (GBP) lines. Local prices
        are converted at IBKR's own FX close before they can join a USD
        portfolio.

    Anything IBKR cannot resolve or price yields None, never a guess.
    """
    ib.reqMarketDataType(3)  # delayed-frozen: last close when nothing live

    out = {}
    resolved = []
    for p in positions:
        try:
            q = ib.qualifyContracts(p.contract)
        except Exception:
            q = None
        if q:
            resolved.append((p, q[0]))
        else:
            out[id(p.contract)] = (None, 1.0)
            print(f"IBKR: no contract definition for {p.contract.symbol} "
                  f"({p.contract.secType}) - left unpriced", file=sys.stderr)

    if resolved:
        tickers = ib.reqTickers(*[c for _, c in resolved])
        for (p, c), t in zip(resolved, tickers):
            mult = float(c.multiplier) if c.multiplier else 1.0
            px = next((v for v in (t.close, t.last, t.marketPrice())
                       if _valid(v)), None)

            if px is not None:
                try:
                    details = ib.reqContractDetails(c)
                    magnifier = float(details[0].priceMagnifier or 1) if details else 1.0
                except Exception:
                    magnifier = 1.0
                if magnifier and magnifier != 1.0:
                    px /= magnifier
                    print(f"IBKR: {c.symbol} quoted in minor units "
                          f"(priceMagnifier={magnifier:g}) - divided",
                          file=sys.stderr)

                rate = fx_to_usd(ib, c.currency)
                if rate is None:
                    px = None
                elif rate != 1.0:
                    px *= rate
                    print(f"IBKR: {c.symbol} converted {c.currency}->USD "
                          f"@ {rate:.4f}", file=sys.stderr)

            if px is None:
                print(f"IBKR: no usable USD price for {c.symbol} "
                      f"({c.secType}) - left unpriced", file=sys.stderr)
            out[id(p.contract)] = (px, mult)

    priced = sum(1 for v in out.values() if v[0] is not None)
    print(f"IBKR: priced {priced}/{len(positions)} positions", file=sys.stderr)
    return out


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

        live = [p for p in positions if p.position]
        prices = snapshot_prices(ib, live)

        rows = []
        for p in live:
            price, mult = prices.get(id(p.contract), (None, 1.0))
            # Downstream still prefers our own yfinance close where it has
            # one; this market_value is the fallback that finally values what
            # yfinance cannot price at all (futures, delisted/illiquid names).
            mv = None if price is None else price * float(p.position) * mult
            rows.append({
                "account": p.account,
                "symbol": p.contract.symbol,
                "sec_type": p.contract.secType,
                "currency": p.contract.currency,
                "quantity": float(p.position),
                "price": price,                # IBKR closing mark (None if unknown)
                "multiplier": mult,
                "market_value": mv,            # notional; downstream may override
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

        # accountSummary normally emits a TotalCashValue row for every
        # managed account, but preserve an explicit zero row if a broker
        # response omits an empty account.  This keeps the account-level daily
        # viewer complete without adding exposure to portfolio math.
        # ib_insync's managedAccounts() returns a list[str]; older/other
        # clients hand back a comma-joined string. Normalize both.
        represented = {str(row["account"]) for row in rows}
        if isinstance(accounts, str):
            accounts = accounts.split(",")
        for account in (accounts or []):
            account = str(account).strip()
            if account and account not in represented:
                rows.append({
                    "account": account,
                    "symbol": "CASH",
                    "sec_type": "CASH",
                    "currency": "USD",
                    "quantity": 0.0,
                    "market_value": 0.0,
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
