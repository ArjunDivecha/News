# Phase 0 Portfolio Feed

Builds `Client.xlsx` from live Schwab + IBKR holdings exports.

## Purpose

Phase 0 is the broker-ingestion layer for Phase 2 portfolio reports.
It refreshes source holdings, normalizes both schemas, aggregates by symbol, and writes the Phase 2 input workbook.

## Default Behavior

`01_build_client_holdings.py` does this by default:

1. Checks Schwab token DB status (`~/.schwabdev/tokens.db`)
2. Checks IBKR API port (`127.0.0.1:7496`)
3. Launches Trader Workstation / IB Gateway if IBKR API is not ready
4. Runs:
   - `/Users/arjundivecha/Dropbox/AAA Backup/Master Valuation/Schwab All Accounts Data.py`
   - `/Users/arjundivecha/Dropbox/AAA Backup/Master Valuation/IBKR.py`
5. Writes:
   - `/Users/arjundivecha/Dropbox/AAA Backup/A Working/News/Client.xlsx`

Cash rows are excluded unless `--include-cash` is passed.

## Usage

From project root:

```bash
python runphase0.py
```

Useful flags:

```bash
# Use existing exports without rerunning source scripts
python runphase0.py --no-refresh-schwab --no-refresh-ibkr

# If your IBKR API uses paper port
python runphase0.py --ibkr-port 7497

# Use a specific Python env for broker source scripts
python runphase0.py --source-python /opt/homebrew/bin/python3.11

# Separate interpreter overrides
python runphase0.py --schwab-python /path/to/schwab_env/bin/python --ibkr-python /path/to/ibkr_env/bin/python

# Explicit app path for TWS
python runphase0.py --tws-app "/Users/arjundivecha/Applications/Trader Workstation/Trader Workstation.app"

# Include cash rows in output feed
python runphase0.py --include-cash
```

## Output Schema

`Client.xlsx` first sheet columns:

1. `Symbol`
2. `Market Value`
3. `Average Price`
4. `Long Quantity`
5. `Long Open Profit/Loss`

This matches the ingestion requirements in `Phase 2 Portfolio Reports/scripts/01_ingest_portfolio.py`.
