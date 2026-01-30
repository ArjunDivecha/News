# Bloomberg Integration Setup Guide

Complete setup for fetching real Bloomberg data and generating reports.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     WINDOWS (Parallels)                              │
│  Bloomberg Terminal + bloomberg_fetcher.py                          │
│                                                                      │
│  Writes CSV files to shared Dropbox folder:                         │
│  - bloomberg_data_YYYY-MM-DD.csv                                    │
│  - factor_returns_YYYY-MM-DD.csv                                    │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ Dropbox Sync
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          MAC                                         │
│  04_load_bloomberg.py → SQLite → 03_generate_daily_report.py        │
│                                                                      │
│  Reads CSVs, loads to database, generates LLM report + PDF          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Windows/Parallels Setup

### 1.1 Install Python Dependencies

Open Command Prompt or PowerShell in Parallels:

```cmd
pip install blpapi pandas numpy
```

### 1.2 Verify Bloomberg Terminal

1. Open Bloomberg Terminal
2. Log in with your credentials
3. Verify Bloomberg is running (green icon in system tray)

### 1.3 Navigate to Script Location

The scripts are in your shared Dropbox folder:

```cmd
cd "C:\Users\<username>\Dropbox\AAA Backup\A Working\News\Step 4 Report Generation\scripts"
```

Or wherever your Dropbox syncs to on Windows.

---

## Step 2: Run Bloomberg Fetcher (Windows)

### Test Mode (No Bloomberg Required)

First, test the script works without Bloomberg:

```cmd
python bloomberg_fetcher.py --test
```

Expected output:
```
======================================================================
BLOOMBERG DATA FETCHER
======================================================================

Date: 2026-01-30
Test mode: True

[1/4] Loading ticker list...
      Loaded 970 tickers

[2/4] Test mode - skipping Bloomberg connection

[3/4] Fetching asset data...
      Generating mock data (test mode)...
      Fetched data for 970 tickers

[4/4] Fetching factor returns...
      Fetched 15 factor returns

Files written to: .../data
  - bloomberg_data_2026-01-30.csv (970 assets)
  - factor_returns_2026-01-30.csv (15 factors)
```

### Live Mode (Bloomberg Required)

With Bloomberg Terminal running:

```cmd
python bloomberg_fetcher.py
```

Or for a specific date:

```cmd
python bloomberg_fetcher.py --date 2026-01-30
```

---

## Step 3: Load Data on Mac

After the CSV files sync via Dropbox, run on Mac:

```bash
cd "/Users/arjundivecha/Dropbox/AAA Backup/A Working/News/Step 4 Report Generation"

# Load Bloomberg data into SQLite
python3 scripts/04_load_bloomberg.py --date 2026-01-30
```

Expected output:
```
======================================================================
LOADING BLOOMBERG DATA FOR 2026-01-30
======================================================================

[1/5] Loading CSV files...
      Loaded 970 rows from bloomberg_data_2026-01-30.csv
      Loaded 15 factor returns from factor_returns_2026-01-30.csv

[2/5] Loading asset data...
      Loaded 970 assets with classifications

[3/5] Computing derived metrics...
      Z-scores computed: 850
      Alpha computed: 920

[4/5] Saving to database...
      Saved 970 price records
      Saved 15 factor returns

[5/5] Computing category statistics...
      Saved 58 category stats

✓ Bloomberg data loaded successfully
```

---

## Step 4: Generate Report

With data loaded, generate the daily report:

```bash
# Generate report with Anthropic (Claude)
python3 scripts/03_generate_daily_report.py --provider anthropic
```

This will:
1. Load the prompt template
2. Pull data from SQLite
3. Inject data into prompt
4. Call Claude Sonnet to generate the report
5. Save markdown to `outputs/daily/`

To also generate PDF:

```bash
python3 -c "
from scripts.utils.pdf import convert_report
convert_report('outputs/daily/daily_wrap_2026-01-30_anthropic.md')
"
```

---

## Complete Daily Workflow

### On Windows (After Market Close ~4:30 PM ET)

```cmd
cd "C:\Users\<username>\Dropbox\AAA Backup\A Working\News\Step 4 Report Generation\scripts"
python bloomberg_fetcher.py
```

Wait for Dropbox to sync (usually 10-30 seconds).

### On Mac

```bash
cd "/Users/arjundivecha/Dropbox/AAA Backup/A Working/News/Step 4 Report Generation"

# Load Bloomberg data
python3 scripts/04_load_bloomberg.py

# Generate report
python3 scripts/03_generate_daily_report.py --provider anthropic

# Convert to PDF
python3 -c "from scripts.utils.pdf import convert_report; convert_report('outputs/daily/daily_wrap_$(date +%Y-%m-%d)_anthropic.md')"

# Open the PDF
open outputs/daily/daily_wrap_$(date +%Y-%m-%d)_anthropic.pdf
```

---

## One-Liner for Mac (After Bloomberg Fetch)

```bash
cd "/Users/arjundivecha/Dropbox/AAA Backup/A Working/News/Step 4 Report Generation" && \
python3 scripts/04_load_bloomberg.py && \
python3 scripts/03_generate_daily_report.py --provider anthropic && \
python3 -c "from scripts.utils.pdf import convert_report; convert_report('outputs/daily/daily_wrap_$(date +%Y-%m-%d)_anthropic.md')" && \
open outputs/daily/daily_wrap_$(date +%Y-%m-%d)_anthropic.pdf
```

---

## Troubleshooting

### Bloomberg Connection Issues

**Error:** "Failed to start Bloomberg session"
- Ensure Bloomberg Terminal is open and logged in
- Check that blpapi is installed: `pip show blpapi`
- Try restarting Bloomberg Terminal

**Error:** "blpapi not installed"
- Run: `pip install blpapi`
- May need Visual C++ Build Tools on Windows

### CSV Not Syncing

- Check Dropbox is running on both Windows and Mac
- Verify both machines are online
- Look in `Step 4 Report Generation/data/` for CSV files

### Database Errors

- Re-initialize database: `python3 database/init_db.py --reset`
- Re-sync static data: `python3 scripts/01_sync_static_data.py`

### API Key Issues

- Ensure `.env` file exists in project root with `ANTHROPIC_API_KEY`
- Check key is valid and has credits

---

## Files Reference

| File | Location | Purpose |
|------|----------|---------|
| `bloomberg_fetcher.py` | scripts/ | Run on Windows, fetches Bloomberg data |
| `04_load_bloomberg.py` | scripts/ | Run on Mac, loads CSV to SQLite |
| `03_generate_daily_report.py` | scripts/ | Generate LLM report |
| `bloomberg_data_*.csv` | data/ | Daily price data from Bloomberg |
| `factor_returns_*.csv` | data/ | Factor returns from Bloomberg |
| `market_data.db` | database/ | SQLite database |

---

## Quick Test (No Bloomberg Needed)

To test the entire pipeline with mock data:

```bash
cd "/Users/arjundivecha/Dropbox/AAA Backup/A Working/News/Step 4 Report Generation"

# Generate mock Bloomberg data
python3 scripts/02_generate_mock_data.py --scenario risk_off

# Generate report with mock data
python3 scripts/03_generate_daily_report.py --provider anthropic

# View report
cat outputs/daily/daily_wrap_$(date +%Y-%m-%d)_anthropic.md
```

---

**Ready to run!**
