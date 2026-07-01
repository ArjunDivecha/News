#!/bin/bash
# =============================================================================
# run_daily.sh — wrapper for launchd daily report automation
#
# Called by launchd every weekday at 1:05 PM PT.
# Runs the full pipeline (with LLM) and emails the resulting PDF.
#
# Logs to: outputs/unified/daily_run.log
# =============================================================================

set -euo pipefail

REPO_ROOT="/Users/arjundivecha/Dropbox/AAA Backup/A Working/News"
LOG_FILE="$REPO_ROOT/outputs/unified/daily_run.log"
mkdir -p "$REPO_ROOT/outputs/unified"

exec >> "$LOG_FILE" 2>&1

echo "========================================="
echo "DAILY REPORT RUN — $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "========================================="

cd "$REPO_ROOT"

# launchd starts with a minimal PATH (/usr/bin:/bin) that cannot find
# Homebrew Python, the Claude CLI, or PrinceXML. Build a proper PATH.
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.local/bin:$HOME/bin"

# Pin to the Homebrew 3.14 interpreter that has all dependencies installed.
# (Anaconda 3.13 is missing `anthropic` and `schwabdev`.)
PYBIN="/opt/homebrew/bin/python3"
if [ ! -x "$PYBIN" ]; then
    echo "[run_daily] !! Homebrew python3 not found at $PYBIN"
    echo "[run_daily] Install via: brew install python@3.14 && pip3 install anthropic schwabdev yfinance pandas openpyxl python-dotenv markdown"
    exit 127
fi
echo "[run_daily] Using Python: $PYBIN ($($PYBIN --version 2>&1))"
echo "[run_daily] Claude CLI: $(command -v claude || echo 'NOT FOUND')"
echo "[run_daily] PrinceXML:  $(command -v prince || echo 'NOT FOUND')"

# Ensure .env is loaded (python-dotenv handles this, but be explicit)
export $(grep -v '^#' .env | grep -v '^$' | xargs) 2>/dev/null || true

# Run the full pipeline (non-interactive — launchd has no TTY)
echo ""
echo "[run_daily] Starting pipeline..."
"$PYBIN" report/main.py --non-interactive
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "[run_daily] !! Pipeline failed with exit code $EXIT_CODE"
    echo "[run_daily] Check $LOG_FILE for details"
    exit $EXIT_CODE
fi

# Find today's PDF
TODAY=$(date '+%Y-%m-%d')
PDF=$(ls -t "$REPO_ROOT/outputs/unified/Unified_Report_${TODAY}"*.pdf 2>/dev/null | head -1)

if [ -z "$PDF" ]; then
    echo ""
    echo "[run_daily] !! No PDF found for $TODAY. Pipeline may have failed silently."
    exit 1
fi

# Email the report
echo ""
echo "[run_daily] Emailing report..."
"$PYBIN" report/notify.py "$PDF" --date "$TODAY"
NOTIFY_CODE=$?

if [ $NOTIFY_CODE -ne 0 ]; then
    # A missing REPORT_EMAIL_TO or Mail.app failure should NOT fail the whole
    # run — the PDF is already on disk. Warn loudly and exit 0 so launchd
    # sees success and doesn't keep retrying.
    echo "[run_daily] !! Email not sent (notify.py exit $NOTIFY_CODE)"
    echo "[run_daily] !! PDF is still available at: $PDF"
    echo "[run_daily] !! Set REPORT_EMAIL_TO in .env to enable email delivery"
    exit 0
fi

echo ""
echo "[run_daily] Done — $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "  Report: $PDF"
echo "  Log:    $LOG_FILE"
