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

# Ensure .env is loaded (python-dotenv handles this, but be explicit)
export $(grep -v '^#' .env | grep -v '^$' | xargs) 2>/dev/null || true

# Run the full pipeline (non-interactive — launchd has no TTY)
echo ""
echo "[run_daily] Starting pipeline..."
python3 report/main.py --non-interactive
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
python3 report/notify.py "$PDF" --date "$TODAY"
NOTIFY_CODE=$?

if [ $NOTIFY_CODE -ne 0 ]; then
    echo "[run_daily] !! Email sending failed"
    exit $NOTIFY_CODE
fi

echo ""
echo "[run_daily] Done — $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "  Report: $PDF"
echo "  Log:    $LOG_FILE"
