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

# Single-instance lock. On 2026-08-14 an external retry ("rerun_once") fired a
# second pipeline TEN SECONDS after the first, because it read a stderr line as
# a failure while the first run was still working. Two concurrent runs write the
# same report.db and the same output paths, and — the expensive part — each
# burns a full Fable pass against a subscription quota whose exhaustion is the
# single most common cause of a genuinely failed run. A retry that doubles token
# spend is worse than no retry, so refuse to start rather than pile on.
LOCK_DIR="$REPO_ROOT/outputs/unified/.daily_run.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    LOCK_PID=$(cat "$LOCK_DIR/pid" 2>/dev/null || echo "")
    if [ -n "$LOCK_PID" ] && kill -0 "$LOCK_PID" 2>/dev/null; then
        echo "[run_daily] ALREADY RUNNING (pid $LOCK_PID) — refusing to start a second pipeline."
        echo "[run_daily] If this is a retry, wait for the in-flight run to finish."
        exit 0
    fi
    # mkdir won but no live owner: a previous run was killed before cleanup.
    echo "[run_daily] Clearing stale lock (owner pid '${LOCK_PID:-unknown}' is gone)."
    rm -rf "$LOCK_DIR"
    mkdir "$LOCK_DIR" || { echo "[run_daily] !! Could not acquire lock"; exit 1; }
fi
echo $$ > "$LOCK_DIR/pid"
# Release on every exit path, including set -e aborts and SIGTERM from launchd.
trap 'rm -rf "$LOCK_DIR"' EXIT INT TERM

# launchd starts with a minimal PATH (/usr/bin:/bin) that cannot find
# Homebrew Python, the Claude CLI, or PrinceXML. Build a proper PATH.
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.local/bin:$HOME/bin"

# The Claude CLI is installed via nvm-managed node, which launchd knows
# nothing about (root cause of the "Claude CLI: NOT FOUND" runs of early
# July 2026 — the report writer silently fell back to metered API billing,
# and Fable's Desk was skipped). Resolve the newest node bin dynamically so
# a node upgrade doesn't re-break this.
NVM_NODE_BIN=$(ls -d "$HOME/.nvm/versions/node/"*/bin 2>/dev/null | sort -V | tail -1)
if [ -n "$NVM_NODE_BIN" ]; then
    export PATH="$NVM_NODE_BIN:$PATH"
fi

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
