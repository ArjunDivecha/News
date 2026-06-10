#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: runphase0.py
=============================================================================

DESCRIPTION:
    Runs the Phase 0 broker holdings aggregation pipeline. Performs preflight
    checks (Schwab token, IBKR API port), refreshes broker source exports
    from Schwab and/or IBKR, normalises and aggregates holdings by symbol,
    and writes the Phase 2 feed file (Client.xlsx).

INPUT FILES:
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/News/Phase 0 Portfolio Feed/scripts/01_build_client_holdings.py
        Child script that performs the actual holdings aggregation.

OUTPUT FILES:
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/News/Client.xlsx
        Client holdings feed file consumed by Phase 2 (path configurable via --output).
    (backup directory)
        Timestamped backup created if output already exists
        (disabled with --no-backup).

VERSION: 1.0
LAST UPDATED: 2026-06-05
AUTHOR: Arjun Divecha

DEPENDENCIES:
    - Python 3
    - subprocess (standard library)
    - pathlib (standard library)

USAGE:
    python runphase0.py
    python runphase0.py --no-refresh-ibkr
    python runphase0.py --output /path/to/Client.xlsx

NOTES:
    - Requires TWS or IB Gateway running on port 7496 (configurable).
    - IBKR source script connects to the IBKR API.
    - Schwab source depends on an active Schwab token.
=============================================================================
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], cwd: Path, description: str) -> None:
    """Run command and exit on failure."""
    print("\n" + "=" * 70)
    print(description)
    print("=" * 70)
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=str(cwd), capture_output=False, text=True)
    if result.returncode != 0:
        print(f"\n[ERROR] {description} failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n[OK] {description} completed successfully")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 0 broker holdings aggregation"
    )
    parser.add_argument(
        "--output",
        default="/Users/arjundivecha/Dropbox/AAA Backup/A Working/News/Client.xlsx",
        help="Output path for Client workbook",
    )
    parser.add_argument(
        "--skip-schwab",
        action="store_true",
        help="Exclude Schwab source",
    )
    parser.add_argument(
        "--skip-ibkr",
        action="store_true",
        help="Exclude IBKR source",
    )
    parser.add_argument(
        "--no-refresh-schwab",
        action="store_true",
        help="Use existing Schwab workbook instead of re-running source script",
    )
    parser.add_argument(
        "--no-refresh-ibkr",
        action="store_true",
        help="Use existing IBKR workbook instead of re-running source script",
    )
    parser.add_argument(
        "--include-cash",
        action="store_true",
        help="Keep cash rows in final output",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Disable output backup creation",
    )
    parser.add_argument(
        "--ibkr-port",
        type=int,
        default=7496,
        help="IBKR API port to probe",
    )
    parser.add_argument(
        "--tws-app",
        default=None,
        help="Optional explicit path to Trader Workstation / IB Gateway .app",
    )
    parser.add_argument(
        "--tws-wait-seconds",
        type=int,
        default=90,
        help="Seconds to wait for IBKR API port after launching app",
    )
    parser.add_argument(
        "--source-python",
        default=None,
        help="Python executable for source scripts (Schwab + IBKR)",
    )
    parser.add_argument(
        "--schwab-python",
        default=None,
        help="Python executable override for Schwab source script",
    )
    parser.add_argument(
        "--ibkr-python",
        default=None,
        help="Python executable override for IBKR source script",
    )

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    phase0_script = script_dir / "Phase 0 Portfolio Feed" / "scripts" / "01_build_client_holdings.py"

    if not phase0_script.exists():
        print(f"[ERROR] Phase 0 script not found: {phase0_script}")
        sys.exit(1)

    cmd = [
        sys.executable,
        str(phase0_script),
        "--output",
        args.output,
        "--ibkr-port",
        str(args.ibkr_port),
        "--tws-wait-seconds",
        str(args.tws_wait_seconds),
    ]

    if args.skip_schwab:
        cmd.append("--skip-schwab")
    if args.skip_ibkr:
        cmd.append("--skip-ibkr")
    if args.no_refresh_schwab:
        cmd.append("--no-refresh-schwab")
    if args.no_refresh_ibkr:
        cmd.append("--no-refresh-ibkr")
    if args.include_cash:
        cmd.append("--include-cash")
    if args.no_backup:
        cmd.append("--no-backup")
    if args.tws_app:
        cmd.extend(["--tws-app", args.tws_app])
    if args.source_python:
        cmd.extend(["--source-python", args.source_python])
    if args.schwab_python:
        cmd.extend(["--schwab-python", args.schwab_python])
    if args.ibkr_python:
        cmd.extend(["--ibkr-python", args.ibkr_python])

    run_command(cmd, script_dir, "PHASE 0: Build Client Holdings Feed")

    print("\n" + "=" * 70)
    print("PHASE 0 COMPLETE")
    print("=" * 70)
    print(f"Client feed ready: {args.output}")


if __name__ == "__main__":
    main()
