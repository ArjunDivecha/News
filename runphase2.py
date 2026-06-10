#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: runphase2.py
=============================================================================

DESCRIPTION:
    Orchestrates the Phase 2 portfolio report pipeline. Fetches daily prices
    from yfinance, computes portfolio analytics, generates a personalized
    portfolio report via Claude Opus 4.6, and converts the report to a
    professional PDF using PrinceXML.

INPUT FILES:
    (none -- subprocess-calls child scripts; no direct file reads)

OUTPUT FILES:
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/News/Phase 2 Portfolio Reports/outputs/{portfolio_id}/portfolio_wrap_{date}.pdf
        Generated portfolio report PDF.

VERSION: 1.0
LAST UPDATED: 2026-06-05
AUTHOR: Arjun Divecha

DEPENDENCIES:
    - Python 3
    - subprocess (standard library)
    - pathlib (standard library)
    - yfinance (via child script 02_fetch_daily_prices.py)
    - PrinceXML (via child script 04_generate_report.py)

USAGE:
    python runphase2.py --portfolio TEST --date 2026-02-03
    python runphase2.py --portfolio TEST --date 2026-02-03 --skip-fetch

NOTES:
    - Child scripts live under Phase 2 Portfolio Reports/scripts/.
    - Use --skip-fetch to reuse previously fetched price data.
    - Use --skip-analytics to reuse previously computed analytics.
=============================================================================
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_command(cmd, cwd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, cwd=cwd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n✗ {description} failed with exit code {result.returncode}")
        return False
    
    print(f"\n✓ {description} completed successfully")
    return True

def main():
    parser = argparse.ArgumentParser(
        description='Run complete Phase 2 portfolio report pipeline'
    )
    parser.add_argument('--portfolio', required=True,
                        help='Portfolio ID (e.g., TEST)')
    parser.add_argument('--date', required=True,
                        help='Report date (YYYY-MM-DD)')
    parser.add_argument('--skip-fetch', action='store_true',
                        help='Skip fetching prices (use if already fetched)')
    parser.add_argument('--skip-analytics', action='store_true',
                        help='Skip computing analytics (use if already computed)')
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    phase2_dir = script_dir / "Phase 2 Portfolio Reports" / "scripts"
    
    if not phase2_dir.exists():
        print(f"✗ Phase 2 directory not found: {phase2_dir}")
        sys.exit(1)
    
    print("="*70)
    print("PHASE 2 PORTFOLIO REPORT PIPELINE")
    print("="*70)
    print(f"\nPortfolio: {args.portfolio}")
    print(f"Date: {args.date}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Fetch Daily Prices
    if not args.skip_fetch:
        cmd = [
            sys.executable, "02_fetch_daily_prices.py",
            "--portfolio", args.portfolio,
            "--date", args.date
        ]
        
        if not run_command(cmd, phase2_dir, "Step 1: Fetch Daily Prices"):
            sys.exit(1)
    else:
        print("\n[Step 1: Fetch Daily Prices - SKIPPED (--skip-fetch)]")
    
    # Step 2: Compute Analytics
    if not args.skip_analytics:
        cmd = [
            sys.executable, "03_compute_analytics.py",
            "--portfolio", args.portfolio,
            "--date", args.date
        ]
        
        if not run_command(cmd, phase2_dir, "Step 2: Compute Analytics"):
            sys.exit(1)
    else:
        print("\n[Step 2: Compute Analytics - SKIPPED (--skip-analytics)]")
    
    # Step 3: Generate Report
    cmd = [
        sys.executable, "04_generate_report.py",
        "--portfolio", args.portfolio,
        "--date", args.date
    ]
    
    if not run_command(cmd, phase2_dir, "Step 3: Generate Portfolio Report"):
        sys.exit(1)
    
    # Output location
    output_dir = script_dir / "Phase 2 Portfolio Reports" / "outputs" / args.portfolio
    pdf_file = output_dir / f"portfolio_wrap_{args.date}.pdf"
    
    print("\n" + "="*70)
    print("PHASE 2 PIPELINE COMPLETE")
    print("="*70)
    print(f"\nOutput files:")
    print(f"  PDF: {pdf_file}")
    print(f"  Directory: {output_dir}")
    
    if pdf_file.exists():
        print(f"\n✓ Report ready: {pdf_file}")
        # Open the PDF
        subprocess.run(["open", str(pdf_file)], check=False)
    else:
        print(f"\n⚠ PDF not found at expected location")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
