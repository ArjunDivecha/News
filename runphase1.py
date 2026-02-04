#!/usr/bin/env python3
"""
=============================================================================
RUN PHASE 1 - Complete Market Report Pipeline
=============================================================================

USAGE:
    ./runphase1 --date 2026-02-03
    ./runphase1 --date 2026-02-03 --skip-load-data

WHAT IT DOES:
    1. Loads Bloomberg data from Excel into SQLite (unless --skip-load-data)
    2. Computes category statistics and factor returns
    3. Generates structured report via Claude
    4. Converts to professional PDF with PrinceXML

OUTPUT:
    - PDF report in Step 4 Report Generation/outputs/comparison/{date}/
    - Markdown and JSON versions
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
        description='Run complete Phase 1 market report pipeline'
    )
    parser.add_argument('--date', required=True,
                        help='Report date (YYYY-MM-DD)')
    parser.add_argument('--skip-load-data', action='store_true',
                        help='Skip loading Excel data (use if already loaded)')
    parser.add_argument('--models', default='anthropic',
                        help='Models to use (default: anthropic)')
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    phase1_dir = script_dir / "Step 4 Report Generation" / "scripts"
    
    if not phase1_dir.exists():
        print(f"✗ Phase 1 directory not found: {phase1_dir}")
        sys.exit(1)
    
    print("="*70)
    print("PHASE 1 MARKET REPORT PIPELINE")
    print("="*70)
    print(f"\nDate: {args.date}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Generate Report (includes data loading if not skipped)
    cmd = [
        sys.executable, "generate_report.py",
        "--date", args.date,
        "--models", args.models,
        "--structured"
    ]
    
    if args.skip_load_data:
        cmd.append("--skip-load-data")
    
    if not run_command(cmd, phase1_dir, "Step 1: Generate Market Report"):
        sys.exit(1)
    
    # Output location
    output_dir = script_dir / "Step 4 Report Generation" / "outputs" / "comparison" / args.date
    pdf_file = output_dir / f"{args.models}_report.pdf"
    
    print("\n" + "="*70)
    print("PHASE 1 PIPELINE COMPLETE")
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
