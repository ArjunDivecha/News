#!/usr/bin/env python3
"""Wrapper to run backfill with forced output"""
import sys
import os
import subprocess

# Change to script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Run the script and capture output
process = subprocess.Popen(
    [sys.executable, "scripts/bloomberg_backfill.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    universal_newlines=True
)

# Print output in real-time
for line in process.stdout:
    print(line, end='', flush=True)

process.wait()
sys.exit(process.returncode)
