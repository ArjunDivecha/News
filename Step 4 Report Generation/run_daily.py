import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

result = subprocess.run(
    [sys.executable, "scripts/bloomberg_daily.py"],
    capture_output=True,
    text=True,
    cwd=os.getcwd()
)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr, file=sys.stderr)

sys.exit(result.returncode)
