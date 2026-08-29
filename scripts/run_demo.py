"""Run the ContextDB demo with the dedicated Windows Conda environment."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONTEXTDB_PYTHON = Path(r"D:\software\Anaconda\ProgramFile\envs\contextdb_env\python.exe")
DATA_ROOT = Path(os.environ.get("CONTEXTDB_DATA_ROOT", str(PROJECT_DIR / "data")))


def main() -> int:
    if not CONTEXTDB_PYTHON.exists():
        print(f"ContextDB environment Python was not found: {CONTEXTDB_PYTHON}", file=sys.stderr)
        return 1
    command = [
        str(CONTEXTDB_PYTHON),
        "-m",
        "contextdb.cli",
        "--root",
        str(DATA_ROOT),
        "demo",
        "--trace",
        "examples/demo_trace.jsonl",
    ]
    return subprocess.run(command, cwd=PROJECT_DIR, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
