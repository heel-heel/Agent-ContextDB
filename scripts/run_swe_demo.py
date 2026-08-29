"""Import the bundled SWE-agent trajectory with the Windows ContextDB environment."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONTEXTDB_PYTHON = Path(r"D:\software\Anaconda\ProgramFile\envs\contextdb_env\python.exe")
DATA_ROOT = Path(os.environ.get("CONTEXTDB_DATA_ROOT", str(PROJECT_DIR / "data")))
TRACE_PATH = PROJECT_DIR / "examples" / "swe_agent" / "marshmallow-code__marshmallow-1867.traj"
NORMALIZED_TRACE = DATA_ROOT / "swe_agent" / "marshmallow-code__marshmallow-1867.contextdb.jsonl"


def main() -> int:
    if not CONTEXTDB_PYTHON.exists():
        print(f"ContextDB environment Python was not found: {CONTEXTDB_PYTHON}", file=sys.stderr)
        return 1
    if not TRACE_PATH.exists():
        print(f"Bundled SWE-agent trace was not found: {TRACE_PATH}", file=sys.stderr)
        return 1

    command = [
        str(CONTEXTDB_PYTHON),
        "-m",
        "contextdb.cli",
        "--root",
        str(DATA_ROOT),
        "swe-demo",
        "--trace",
        str(TRACE_PATH),
        "--out",
        str(NORMALIZED_TRACE),
    ]
    if os.environ.get("CONTEXTDB_SWE_NO_LLM") == "1":
        command.append("--no-llm-annotate")
    return subprocess.run(command, cwd=PROJECT_DIR, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
