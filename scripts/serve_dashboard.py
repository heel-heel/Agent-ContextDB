"""Start the ContextDB dashboard with the dedicated Windows Conda environment."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONTEXTDB_PYTHON = Path(r"D:\software\Anaconda\ProgramFile\envs\contextdb_env\python.exe")
DATA_ROOT = Path(os.environ.get("CONTEXTDB_DATA_ROOT", str(PROJECT_DIR / "data")))
PORT = os.environ.get("CONTEXTDB_PORT", "8765")
LOCAL_API_ENV_PATHS = (
    PROJECT_DIR / "_API" / "dashboard.env",
    PROJECT_DIR / "_API" / "background_llm.env",
)


def _load_local_api_environment() -> dict[str, str]:
    """Read local dashboard credentials without placing them in project config."""
    values: dict[str, str] = {}
    for env_path in LOCAL_API_ENV_PATHS:
        if not env_path.is_file():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if re.fullmatch(r"[A-Z_][A-Z0-9_]*", key):
                values[key] = value.strip()
    return values


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
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        PORT,
    ]
    environment = os.environ.copy()
    environment.update(_load_local_api_environment())
    return subprocess.run(command, cwd=PROJECT_DIR, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
