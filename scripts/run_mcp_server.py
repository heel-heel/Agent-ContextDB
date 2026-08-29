"""Launch the ContextDB MCP server with the project's Windows Conda environment.

This script is intended for direct use from an MCP configuration.  It writes
JSON-RPC only to stdout, so do not add normal print statements here.
"""

from pathlib import Path
import os
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(r"D:\software\Anaconda\ProgramFile\envs\contextdb_env\python.exe")


def main() -> int:
    root = os.environ.get("CONTEXTDB_DATA_ROOT", str(PROJECT_ROOT / "data"))
    command = [str(PYTHON), "-m", "contextdb.cli", "--root", root, "mcp-serve"]
    return subprocess.call(command, cwd=str(PROJECT_ROOT))


if __name__ == "__main__":
    sys.exit(main())
