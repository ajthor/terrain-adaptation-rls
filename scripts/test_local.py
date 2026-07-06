"""Run the lightweight local test suite."""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests"]
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
