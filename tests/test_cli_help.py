"""CLI starts & shows help — smoke-test that imports are clean."""
from __future__ import annotations

import subprocess
import sys


def test_palimpsest_help():
    r = subprocess.run([sys.executable, "-m", "humanizer.cli", "--help"],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, f"--help failed: {r.stderr}"
    assert "palimpsest" in r.stdout.lower() or "Palimpsest" in r.stdout
    assert "--env" in r.stdout
    assert "--chunked" in r.stdout
    assert "--legacy" in r.stdout


def test_palimpsest_missing_input_errors():
    r = subprocess.run([sys.executable, "-m", "humanizer.cli"],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode != 0
