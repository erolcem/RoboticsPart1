"""Thin wrapper: the demo now lives in the package (sitestate.demo) so the
CLI and tests share it.  Run:  python examples/demo_two_missions.py [out_dir]
Equivalent to:  sitestate demo --out [out_dir]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sitestate.demo import run_full_demo

if __name__ == "__main__":
    run_full_demo(sys.argv[1] if len(sys.argv) > 1 else "demo_output")
