"""Shared fixtures.

`full` runs the complete two-mission demo once per test session; the
feature test files all read from it. Tests that mutate state (review
rejections, competing-claim registration) only ever *add* ledger rows or
flip explicitly-targeted claim statuses, and are ordered within their
modules, so sharing is safe - and it keeps the suite fast enough to run
on every change.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sitestate import demo  # noqa: E402


@pytest.fixture(scope="session")
def full(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo_shared")
    platform, m1, m2, version = demo.run_full_demo(out, verbose=False)
    return platform, m1, m2, version, out
