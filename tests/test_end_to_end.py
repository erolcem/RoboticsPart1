"""End-to-end tests against the acceptance statement of the proposal:

capture the same area twice, align both missions to the project frame,
identify meaningful changes, show where capture is uncertain, and trace
every reported result to supporting sensor evidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sitestate import demo


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("project")
    platform = demo.make_platform(root)
    m1 = demo.run_capture(platform, demo.build_world(1), "day1",
                          frame_offset=(0.35, -0.22, 0.03), seed=10)
    m2 = demo.run_capture(platform, demo.build_world(2), "day2",
                          frame_offset=(-0.28, 0.31, -0.025), seed=20)
    for mid in (m1.id, m2.id):
        for name in ("control-point-registration", "occupancy-mapping", "coverage-analysis"):
            act = platform.process(name, mid)
            assert act.status == "succeeded", act.notes
    act = platform.process("occupancy-change-detection", m2.id, baseline_mission_id=m1.id)
    assert act.status == "succeeded", act.notes
    return platform, m1, m2


def test_registration_accuracy(pipeline):
    platform, m1, m2 = pipeline
    for mid in (m1.id, m2.id):
        regs = platform.ledger.claims(mission_id=mid, kind="registration", status="accepted")
        assert len(regs) == 1
        assert regs[0]["payload"]["rmse_m"] < 0.05  # under 5 cm against control points
        assert regs[0]["confidence"] > 0.7


def test_change_detection_finds_seeded_changes(pipeline):
    platform, m1, m2 = pipeline
    changes = platform.ledger.claims(mission_id=m2.id, kind="change", status="accepted")
    assert changes, "no changes detected"
    appeared = [c for c in changes if c["payload"]["change_type"] == "appeared"]
    disappeared = [c for c in changes if c["payload"]["change_type"] == "disappeared"]

    # the new pallet at (12.9..13.9, 5.4..6.4) must be among appeared regions
    assert any(
        12.5 <= c["payload"]["centroid"][0] <= 14.2 and 5.0 <= c["payload"]["centroid"][1] <= 6.8
        for c in appeared
    ), f"pallet not found in {[c['payload']['centroid'] for c in appeared]}"
    # the removed material stack around (2.5..3.7, 6.4..7.3) must be among disappeared
    assert any(
        2.0 <= c["payload"]["centroid"][0] <= 4.2 and 6.0 <= c["payload"]["centroid"][1] <= 7.9
        for c in disappeared
    ), f"stack not found in {[c['payload']['centroid'] for c in disappeared]}"
    for c in changes:
        assert 0.0 < c["confidence"] < 1.0


def test_coverage_reports_unobserved_regions(pipeline):
    platform, m1, _ = pipeline
    covs = platform.ledger.claims(mission_id=m1.id, kind="coverage", status="accepted")
    assert len(covs) == 1
    fr = covs[0]["payload"]["fractions"]
    assert abs(sum(fr.values()) - 1.0) < 1e-6
    assert fr["observed"] > 0.3  # robot saw a good part of the area
    assert fr["unobserved"] > 0.0  # and honestly reports what it did not see


def test_provenance_traces_to_sensors(pipeline):
    platform, _, m2 = pipeline
    for claim in platform.ledger.claims(mission_id=m2.id, status="accepted"):
        trace = platform.ledger.trace(claim["id"])
        assert trace["activity"].get("plugin"), claim["kind"]
        if claim["kind"] in ("change", "occupancy_geometry", "trajectory", "registration"):
            assert trace["evidence"], f"{claim['kind']} claim has no evidence"
            assert trace["sensors"], f"{claim['kind']} claim does not reach a sensor"
            for s in trace["sensors"]:
                assert s["manifest"]["calibration_version"]


def test_reprocessing_supersedes_not_deletes(pipeline):
    platform, m1, _ = pipeline
    before = platform.ledger.claims(mission_id=m1.id, kind="occupancy_geometry")
    act = platform.process("occupancy-mapping", m1.id, res=0.1)
    assert act.status == "succeeded"
    accepted = platform.ledger.claims(mission_id=m1.id, kind="occupancy_geometry",
                                      status="accepted")
    superseded = platform.ledger.claims(mission_id=m1.id, kind="occupancy_geometry",
                                        status="superseded")
    assert len(accepted) == 1
    assert len(superseded) >= 1  # old interpretation retained, not deleted
    assert len(accepted) + len(superseded) > len(before) - 1


def test_unsuitable_configuration_is_flagged(pipeline):
    platform, m1, _ = pipeline
    act = platform.process("occupancy-change-detection", "mis_nonexistent")
    assert act.status == "failed"
    assert "missing inputs" in act.notes[0]


def test_versioning_snapshot(pipeline):
    platform, m1, m2 = pipeline
    v = platform.commit_version("test snapshot", [m1.id, m2.id])
    stored = platform.ledger.version(v.id)
    assert stored is not None
    assert set(stored["mission_ids"]) == {m1.id, m2.id}
    assert stored["claim_ids"]
    for cid in stored["claim_ids"]:
        assert platform.ledger.claim(cid) is not None
