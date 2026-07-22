"""Tests for the Phase 3/4 features: plan comparison, semantics,
traversability, robot exports, query API, review workflow, competing
claims and the dataset record/replay round trip."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sitestate import demo
from sitestate.ingest import FileReplayAdapter, export_mission_dataset
from sitestate.plugins.base import PluginManifest
from sitestate.processing.mapping import OccupancyMapping
from sitestate.query import SiteStateQuery
from sitestate.review import ReviewQueue


@pytest.fixture(scope="module")
def full(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    platform, m1, m2, version = demo.run_full_demo(out, verbose=False)
    return platform, m1, m2, version, out


def test_plan_comparison_finds_seeded_deviations(full):
    platform, _, m2, _, _ = full
    devs = platform.ledger.claims(mission_id=m2.id, kind="deviation", status="accepted")
    assert devs
    built = [d for d in devs if d["payload"]["deviation_type"] == "built_not_planned"]
    missing = [d for d in devs if d["payload"]["deviation_type"] == "planned_not_built"]
    # the temporary partition (x=10.2) and the pallet are built but unplanned
    assert any(9.7 <= d["payload"]["centroid"][0] <= 10.7 for d in built)
    assert any(12.5 <= d["payload"]["centroid"][0] <= 14.2 for d in built)
    # the planned wall at x=3 (y 0..2) was never built
    assert any(
        2.5 <= d["payload"]["centroid"][0] <= 3.5 and d["payload"]["centroid"][1] <= 2.5
        for d in missing
    ), [d["payload"]["centroid"] for d in missing]


def test_semantics_classifies_pallet_and_walls(full):
    platform, _, m2, _, _ = full
    entities = platform.ledger.claims(mission_id=m2.id, kind="entity", status="accepted")
    assert entities
    # pallet around (13.3, 5.9) should be a movable_object
    pallet = [
        e for e in entities
        if 12.5 <= e["payload"]["centroid"][0] <= 14.2
        and 5.0 <= e["payload"]["centroid"][1] <= 6.8
    ]
    assert pallet and pallet[0]["payload"]["top_class"] == "movable_object"
    # the wall ring matches the plan
    assert any(e["payload"]["top_class"] == "planned_structure" for e in entities)
    # probabilities are a distribution
    for e in entities:
        assert abs(sum(e["payload"]["class_probs"].values()) - 1.0) < 1e-6


def test_traversability_and_costmap_export(full):
    platform, _, m2, version, out = full
    travs = platform.ledger.claims(mission_id=m2.id, kind="traversability", status="accepted")
    assert len(travs) == 1
    fr = travs[0]["payload"]["fractions"]
    assert abs(sum(fr.values()) - 1.0) < 1e-6
    assert fr["traversable"] > 0.2
    assert fr["unknown"] > 0.0

    meta = json.loads((Path(out) / "costmap.json").read_text())
    assert meta["source_claim"] == travs[0]["id"]
    with np.load(Path(out) / "costmap.npz") as z:
        cost = z["cost"]
    assert set(np.unique(cost)) <= {0, 253, 254, 255}
    scene = json.loads((Path(out) / "planning_scene.json").read_text())
    assert scene["obstacles"], "planning scene should list entities"


def test_query_api(full):
    platform, _, _, version, _ = full
    q = SiteStateQuery(platform.ledger, version.id)
    # open floor near the start of the route
    open_floor = q.at_point(2.0, 2.0)
    assert open_floor["occupancy"] == "free"
    assert open_floor["traversability"] == "traversable"
    assert 0.0 < open_floor["freshness"]["freshness"] <= 1.0
    assert open_floor["sources"]["occupancy"]
    # somewhere on the pallet's edges must be occupied (its hollow interior
    # is honestly "unknown"; the exact edge cell may shift by one cell)
    probes = [q.at_point(x, y) for x in (12.65, 12.75)
              for y in (5.45, 5.9, 6.35)]
    assert any(p["occupancy"] == "occupied" for p in probes)
    kinds_here = {c["kind"] for p in probes for c in p["claims_here"]}
    assert {"change", "entity"} <= kinds_here
    summary = q.summary()
    assert summary["claims_by_kind"]["change"] >= 3


def test_review_workflow(full):
    platform, _, _, _, _ = full
    queue = ReviewQueue(platform.ledger)
    pending = queue.pending(min_confidence=0.7)
    assert pending, "low-confidence change claims should await review"
    victim = pending[0]
    queue.reject(victim["id"], "erol", "noise at column edge")
    assert platform.ledger.claim(victim["id"])["status"] == "rejected"
    assert victim["id"] not in {c["id"] for c in queue.pending(min_confidence=0.7)}
    reviews = platform.ledger.reviews(victim["id"])
    assert reviews and reviews[0]["action"] == "rejected"

    target = queue.pending(min_confidence=0.99)[0]
    queue.request_recapture(target["id"], "erol", "verify on next visit")
    assert any(r["claim_id"] == target["id"] for r in queue.recapture_requests())


class _AltMapping(OccupancyMapping):
    """Same algorithm under a different plug-in identity - stands in for a
    competing third-party reconstruction model."""

    _manifest = PluginManifest(
        name="alt-mapping-model",
        version="9.9.9",
        consumes=["scan_2d", "registration"],
        produces=["occupancy_geometry"],
        description="competing mapper",
    )

    @property
    def manifest(self):
        return self._manifest


def test_competing_claims_coexist(full):
    platform, m1, _, _, _ = full
    platform.registry.register_processor(_AltMapping())
    act = platform.process("alt-mapping-model", m1.id)
    assert act.status == "succeeded"
    accepted = platform.ledger.claims(mission_id=m1.id, kind="occupancy_geometry",
                                      status="accepted")
    competing = platform.ledger.claims(mission_id=m1.id, kind="occupancy_geometry",
                                       status="competing")
    assert len(accepted) == 1, "original model's claim stays accepted"
    assert len(competing) == 1, "different model's claim coexists as competing"
    assert ReviewQueue(platform.ledger).pending(), "competing claims await review"


def test_dataset_record_replay_roundtrip(full, tmp_path):
    platform, m1, _, _, _ = full
    ds = export_mission_dataset(platform.ledger, m1.id, tmp_path / "ds")
    data = json.loads((ds / "dataset.json").read_text())
    assert data["schema"] == "sitestate/dataset@0.1"
    assert len(data["observations"]) > 100

    replay_platform = demo.make_platform(tmp_path / "replayed")
    duration = data["observations"][-1]["t"]
    for name in data["sensors"]:
        replay_platform.subscribe(FileReplayAdapter(ds, name))
    mission = replay_platform.run_mission(
        "replayed day1", {"type": "replay"}, duration=duration, dt=0.5
    )
    n_orig = len(platform.ledger.observations(m1.id))
    n_replay = len(replay_platform.ledger.observations(mission.id))
    assert n_replay == n_orig, "replay must reproduce every recorded observation"

    for name in ("control-point-registration", "occupancy-mapping"):
        act = replay_platform.process(name, mission.id)
        assert act.status == "succeeded", act.notes
    reg = replay_platform.ledger.claims(mission_id=mission.id, kind="registration",
                                        status="accepted")[0]
    assert reg["payload"]["rmse_m"] < 0.05
