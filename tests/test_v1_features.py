"""v1.0 feature tests: pose refinement, QA plug-ins, LOS occlusion,
calibration bias detection, degraded operation, capture planning, the
benchmark harness, PNG encoding, the pipeline runner, project init and
the HTTP server (viewer + API)."""

from __future__ import annotations

import json
import struct
import sys
import threading
import urllib.request
import zlib
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sitestate import demo
from sitestate.benchmark import run_scenario
from sitestate.outputs.png import data_uri, depth_strip, encode_png
from sitestate.planning import propose_capture_plan
from sitestate.review import ReviewQueue
from sitestate.sensors import SimCarrier, SimFiducialCamera, SimLidar2D, SimWorld


@pytest.fixture(scope="module")
def full(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo_v1")
    platform, m1, m2, version = demo.run_full_demo(out, verbose=False)
    return platform, m1, m2, version, out


# -- core algorithm upgrades -------------------------------------------------

def test_pose_refinement_ran_and_corrected(full):
    platform, m1, m2, _, _ = full
    for mid in (m1.id, m2.id):
        pcs = platform.ledger.claims(mission_id=mid, kind="pose_corrections",
                                     status="accepted")
        assert len(pcs) == 1
        p = pcs[0]["payload"]
        assert p["n_refined"] > 0.8 * p["n_scans"]
        assert 0.0 < p["mean_correction_m"] < 0.3
        # mapping actually consumed the refined poses
        geom = platform.ledger.claims(mission_id=mid, kind="occupancy_geometry",
                                      status="accepted")[0]
        assert geom["payload"]["used_pose_corrections"] is True
        assert geom["payload"]["decision_entropy"] < 0.35


def test_alignment_check_within_tolerance(full):
    platform, _, m2, _, _ = full
    aligns = platform.ledger.claims(mission_id=m2.id, kind="alignment_check",
                                    status="accepted")
    assert len(aligns) == 1
    p = aligns[0]["payload"]
    assert p["within_tolerance"] is True
    assert p["residual_translation_m"] < 0.1


def test_calibration_check_passes_for_unbiased_sensors(full):
    platform, m1, _, _, _ = full
    checks = platform.ledger.claims(mission_id=m1.id, kind="calibration_check",
                                    status="accepted")
    assert checks
    assert all(c["payload"]["within_tolerance"] for c in checks)


def test_calibration_bias_is_detected(tmp_path):
    """A fiducial camera with a 12 cm systematic bias must be flagged."""
    platform = demo.make_platform(tmp_path / "biased")
    world = demo.build_world(1)
    carrier = SimCarrier(demo.WAYPOINTS, speed=0.7, frame_offset=(0.1, -0.1, 0.01),
                         drift_rate=0.004, seed=7)
    from sitestate.sensors import SimOdometry

    platform.subscribe(SimLidar2D(carrier, world, seed=8))
    platform.subscribe(SimOdometry(carrier))
    platform.subscribe(SimFiducialCamera(carrier, world, seed=9,
                                         calibration_bias=(0.12, 0.0)))
    mission = platform.run_mission("biased", carrier.describe(),
                                   duration=carrier.duration, dt=0.5)
    platform.process_all(mission.id)
    checks = platform.ledger.claims(mission_id=mission.id, kind="calibration_check",
                                    status="accepted")
    assert checks
    flagged = [c for c in checks if not c["payload"]["within_tolerance"]]
    assert flagged, "12 cm bias must exceed the 5 cm tolerance"
    assert flagged[0]["payload"]["bias_magnitude_m"] > 0.05
    # a flagged calibration is a pending review item
    assert any(c["kind"] == "calibration_check"
               for c in ReviewQueue(platform.ledger).pending())


def test_fiducial_line_of_sight_occlusion():
    """A fiducial behind a wall must not be detected."""
    world = SimWorld()
    world.walls.append(((5.0, -5.0), (5.0, 5.0)))  # wall between carrier and target
    world.fiducials = {"HIDDEN": (8.0, 0.0), "SEEN": (2.0, 1.0)}
    carrier = SimCarrier([(0.0, 0.0), (0.1, 0.0)], speed=0.1, seed=1)
    cam = SimFiducialCamera(carrier, world, detection_range=10.0, seed=2)
    ids = {str(s.payload["fiducial_id"]) for s in cam.sample(0.0)}
    assert "SEEN" in ids
    assert "HIDDEN" not in ids


# -- degraded operation + capture planning -----------------------------------

def test_degraded_capture_flags_and_plans_recapture(tmp_path):
    """Lidar dies mid-mission -> coverage collapses -> capture planning
    proposes frontier waypoints to close the gap."""
    platform = demo.make_platform(tmp_path / "degraded")
    world = demo.build_world(1)
    carrier = SimCarrier(demo.WAYPOINTS, speed=0.7, frame_offset=(0.1, 0.1, 0.0),
                         drift_rate=0.004, seed=31)
    from sitestate.sensors import SimDepthCamera, SimOdometry

    lidar = SimLidar2D(carrier, world, seed=32, fail_after=carrier.duration * 0.45)
    platform.subscribe(lidar)
    platform.subscribe(SimOdometry(carrier))
    platform.subscribe(SimFiducialCamera(carrier, world, seed=33))
    platform.subscribe(SimDepthCamera(carrier, world, seed=34))
    mission = platform.run_mission("degraded", carrier.describe(),
                                   duration=carrier.duration, dt=0.5)
    platform.process_all(mission.id)

    covs = platform.ledger.claims(mission_id=mission.id, kind="coverage",
                                  status="accepted")
    assert covs[0]["payload"]["fractions"]["observed"] < 0.7, \
        "half-mission lidar loss must show up as reduced coverage"

    platform.commit_version("degraded snapshot", [mission.id])
    plan = propose_capture_plan(platform.ledger, min_target_cells=8)
    assert plan["n_targets"] >= 1, "coverage gaps must become capture targets"
    assert plan["waypoints"], "each target needs a reachable viewpoint"
    # waypoints stand on traversable cells (they came from the trav grid)
    trav = platform.ledger.claims(mission_id=mission.id, kind="traversability",
                                  status="accepted")[0]
    grid = platform.ledger.evidence_payload(trav["payload"]["evidence_id"])
    classes, x0, y0, res = grid["classes"], float(grid["x0"]), float(grid["y0"]), float(grid["res"])
    for w in plan["waypoints"]:
        i, j = int((w["x"] - x0) / res), int((w["y"] - y0) / res)
        assert classes[j, i] == 0, "waypoint must be traversable"


def test_recapture_request_enters_plan(full):
    platform, _, m2, version, _ = full
    queue = ReviewQueue(platform.ledger)
    target = platform.ledger.claims(mission_id=m2.id, kind="change", status="accepted")[0]
    queue.request_recapture(target["id"], "erol", "verify before close-out")
    plan = propose_capture_plan(platform.ledger, version.id)
    reasons = [t["reason"] for t in plan["targets"]]
    assert "recapture_requested" in reasons
    assert plan["targets"][0]["reason"] == "recapture_requested", \
        "human requests take priority"


# -- benchmark ----------------------------------------------------------------

def test_benchmark_scenario_meets_thresholds(tmp_path):
    m = run_scenario(seed=42, root=tmp_path / "bench")
    assert m["trajectory_rmse_m"] < 0.08
    assert m["map_precision"] > 0.97
    assert m["map_recall"] > 0.95
    assert m["change_recall"] >= 0.75
    assert m["change_precision"] >= 0.75
    assert m["coverage_honesty"] > 0.9
    assert m["traceability"] == 1.0
    assert m["alignment_within_tolerance"] is True


# -- outputs ------------------------------------------------------------------

def test_png_encoder_produces_valid_png():
    img = (np.arange(0, 240, dtype=np.uint8).reshape(12, 20))
    png = encode_png(img)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    w, h = struct.unpack(">II", png[16:24])
    assert (w, h) == (20, 12)
    idat_start = png.index(b"IDAT") + 4
    idat_len = struct.unpack(">I", png[png.index(b"IDAT") - 4:png.index(b"IDAT")])[0]
    raw = zlib.decompress(png[idat_start:idat_start + idat_len])
    assert len(raw) == 12 * (20 + 1)  # scanlines + filter bytes
    assert data_uri(img).startswith("data:image/png;base64,")
    strip = depth_strip(np.linspace(0, 8, 48), max_range=8.0)
    assert strip.shape == (28, 48) and strip.dtype == np.uint8


def test_report_embeds_linked_imagery_and_qa(full):
    _, _, _, _, out = full
    html = (Path(out) / "report.html").read_text()
    assert html.count("img class='strip'") >= 6, "changes must show supporting frames"
    assert "Quality assurance" in html
    assert "within tolerance" in html
    assert "toggleLayer" in html


# -- platform / CLI / server --------------------------------------------------

def test_project_json_roundtrip(full, tmp_path):
    from sitestate import SiteStatePlatform

    p1 = SiteStatePlatform(tmp_path / "proj", project={"name": "X", "zones": {}})
    p2 = SiteStatePlatform(tmp_path / "proj")  # no project arg -> loads file
    assert p2.project["name"] == "X"


def test_cli_init_and_missions(tmp_path, capsys):
    from sitestate.cli import main

    proj = tmp_path / "fresh"
    main(["init", "--project", str(proj), "--template", "demo"])
    out = capsys.readouterr().out
    assert "initialised" in out
    assert json.loads((proj / "project.json").read_text())["control_points"]
    with pytest.raises(SystemExit):
        main(["init", "--project", str(proj)])  # refuses to overwrite


def test_http_server_endpoints(full):
    from sitestate.serve import make_server

    platform, _, _, version, _ = full
    server = make_server(platform.ledger.root, port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        def get(path):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}") as r:
                return r.status, r.read()

        status, body = get("/api/summary")
        assert status == 200 and json.loads(body)["version"] == version.id
        status, body = get("/viewer")
        assert status == 200 and b"click the map" in body
        status, body = get("/api/query?x=2.0&y=2.0")
        assert status == 200 and json.loads(body)["occupancy"] == "free"
        status, body = get("/api/plan")
        assert status == 200 and "waypoints" in json.loads(body)
        status, body = get("/report")
        assert status == 200 and b"Site State Capture Package" in body
    finally:
        server.shutdown()
        thread.join(timeout=5)
