"""v2 feature tests: pose-graph optimization beats coarse registration,
evidential occupancy layer, progress tracking, confidence calibration
reporting, and deployment endpoints."""

from __future__ import annotations

import json
import sys
import threading
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sitestate.benchmark import run_benchmark, run_scenario



def test_pose_graph_payload_and_factors(full):
    platform, m1, _, _, _ = full
    pc = platform.ledger.claims(mission_id=m1.id, kind="pose_corrections",
                                status="accepted")[0]
    p = pc["payload"]
    assert p["method"] == "pose-graph"
    assert p["factors"]["odometry"] >= p["n_scans"] - 1
    assert p["factors"]["landmark"] > 20, "fiducial detections must become landmark factors"
    assert p["factors"]["scan_match"] > 0.8 * p["n_scans"]
    assert p["gauss_newton_iterations"] >= 1
    assert p["n_refined"] > 0.9 * p["n_scans"]


def test_pose_graph_beats_coarse_registration(tmp_path):
    m = run_scenario(seed=11, root=tmp_path / "pg")
    assert m["trajectory_rmse_m"] < m["coarse_trajectory_rmse_m"], \
        "the optimized trajectory must be more accurate than registration alone"
    assert m["trajectory_rmse_m"] < 0.03, "pose graph should reach ~cm accuracy"


def test_evidential_layer(full):
    platform, _, m2, _, _ = full
    geom = platform.ledger.claims(mission_id=m2.id, kind="occupancy_geometry",
                                  status="accepted")[0]
    p = geom["payload"]
    assert 0.0 <= p["mean_conflict"] < 0.1, "a static site should show little conflict"
    assert p["mean_ignorance"] > 0.05, "unobserved space must register as ignorance"
    grid = platform.ledger.evidence_payload(p["evidence_id"])
    assert "ignorance" in grid and "conflict" in grid
    # masses are valid: ignorance high exactly where nothing was seen
    unseen = (grid["hits"] + grid["passes"]) == 0
    assert float(grid["ignorance"][unseen].min()) > 0.99
    # change claims carry the region's evidential conflict
    ch = platform.ledger.claims(mission_id=m2.id, kind="change", status="accepted")[0]
    assert "evidential_conflict" in ch["payload"]


def test_progress_tracking(full):
    platform, m1, m2, _, out = full
    for mid in (m1.id, m2.id):
        prog = platform.ledger.claims(mission_id=mid, kind="progress", status="accepted")
        assert len(prog) == 1
        p = prog[0]["payload"]
        # the plan is mostly built; the never-built wall at x=3 keeps it < 1
        assert 0.85 < p["overall_completion"] < 1.0
        assert p["observed_plan_fraction"] > 0.7
        assert set(p["by_zone"]) == {"Room A (west)", "Room B (east)"}
        # Room A contains the missing wall -> less complete than Room B
        assert (p["by_zone"]["Room A (west)"]["completion"]
                < p["by_zone"]["Room B (east)"]["completion"])
    html = (Path(out) / "report.html").read_text()
    assert "Progress against the plan" in html


def test_benchmark_reports_calibration(tmp_path):
    summary = run_benchmark(seeds=1, out_dir=tmp_path, verbose=False)
    assert summary["schema"] == "sitestate/benchmark@2.0"
    cal = summary["calibration"]
    assert cal["n_claims"] > 0
    assert 0.0 <= cal["expected_calibration_error"] <= 1.0
    assert cal["reliability_bins"]
    report = json.loads((tmp_path / "benchmark_report.json").read_text())
    assert report["measures"]["trajectory_rmse_m"]["mean"] < 0.05


def test_health_and_version_endpoints(full):
    from sitestate.serve import make_server

    platform, _, _, version, _ = full
    server = make_server(platform.ledger.root, port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz") as r:
            health = json.loads(r.read())
        assert health["status"] == "ok"
        assert health["site_state_version"] == version.id
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/version") as r:
            ver = json.loads(r.read())
        assert ver["sitestate"] == __import__("sitestate").__version__
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_worked_example_runs(tmp_path):
    """examples/custom_asset_tracking.py is living documentation of the
    extension recipes - it must keep working."""
    import subprocess

    script = Path(__file__).resolve().parents[1] / "examples" / "custom_asset_tracking.py"
    out = subprocess.run([sys.executable, str(script)], capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert "CRATE-A17" in out.stdout and "GENSET-02" in out.stdout
    assert "asset-tracking v0.1.0" in out.stdout
    assert "sim-rfid-scanner" in out.stdout


def test_gnc_rejects_gross_outliers_where_huber_cannot():
    """Synthetic pose graph: perfect odometry, correct absolute factors
    except 4 gross (2 m) outliers. The GNC kernel must recover the true
    trajectory; plain Huber only dampens the outliers and stays biased."""
    import numpy as np
    from sitestate.processing.refine import PoseGraph

    rng = np.random.default_rng(3)
    n = 20
    truth = np.stack([np.arange(n) * 0.5, np.zeros(n), np.zeros(n)], axis=1)

    def build() -> PoseGraph:
        g = PoseGraph(truth + rng.normal(0, 0.05, size=(n, 3)) * [1, 1, 0.1])
        for i in range(n - 1):
            g.add_odometry(i, i + 1, np.array([0.5, 0.0, 0.0]), 400.0, 1000.0)
        for i in range(n):
            m = truth[i].copy()
            if i in (4, 9, 14, 17):  # 20% gross outliers
                m[1] += 2.0
            g.add_absolute(i, m, 625.0, 2500.0)
        return g

    g_gnc = build()
    g_gnc.optimize(kernel="gnc")
    err_gnc = float(np.abs(g_gnc.poses[:, 1] - truth[:, 1]).max())

    g_huber = build()
    g_huber.optimize(kernel="huber")
    err_huber = float(np.abs(g_huber.poses[:, 1] - truth[:, 1]).max())

    assert err_gnc < 0.02, f"GNC should reject outliers entirely (err {err_gnc:.3f} m)"
    assert err_gnc < err_huber / 3, (
        f"GNC ({err_gnc:.3f} m) must beat Huber ({err_huber:.3f} m) decisively"
    )


def test_scene_graph_export(full):
    platform, _, _, version, out = full
    graph = json.loads((Path(out) / "scene_graph.json").read_text())
    assert graph["schema"] == "sitestate/scene-graph@1.0"
    layers = graph["layers"]
    assert layers["site"]["name"]
    assert {z["name"] for z in layers["zones"]} == {"Room A (west)", "Room B (east)"}
    assert layers["entities"] and layers["changes"] and layers["deviations"]
    in_zone = [e for e in graph["edges"] if e["type"] == "in_zone"]
    assert in_zone, "entities must be linked to their zones"
    changed_since = [e for e in graph["edges"] if e["type"] == "changed_since"]
    assert changed_since, "changes must reference their baseline mission"
    # every node's claim is traceable in the ledger
    for e in layers["entities"][:3]:
        assert platform.ledger.trace(e["claim_id"])["activity"]["plugin"]


def test_capture_plan_reports_information_gain(tmp_path):
    """Degraded capture -> plan waypoints carry expected information gain
    and are ordered so early stops buy the most map."""
    from sitestate import demo as d
    from sitestate.planning import propose_capture_plan
    from sitestate.sensors import (SimCarrier, SimDepthCamera, SimFiducialCamera,
                                   SimLidar2D, SimOdometry)

    platform = d.make_platform(tmp_path / "gain")
    world = d.build_world(1)
    carrier = SimCarrier(d.WAYPOINTS, speed=0.7, frame_offset=(0.1, 0.1, 0.0),
                         drift_rate=0.004, seed=77)
    platform.subscribe(SimLidar2D(carrier, world, seed=78,
                                  fail_after=carrier.duration * 0.45))
    platform.subscribe(SimOdometry(carrier))
    platform.subscribe(SimFiducialCamera(carrier, world, seed=79))
    platform.subscribe(SimDepthCamera(carrier, world, seed=80))
    mission = platform.run_mission("gain", carrier.describe(),
                                   duration=carrier.duration, dt=0.5)
    platform.process_all(mission.id)
    platform.commit_version("gain snapshot", [mission.id])

    plan = propose_capture_plan(platform.ledger, min_target_cells=8)
    assert plan["schema"] == "sitestate/capture-plan@2.0"
    assert plan["waypoints"]
    for wp in plan["waypoints"]:
        assert wp["expected_gain_cells"] >= 0
    gains = [wp["expected_gain_cells"] for wp in plan["waypoints"]]
    assert max(gains) > 0, "waypoints must report expected information gain"
