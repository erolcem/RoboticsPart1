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
