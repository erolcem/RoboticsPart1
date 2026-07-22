"""End-to-end demo: the acceptance scenario from the proposal (section 16).

Capture the same simulated indoor construction area on Day 1 and Day 2
after objects changed, align both missions to the project frame via
surveyed control points, detect the changes, report coverage and
uncertainty, and export a reviewable package where every result traces
back to sensor evidence.

Run:  python examples/demo_two_missions.py [output_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sitestate import SiteStatePlatform
from sitestate.outputs import HtmlReport, JsonPackageExport
from sitestate.processing import (
    ChangeDetection,
    ControlPointRegistration,
    CoverageAnalysis,
    OccupancyMapping,
)
from sitestate.sensors import (
    SimCarrier,
    SimDepthCamera,
    SimFiducialCamera,
    SimLidar2D,
    SimOdometry,
    SimWorld,
)


def build_world(day: int) -> SimWorld:
    """A 14x9 m indoor area; day 2 has a new pallet, a moved partition
    and a removed material stack."""
    w = SimWorld()
    # outer walls
    for seg in [((0, 0), (14, 0)), ((14, 0), (14, 9)), ((14, 9), (0, 9)), ((0, 9), (0, 0))]:
        w.walls.append(seg)
    # fixed interior wall with a doorway
    w.walls.append(((6, 0), (6, 3.4)))
    w.walls.append(((6, 4.6), (6, 9)))
    # fixed column
    w.add_box(10.0, 4.2, 0.5, 0.5)
    if day == 1:
        w.add_box(2.5, 6.5, 1.2, 0.9)   # material stack (removed by day 2)
        w.walls.append(((9.0, 1.0), (9.0, 2.6)))  # temporary partition
    else:
        w.add_box(11.5, 6.8, 1.2, 1.0)  # new pallet appears
        w.walls.append(((10.2, 1.0), (10.2, 2.6)))  # partition moved 1.2 m
    # surveyed fiducial targets (project frame)
    w.fiducials = {
        "F1": (0.4, 0.4), "F2": (13.6, 0.4), "F3": (13.6, 8.6),
        "F4": (0.4, 8.6), "F5": (6.4, 4.0),
    }
    return w


WAYPOINTS = [
    (1.5, 1.5), (4.8, 1.5), (4.8, 7.5), (1.5, 7.5), (1.5, 1.8),
    (5.0, 1.8), (5.2, 4.0), (7.5, 4.0), (7.5, 1.5), (12.5, 1.5),
    (12.5, 7.5), (7.5, 7.5), (7.5, 4.2),
]

PROJECT = {
    "name": "Demo indoor fit-out area",
    "bounds": {"x0": -0.5, "y0": -0.5, "w": 15.0, "h": 10.0},
    "control_points": {
        "F1": (0.4, 0.4), "F2": (13.6, 0.4), "F3": (13.6, 8.6),
        "F4": (0.4, 8.6), "F5": (6.4, 4.0),
    },
}


def run_capture(platform: SiteStatePlatform, world: SimWorld, name: str,
                frame_offset, seed: int):
    carrier = SimCarrier(
        WAYPOINTS, speed=0.7, frame_offset=frame_offset, drift_rate=0.004, seed=seed
    )
    sensors = [
        SimLidar2D(carrier, world, seed=seed + 1),
        SimOdometry(carrier),
        SimFiducialCamera(carrier, world, seed=seed + 2),
        SimDepthCamera(carrier, world, seed=seed + 3),
    ]
    ids = [platform.subscribe(s).id for s in sensors]
    mission = platform.run_mission(
        name=name,
        carrier=carrier.describe(),
        duration=carrier.duration,
        dt=0.5,
        operator="demo-operator",
        area=PROJECT["name"],
    )
    for sid in ids:
        platform.unsubscribe(sid)
    return mission


def main(out_dir: str = "demo_output") -> None:
    root = Path(out_dir)
    platform = SiteStatePlatform(root / "project_data", project=PROJECT)
    for plugin in (ControlPointRegistration(), OccupancyMapping(),
                   CoverageAnalysis(), ChangeDetection()):
        platform.registry.register_processor(plugin)
    platform.register_output(HtmlReport())
    platform.register_output(JsonPackageExport())

    print("== Day 1 capture ==")
    m1 = run_capture(platform, build_world(1), "Day 1 baseline",
                     frame_offset=(0.35, -0.22, 0.03), seed=10)
    print("== Day 2 capture (site has changed) ==")
    m2 = run_capture(platform, build_world(2), "Day 2 repeat",
                     frame_offset=(-0.28, 0.31, -0.025), seed=20)

    print("== Processing ==")
    for mid in (m1.id, m2.id):
        for name in ("control-point-registration", "occupancy-mapping", "coverage-analysis"):
            act = platform.process(name, mid)
            print(f"  {name} on {mid}: {act.status} {act.notes or ''}")
    act = platform.process("occupancy-change-detection", m2.id, baseline_mission_id=m1.id)
    print(f"  occupancy-change-detection: {act.status} {act.notes or ''}")

    version = platform.commit_version("Day 1 vs Day 2 capture package", [m1.id, m2.id])
    files = platform.export("html-report", version.id, root)
    files += platform.export("json-package", version.id, root)

    print("\n== Results ==")
    for reg in platform.ledger.claims(kind="registration", status="accepted"):
        print(f"  registration {reg['mission_id']}: RMSE "
              f"{reg['payload']['rmse_m'] * 100:.1f} cm, "
              f"{reg['payload']['n_control_points']} control points, "
              f"confidence {reg['confidence']:.2f}")
    for ch in platform.ledger.claims(kind="change", status="accepted"):
        p = ch["payload"]
        print(f"  change: {p['change_type']:<11s} at ({p['centroid'][0]:.1f}, "
              f"{p['centroid'][1]:.1f}) area {p['area_m2']:.2f} m² "
              f"confidence {ch['confidence']:.2f}")
    for cov in platform.ledger.claims(kind="coverage", status="accepted"):
        fr = cov["payload"]["fractions"]
        print(f"  coverage {cov['mission_id']}: observed {fr['observed']:.0%}, "
              f"insufficient {fr['insufficient']:.0%}, unobserved {fr['unobserved']:.0%}")

    ch = platform.ledger.claims(kind="change", status="accepted")
    if ch:
        trace = platform.ledger.trace(ch[0]["id"])
        print(f"\n  provenance of first change claim: plugin "
              f"{trace['activity']['plugin']} v{trace['activity']['plugin_version']}, "
              f"{len(trace['evidence'])} evidence assets, "
              f"sensors: {[s['manifest']['name'] for s in trace['sensors']]}")

    print(f"\n  exported: {[str(f) for f in files]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "demo_output")
