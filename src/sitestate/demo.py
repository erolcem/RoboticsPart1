"""The built-in end-to-end demonstration (proposal section 16, extended to
the full pipeline).

Simulates the acceptance scenario: capture the same indoor area on Day 1
and Day 2 after real changes, then run every processing plug-in, commit a
version and export every output format. Also used by the test suite and
the `sitestate demo` CLI command, so it doubles as executable
documentation of the intended API usage.
"""

from __future__ import annotations

from pathlib import Path

from . import SiteStatePlatform
from .outputs import HtmlReport, JsonPackageExport
from .outputs.costmap_export import RobotCostmapExport
from .outputs.scenegraph_export import SceneGraphExport
from .processing import ALL_PLUGINS
from .sensors import (
    SimCarrier,
    SimDepthCamera,
    SimFiducialCamera,
    SimLidar2D,
    SimOdometry,
    SimWorld,
)

WAYPOINTS = [
    (1.5, 1.5), (4.8, 1.5), (4.8, 7.5), (1.5, 7.5), (1.5, 1.8),
    (5.0, 1.8), (5.2, 4.0), (7.5, 4.0), (7.5, 1.5), (12.5, 1.5),
    (12.5, 7.5), (7.5, 7.5), (7.5, 4.2),
]

_FIDUCIALS = {
    "F1": (0.4, 0.4), "F2": (13.6, 0.4), "F3": (13.6, 8.6),
    "F4": (0.4, 8.6), "F5": (6.4, 4.0),
}

# the designed state: outer walls, interior wall with doorway, the column -
# plus one planned wall at x=3 that was never built (a seeded deviation)
_DESIGN_WALLS = [
    [[0, 0], [14, 0]], [[14, 0], [14, 9]], [[14, 9], [0, 9]], [[0, 9], [0, 0]],
    [[6, 0], [6, 3.4]], [[6, 4.6], [6, 9]],
    [[10.0, 4.2], [10.5, 4.2]], [[10.5, 4.2], [10.5, 4.7]],
    [[10.5, 4.7], [10.0, 4.7]], [[10.0, 4.7], [10.0, 4.2]],
    [[3.0, 0.0], [3.0, 2.0]],
]

PROJECT = {
    "name": "Demo indoor fit-out area",
    "bounds": {"x0": -0.5, "y0": -0.5, "w": 15.0, "h": 10.0},
    "control_points": _FIDUCIALS,
    "design": {"name": "fit-out plan rev A", "walls": _DESIGN_WALLS},
    "zones": {
        "Room A (west)": [0.0, 0.0, 6.0, 9.0],
        "Room B (east)": [6.0, 0.0, 14.0, 9.0],
    },
}


def build_world(day: int) -> SimWorld:
    """A 14x9 m indoor area; day 2 has a new pallet, a moved partition
    and a removed material stack. The planned wall at x=3 is absent on
    both days (a construction omission the plan comparison must find)."""
    w = SimWorld()
    for seg in [((0, 0), (14, 0)), ((14, 0), (14, 9)), ((14, 9), (0, 9)), ((0, 9), (0, 0))]:
        w.walls.append(seg)
    w.walls.append(((6, 0), (6, 3.4)))
    w.walls.append(((6, 4.6), (6, 9)))
    w.add_box(10.0, 4.2, 0.5, 0.5)  # fixed column (planned)
    # seeded objects deliberately clear of the capture route: the carrier
    # must observe them from outside, so hollow interiors stay unobserved
    if day == 1:
        w.add_box(2.5, 6.4, 1.2, 0.9)   # material stack (removed by day 2)
        w.walls.append(((9.0, 2.0), (9.0, 3.6)))  # temporary partition
    else:
        w.add_box(12.7, 5.4, 1.0, 1.0)  # new pallet appears
        w.walls.append(((10.2, 2.0), (10.2, 3.6)))  # partition moved 1.2 m
    w.fiducials = dict(_FIDUCIALS)
    return w


def run_capture(platform: SiteStatePlatform, world: SimWorld, name: str,
                frame_offset, seed: int):
    carrier = SimCarrier(
        WAYPOINTS, speed=0.7, frame_offset=frame_offset, drift_rate=0.004, seed=seed
    )
    sensors = [
        SimLidar2D(carrier, world, seed=seed + 1),
        SimOdometry(carrier),
        SimFiducialCamera(carrier, world, seed=seed + 2),
        SimDepthCamera(carrier, world, fov_deg=110.0, rate_hz=1.0, seed=seed + 3),
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


def make_platform(project_dir: str | Path) -> SiteStatePlatform:
    """A platform with every built-in plug-in and output registered."""
    platform = SiteStatePlatform(project_dir, project=PROJECT)
    for plugin_cls in ALL_PLUGINS:
        platform.registry.register_processor(plugin_cls())
    platform.register_output(HtmlReport())
    platform.register_output(JsonPackageExport())
    platform.register_output(RobotCostmapExport())
    platform.register_output(SceneGraphExport())
    return platform


def run_full_demo(out_dir: str | Path = "demo_output", verbose: bool = True):
    def say(msg: str) -> None:
        if verbose:
            print(msg)

    root = Path(out_dir)
    platform = make_platform(root / "project_data")

    say("== Day 1 capture ==")
    m1 = run_capture(platform, build_world(1), "Day 1 baseline",
                     frame_offset=(0.35, -0.22, 0.03), seed=10)
    say("== Day 2 capture (site has changed) ==")
    m2 = run_capture(platform, build_world(2), "Day 2 repeat",
                     frame_offset=(-0.28, 0.31, -0.025), seed=20)

    say("== Processing (auto-ordered pipeline) ==")
    for act in platform.process_all(m1.id):
        say(f"  m1 {act.plugin}: {act.status} {act.notes or ''}")
    for act in platform.process_all(m2.id, baseline_mission_id=m1.id):
        say(f"  m2 {act.plugin}: {act.status} {act.notes or ''}")

    version = platform.commit_version("Day 1 vs Day 2 capture package", [m1.id, m2.id])
    files = platform.export("html-report", version.id, root)
    files += platform.export("json-package", version.id, root)
    files += platform.export("robot-costmap", version.id, root)
    files += platform.export("scene-graph", version.id, root)

    if verbose:
        _print_results(platform, files)
    return platform, m1, m2, version


def _print_results(platform: SiteStatePlatform, files) -> None:
    ledger = platform.ledger
    print("\n== Results ==")
    for reg in ledger.claims(kind="registration", status="accepted"):
        print(f"  registration {reg['mission_id']}: RMSE "
              f"{reg['payload']['rmse_m'] * 100:.1f} cm, "
              f"{reg['payload']['n_control_points']} control points, "
              f"confidence {reg['confidence']:.2f}")
    for ch in ledger.claims(kind="change", status="accepted"):
        p = ch["payload"]
        print(f"  change: {p['change_type']:<12s} at ({p['centroid'][0]:.1f}, "
              f"{p['centroid'][1]:.1f}) area {p['area_m2']:.2f} m² "
              f"confidence {ch['confidence']:.2f}")
    for dv in ledger.claims(kind="deviation", status="accepted"):
        p = dv["payload"]
        print(f"  deviation: {p['deviation_type']:<18s} at ({p['centroid'][0]:.1f}, "
              f"{p['centroid'][1]:.1f}) area {p['area_m2']:.2f} m² "
              f"confidence {dv['confidence']:.2f}")
    for en in ledger.claims(kind="entity", status="accepted"):
        p = en["payload"]
        print(f"  entity: {p['top_class']:<20s} at ({p['centroid'][0]:.1f}, "
              f"{p['centroid'][1]:.1f}) p={max(p['class_probs'].values()):.2f}")
    for tv in ledger.claims(kind="traversability", status="accepted"):
        fr = tv["payload"]["fractions"]
        print(f"  traversability: {fr['traversable']:.0%} drivable, "
              f"{fr['unknown']:.0%} unknown (robot radius "
              f"{tv['payload']['robot_radius_m']} m)")
    for cov in ledger.claims(kind="coverage", status="accepted"):
        fr = cov["payload"]["fractions"]
        print(f"  coverage {cov['mission_id']}: observed {fr['observed']:.0%}, "
              f"unobserved {fr['unobserved']:.0%}")
    print(f"\n  exported: {[str(f) for f in files]}")
