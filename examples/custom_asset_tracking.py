"""Worked example: extending the platform with a NEW sensor type and a NEW
processing plug-in - without touching a single core file.

Scenario: assets on site (tool crates, equipment) carry RFID tags. We add

  1. `SimRfidScanner`  - a SensorAdapter emitting a new observation
     data type, `rfid_detection`;
  2. `AssetTracking`   - a ProcessingPlugin consuming `rfid_detection` +
     the `registration` claim, emitting `asset` claims (tag, position,
     spread, zone) with full provenance;

then runs a capture, processes it with the standard pipeline PLUS the new
plug-in (process_all picks it up automatically once registered), and
traces an asset claim back to the scanner's manifest.

Run:  python examples/custom_asset_tracking.py

This file is the extension tutorial in executable form - copy it as the
starting point for any real sensor/model integration. The corresponding
prose recipes are README §8; the claim/evidence contracts are
docs/data-reference.md.
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from sitestate import demo
from sitestate.core.entities import SensorManifest
from sitestate.design.zones import zone_of
from sitestate.plugins.base import (
    PluginManifest,
    ProcessingContext,
    ProcessingPlugin,
    Sample,
    SensorAdapter,
)
from sitestate.processing.icp import transform_pose
from sitestate.sensors import SimCarrier


# ---------------------------------------------------------------------------
# 1. A new sensor: implement the SensorAdapter contract.
#    A real RFID driver would buffer reads from its serial/BLE interface
#    and drain them in sample(); the platform neither knows nor cares.
# ---------------------------------------------------------------------------
class SimRfidScanner(SensorAdapter):
    def __init__(self, carrier: SimCarrier, tags: dict[str, tuple[float, float]],
                 read_range: float = 3.0, rate_hz: float = 1.0, seed: int = 0):
        self.carrier = carrier
        self.tags = tags
        self.read_range = read_range
        self.period = 1.0 / rate_hz
        self._last_t = -1e9
        self._rng = np.random.default_rng(seed)
        # The manifest is the sensor's honest self-description. data_types
        # is what the suitability check matches against plug-in `consumes`.
        self._manifest = SensorManifest(
            name="sim-rfid-scanner",
            sensor_type="rfid_scanner",
            data_types=["rfid_detection"],
            units={"position": "m"},
            expected_accuracy={"position_sigma_m": 0.5},  # RFID is coarse!
            calibration_version="factory-v1",
            limitations=[f"read range ~{read_range} m", "no bearing, position = carrier pose"],
        )

    @property
    def manifest(self) -> SensorManifest:
        return self._manifest

    def health_check(self) -> dict:
        return {"ok": True, "notes": []}

    def sample(self, t: float) -> list[Sample]:
        if t - self._last_t < self.period - 1e-9:
            return []
        self._last_t = t
        tx, ty, _ = self.carrier.true_pose(t)
        ex, ey, eth, sigma = self.carrier.estimated_pose(t)
        samples = []
        for tag, (ax, ay) in self.tags.items():
            if math.hypot(ax - tx, ay - ty) > self.read_range:
                continue
            # an RFID read has no bearing: best position estimate is the
            # carrier's own estimated pose at read time
            samples.append(Sample(
                data_type="rfid_detection",
                payload={"tag": tag, "pose_est": np.array([ex, ey, eth])},
                quality={"pose_sigma_xy": sigma, "read_range_m": self.read_range},
                frame="mission-estimated",
            ))
        return samples


# ---------------------------------------------------------------------------
# 2. A new model: implement the ProcessingPlugin contract.
#    consumes lists an observation data type AND a claim kind - the
#    platform won't run this until registration exists, and process_all
#    orders it automatically.
# ---------------------------------------------------------------------------
class AssetTracking(ProcessingPlugin):
    _manifest = PluginManifest(
        name="asset-tracking",
        version="0.1.0",
        consumes=["rfid_detection", "registration"],
        produces=["asset"],
        description="Estimate tagged-asset positions from RFID reads",
        validation="example-only",
    )

    @property
    def manifest(self) -> PluginManifest:
        return self._manifest

    def run(self, ctx: ProcessingContext, **params) -> None:
        reg = ctx.claims("registration")[0]  # guaranteed by `consumes`
        r = np.array(reg["payload"]["rotation"])
        t = np.array(reg["payload"]["translation"])

        reads: dict[str, list[np.ndarray]] = {}
        for obs in ctx.observations("rfid_detection"):
            for ev in ctx.evidence_for(obs["id"]):  # tracked -> provenance
                p = ctx.payload(ev["id"])
                pose = transform_pose(r, t, np.asarray(p["pose_est"], dtype=float))
                reads.setdefault(str(p["tag"]), []).append(pose[:2])

        for tag, positions in reads.items():
            arr = np.array(positions)
            center = arr.mean(axis=0)
            spread = float(arr.std(axis=0).mean()) + 0.5  # + RFID coarseness
            ctx.emit_claim(
                kind="asset",
                payload={
                    "tag": tag,
                    "position": [float(center[0]), float(center[1])],
                    "position_sigma_m": spread,
                    "n_reads": len(positions),
                    "zone": zone_of(ctx.project.get("zones"), *center),
                },
                # honest confidence: RFID localizes to metres, not cm
                confidence=float(np.clip(0.9 - spread / 5.0, 0.1, 0.9)),
                subject=f"asset:{tag}",
            )


# ---------------------------------------------------------------------------
# 3. Wire it together: subscribe, capture, process_all, inspect.
# ---------------------------------------------------------------------------
def main() -> None:
    tags = {"CRATE-A17": (3.2, 3.0), "GENSET-02": (11.8, 6.9)}
    with tempfile.TemporaryDirectory() as tmp:
        platform = demo.make_platform(Path(tmp) / "assets_project")
        platform.registry.register_processor(AssetTracking())  # <- plug in

        world = demo.build_world(1)
        carrier = SimCarrier(demo.WAYPOINTS, speed=0.7,
                             frame_offset=(0.2, -0.15, 0.02), seed=99)
        from sitestate.sensors import SimFiducialCamera, SimLidar2D, SimOdometry

        for adapter in (
            SimLidar2D(carrier, world, seed=100),
            SimOdometry(carrier),
            SimFiducialCamera(carrier, world, seed=101),
            SimRfidScanner(carrier, tags, seed=102),  # <- subscribe
        ):
            platform.subscribe(adapter)

        mission = platform.run_mission("asset sweep", carrier.describe(),
                                       duration=carrier.duration, dt=0.5)
        platform.process_all(mission.id)

        print("asset claims:")
        assets = platform.ledger.claims(mission_id=mission.id, kind="asset",
                                        status="accepted")
        for a in assets:
            p = a["payload"]
            true_pos = tags[p["tag"]]
            err = math.hypot(p["position"][0] - true_pos[0],
                             p["position"][1] - true_pos[1])
            print(f"  {p['tag']:<10s} at ({p['position'][0]:.1f}, {p['position'][1]:.1f}) "
                  f"±{p['position_sigma_m']:.1f} m  zone={p['zone']}  "
                  f"reads={p['n_reads']}  conf={a['confidence']:.2f}  "
                  f"(true error {err:.2f} m)")

        trace = platform.ledger.trace(assets[0]["id"])
        print(f"\nprovenance: plugin {trace['activity']['plugin']} "
              f"v{trace['activity']['plugin_version']} <- "
              f"{len(trace['evidence'])} evidence assets <- sensors "
              f"{sorted({s['manifest']['name'] for s in trace['sensors']})}")


if __name__ == "__main__":
    main()
