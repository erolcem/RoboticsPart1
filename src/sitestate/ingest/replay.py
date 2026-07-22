"""The hardware bridge: a documented on-disk dataset format plus a sensor
adapter that replays it into the platform.

This is how real sensors reach the platform without the core ever
depending on ROS or vendor SDKs: any recorder (a ROS 2 bag converter, a
vendor logger, a phone app) writes this format -

    dataset_dir/
      dataset.json          index: mission meta, sensor manifests, observations
      payloads/obs_*.npz    one payload file per observation

- and `FileReplayAdapter` subscribes it like any live sensor. The same
format also round-trips: `export_mission_dataset` dumps a recorded
mission back out, so captures can be shared, archived and re-ingested.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..core.entities import SensorManifest
from ..plugins.base import Sample, SensorAdapter
from ..ledger.ledger import ObservationLedger


def export_mission_dataset(
    ledger: ObservationLedger, mission_id: str, out_dir: str | Path
) -> Path:
    """Write a mission's raw observations to the portable dataset format."""
    out = Path(out_dir)
    (out / "payloads").mkdir(parents=True, exist_ok=True)
    mission = ledger.mission(mission_id)
    if mission is None:
        raise KeyError(f"unknown mission {mission_id}")

    sensors: dict[str, dict] = {}
    index: list[dict] = []
    for obs in ledger.observations(mission_id):
        sensor = ledger.sensor(obs["sensor_id"])
        if sensor:
            sensors[sensor["manifest"]["name"]] = sensor
        for ev in ledger.evidence_for_observation(obs["id"]):
            payload = ledger.evidence_payload(ev["id"])
            fname = f"payloads/{obs['id']}.npz"
            np.savez_compressed(out / fname, **payload)
            index.append(
                {
                    "t": obs["t"],
                    "sensor": sensor["manifest"]["name"] if sensor else "unknown",
                    "data_type": obs["data_type"],
                    "frame": obs["frame"],
                    "quality": obs.get("quality", {}),
                    "payload": fname,
                }
            )

    dataset = {
        "schema": "sitestate/dataset@0.1",
        "mission": mission,
        "sensors": {
            name: {"manifest": s["manifest"], "health": s.get("health", {})}
            for name, s in sensors.items()
        },
        "observations": sorted(index, key=lambda o: o["t"]),
    }
    (out / "dataset.json").write_text(json.dumps(dataset, indent=2))
    return out


class FileReplayAdapter(SensorAdapter):
    """Replays one recorded sensor from a dataset directory.

    Subscribe one adapter per sensor name in the dataset; run the mission
    with the same (or finer) dt and at least the recorded duration, and
    every recorded observation is re-emitted at its original timestamp.
    """

    def __init__(self, dataset_dir: str | Path, sensor_name: str):
        self.dir = Path(dataset_dir)
        data = json.loads((self.dir / "dataset.json").read_text())
        if sensor_name not in data["sensors"]:
            raise KeyError(
                f"sensor {sensor_name!r} not in dataset; has: {sorted(data['sensors'])}"
            )
        m = dict(data["sensors"][sensor_name]["manifest"])
        m["limitations"] = list(m.get("limitations", [])) + ["replayed from recorded dataset"]
        self._manifest = SensorManifest(**m)
        self._obs = [o for o in data["observations"] if o["sensor"] == sensor_name]
        self._cursor = 0
        self.recorded_duration = data["observations"][-1]["t"] if data["observations"] else 0.0

    @property
    def manifest(self) -> SensorManifest:
        return self._manifest

    def health_check(self) -> dict:
        return {
            "ok": self._cursor < len(self._obs) or not self._obs,
            "notes": [f"{len(self._obs) - self._cursor} recorded observations remaining"],
        }

    def sample(self, t: float) -> list[Sample]:
        samples: list[Sample] = []
        while self._cursor < len(self._obs) and self._obs[self._cursor]["t"] <= t + 1e-9:
            rec = self._obs[self._cursor]
            self._cursor += 1
            with np.load(self.dir / rec["payload"]) as z:
                payload = {k: z[k] for k in z.files}
            samples.append(
                Sample(
                    data_type=rec["data_type"],
                    payload=payload,
                    quality=rec.get("quality", {}),
                    frame=rec.get("frame", "carrier"),
                )
            )
        return samples
