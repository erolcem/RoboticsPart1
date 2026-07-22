"""SiteStatePlatform: the orchestrator tying all layers together.

Workflow (proposal section 8):
  1. define project (coordinate frame, control points, bounds)
  2. subscribe sensors through adapters; verify health and manifests
  3. run a supervised capture mission; observations go to the ledger
  4. run selected processing plug-ins; claims are fused with provenance
  5. commit a versioned site-state snapshot
  6. export through output adapters
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .core.entities import (
    Mission,
    Observation,
    ProcessingActivity,
    Sensor,
    new_id,
    now_iso,
)
from .ledger.ledger import ObservationLedger
from .plugins.base import OutputAdapter, ProcessingContext, ProcessingPlugin, SensorAdapter
from .plugins.registry import PluginRegistry
from .statemodel.model import SiteStateModel


class SiteStatePlatform:
    def __init__(self, root: str | Path, project: dict[str, Any] | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ledger = ObservationLedger(self.root)
        self.registry = PluginRegistry()
        self.state = SiteStateModel(self.ledger)
        self.project = project or {}
        self._adapters: dict[str, SensorAdapter] = {}  # sensor_id -> adapter

    # -- sensor subscription ----------------------------------------------
    def subscribe(self, adapter: SensorAdapter) -> Sensor:
        """Subscribe a sensor to the platform through its adapter."""
        health = adapter.health_check()
        sensor = Sensor(id=new_id("sen"), manifest=adapter.manifest, health=health)
        self.ledger.add_sensor(sensor)
        self._adapters[sensor.id] = adapter
        if not health.get("ok", False):
            # flagged, not silently used (degraded-operation requirement)
            sensor.health.setdefault("notes", []).append("subscribed in degraded state")
        return sensor

    def unsubscribe(self, sensor_id: str) -> None:
        self._adapters.pop(sensor_id, None)

    def subscribed_data_types(self) -> list[str]:
        types: set[str] = set()
        for adapter in self._adapters.values():
            types.update(adapter.manifest.data_types)
        return sorted(types)

    # -- missions ----------------------------------------------------------
    def run_mission(
        self,
        name: str,
        carrier: dict[str, Any],
        duration: float,
        dt: float = 0.5,
        operator: str = "",
        area: str = "",
    ) -> Mission:
        """Supervised capture: poll every healthy subscribed adapter over time."""
        unhealthy = [
            sid for sid, a in self._adapters.items() if not a.health_check().get("ok", False)
        ]
        mission = Mission(
            id=new_id("mis"),
            name=name,
            carrier=carrier,
            operator=operator,
            area=area,
            started_at=now_iso(),
            configuration={
                "dt": dt,
                "duration": duration,
                "excluded_unhealthy_sensors": unhealthy,
            },
            sensor_ids=[sid for sid in self._adapters if sid not in unhealthy],
        )
        self.ledger.add_mission(mission)
        for t in np.arange(0.0, duration + 1e-9, dt):
            t = float(t)
            for sensor_id in mission.sensor_ids:
                adapter = self._adapters[sensor_id]
                for sample in adapter.sample(t):
                    obs = Observation(
                        id=new_id("obs"),
                        mission_id=mission.id,
                        sensor_id=sensor_id,
                        t=t,
                        data_type=sample.data_type,
                        frame=sample.frame,
                        quality=sample.quality,
                    )
                    self.ledger.add_observation(obs)
                    self.ledger.store_evidence(
                        kind=sample.data_type,
                        payload=sample.payload,
                        observation_id=obs.id,
                        meta={"t": t, "sensor_id": sensor_id},
                    )
        mission.ended_at = now_iso()
        self.ledger.add_mission(mission)  # upsert with end time
        return mission

    # -- processing --------------------------------------------------------
    def available_inputs(self, mission_id: str) -> list[str]:
        """Data types observed in a mission plus claim kinds already derived."""
        types = {o["data_type"] for o in self.ledger.observations(mission_id)}
        types.update(c["kind"] for c in self.ledger.claims(mission_id=mission_id, status="accepted"))
        return sorted(types)

    def process(
        self, plugin_name: str, mission_id: str, **params: Any
    ) -> ProcessingActivity:
        """Run one processing plug-in over a mission, recording the activity,
        derived evidence, claims and provenance. Re-runs supersede prior
        claims on the same subject instead of overwriting them."""
        plugin: ProcessingPlugin = self.registry.processor(plugin_name)
        missing = self.registry.missing_inputs(plugin_name, self.available_inputs(mission_id))
        activity = ProcessingActivity(
            id=new_id("act"),
            plugin=plugin.manifest.name,
            plugin_version=plugin.manifest.version,
            mission_id=mission_id,
            params={k: v for k, v in params.items()},
            started_at=now_iso(),
        )
        if missing:
            activity.status = "failed"
            activity.notes.append(
                f"sensor/claim configuration unsuitable: missing inputs {missing}"
            )
            activity.ended_at = now_iso()
            self.ledger.add_activity(activity)
            return activity

        ctx = ProcessingContext(self.ledger, mission_id, activity.id, self.project)
        try:
            plugin.run(ctx, **params)
            activity.status = "succeeded"
        except Exception as exc:  # recorded, never silent
            activity.status = "failed"
            activity.notes.append(f"{type(exc).__name__}: {exc}")
        activity.input_evidence_ids = sorted(set(ctx.read_evidence_ids))
        activity.input_claim_ids = sorted(set(ctx.read_claim_ids))
        activity.notes.extend(ctx.notes)
        activity.ended_at = now_iso()
        self.ledger.add_activity(activity)
        for claim in ctx.emitted_claims:
            if claim.subject and claim.status == "accepted":
                self.state.supersede(mission_id, claim.kind, claim.subject, activity.id)
            self.ledger.add_claim(claim)
        return activity

    # -- versions & export -------------------------------------------------
    def commit_version(self, label: str, mission_ids: list[str] | None = None):
        return self.state.commit_version(label, mission_ids)

    def register_output(self, adapter: OutputAdapter) -> None:
        self.registry.register_output(adapter)

    def export(self, output_name: str, version_id: str, out_dir: str | Path) -> list[Path]:
        version = self.ledger.version(version_id)
        if version is None:
            raise KeyError(f"unknown version {version_id}")
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        return self.registry.output(output_name).render(self.ledger, version, out)
