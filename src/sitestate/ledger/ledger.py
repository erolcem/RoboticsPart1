"""Observation ledger (proposal section 9.2).

Stores evidence and metadata independently from the current world
interpretation. Large binary artefacts (scans, images, grids) live as .npz
files in a blob directory; indexed metadata, relationships and processing
records live in SQLite. Historical observations remain available for
reprocessing with better algorithms, and every claim can be traced back to
raw evidence, sensors, calibration and software versions.
"""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..core.entities import (
    Claim,
    EvidenceAsset,
    Mission,
    Observation,
    ProcessingActivity,
    Sensor,
    SensorManifest,
    SiteStateVersion,
    new_id,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sensors      (id TEXT PRIMARY KEY, json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS missions     (id TEXT PRIMARY KEY, json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS observations (
    id TEXT PRIMARY KEY, mission_id TEXT, sensor_id TEXT, t REAL,
    data_type TEXT, json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS evidence     (
    id TEXT PRIMARY KEY, observation_id TEXT, activity_id TEXT,
    kind TEXT, path TEXT, sha256 TEXT, json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS activities   (id TEXT PRIMARY KEY, mission_id TEXT, json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS claims       (
    id TEXT PRIMARY KEY, mission_id TEXT, activity_id TEXT,
    kind TEXT, subject TEXT, status TEXT, json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS versions     (id TEXT PRIMARY KEY, json TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_obs_mission ON observations (mission_id, data_type);
CREATE INDEX IF NOT EXISTS idx_ev_obs ON evidence (observation_id);
CREATE INDEX IF NOT EXISTS idx_claims_mission ON claims (mission_id, kind, status);
"""


def _jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays inside payloads to plain JSON types."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


class ObservationLedger:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.blob_dir = self.root / "evidence"
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "ledger.sqlite")
        self.db.executescript(_SCHEMA)
        self.db.commit()

    # -- generic helpers ---------------------------------------------------
    def _put(self, table: str, obj_id: str, extra: dict[str, Any], payload: dict) -> None:
        cols = ["id", *extra.keys(), "json"]
        vals = [obj_id, *extra.values(), json.dumps(_jsonable(payload))]
        placeholders = ",".join("?" for _ in cols)
        self.db.execute(
            f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({placeholders})", vals
        )
        self.db.commit()

    def _get(self, table: str, obj_id: str) -> dict | None:
        row = self.db.execute(f"SELECT json FROM {table} WHERE id=?", (obj_id,)).fetchone()
        return json.loads(row[0]) if row else None

    # -- sensors -----------------------------------------------------------
    def add_sensor(self, sensor: Sensor) -> None:
        self._put("sensors", sensor.id, {}, sensor.to_dict())

    def sensors(self) -> list[dict]:
        return [json.loads(r[0]) for r in self.db.execute("SELECT json FROM sensors")]

    def sensor(self, sensor_id: str) -> dict | None:
        return self._get("sensors", sensor_id)

    # -- missions ----------------------------------------------------------
    def add_mission(self, mission: Mission) -> None:
        self._put("missions", mission.id, {}, mission.to_dict())

    def mission(self, mission_id: str) -> dict | None:
        return self._get("missions", mission_id)

    def missions(self) -> list[dict]:
        return [json.loads(r[0]) for r in self.db.execute("SELECT json FROM missions")]

    # -- observations & evidence -------------------------------------------
    def add_observation(self, obs: Observation) -> None:
        self._put(
            "observations",
            obs.id,
            {
                "mission_id": obs.mission_id,
                "sensor_id": obs.sensor_id,
                "t": obs.t,
                "data_type": obs.data_type,
            },
            obs.to_dict(),
        )

    def observations(self, mission_id: str, data_type: str | None = None) -> list[dict]:
        q = "SELECT json FROM observations WHERE mission_id=?"
        args: list[Any] = [mission_id]
        if data_type:
            q += " AND data_type=?"
            args.append(data_type)
        q += " ORDER BY t"
        return [json.loads(r[0]) for r in self.db.execute(q, args)]

    def store_evidence(
        self,
        kind: str,
        payload: dict[str, Any],
        observation_id: str = "",
        activity_id: str = "",
        meta: dict[str, Any] | None = None,
    ) -> EvidenceAsset:
        """Persist a payload of named arrays/scalars as an .npz blob."""
        ev_id = new_id("ev")
        rel_path = f"{kind}/{ev_id}.npz"
        path = self.blob_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        np.savez_compressed(buf, **{k: np.asarray(v) for k, v in payload.items()})
        data = buf.getvalue()
        path.write_bytes(data)
        asset = EvidenceAsset(
            id=ev_id,
            kind=kind,
            path=rel_path,
            sha256=hashlib.sha256(data).hexdigest(),
            observation_id=observation_id,
            activity_id=activity_id,
            meta=meta or {},
        )
        self._put(
            "evidence",
            ev_id,
            {
                "observation_id": observation_id,
                "activity_id": activity_id,
                "kind": kind,
                "path": rel_path,
                "sha256": asset.sha256,
            },
            asset.to_dict(),
        )
        return asset

    def evidence(self, evidence_id: str) -> dict | None:
        return self._get("evidence", evidence_id)

    def evidence_payload(self, evidence_id: str) -> dict[str, np.ndarray]:
        ev = self.evidence(evidence_id)
        if ev is None:
            raise KeyError(f"unknown evidence {evidence_id}")
        with np.load(self.blob_dir / ev["path"]) as z:
            return {k: z[k] for k in z.files}

    def evidence_for_observation(self, observation_id: str) -> list[dict]:
        rows = self.db.execute(
            "SELECT json FROM evidence WHERE observation_id=?", (observation_id,)
        )
        return [json.loads(r[0]) for r in rows]

    # -- activities & claims -----------------------------------------------
    def add_activity(self, act: ProcessingActivity) -> None:
        self._put("activities", act.id, {"mission_id": act.mission_id}, act.to_dict())

    def activity(self, activity_id: str) -> dict | None:
        return self._get("activities", activity_id)

    def add_claim(self, claim: Claim) -> None:
        self._put(
            "claims",
            claim.id,
            {
                "mission_id": claim.mission_id,
                "activity_id": claim.activity_id,
                "kind": claim.kind,
                "subject": claim.subject,
                "status": claim.status,
            },
            claim.to_dict(),
        )

    def claim(self, claim_id: str) -> dict | None:
        return self._get("claims", claim_id)

    def claims(
        self,
        mission_id: str | None = None,
        kind: str | None = None,
        status: str | None = None,
    ) -> list[dict]:
        q = "SELECT json FROM claims WHERE 1=1"
        args: list[Any] = []
        for col, val in (("mission_id", mission_id), ("kind", kind), ("status", status)):
            if val is not None:
                q += f" AND {col}=?"
                args.append(val)
        return [json.loads(r[0]) for r in self.db.execute(q, args)]

    def set_claim_status(self, claim_id: str, status: str) -> None:
        c = self.claim(claim_id)
        if c is None:
            raise KeyError(f"unknown claim {claim_id}")
        c["status"] = status
        self._put(
            "claims",
            claim_id,
            {
                "mission_id": c["mission_id"],
                "activity_id": c["activity_id"],
                "kind": c["kind"],
                "subject": c.get("subject", ""),
                "status": status,
            },
            c,
        )

    # -- versions ----------------------------------------------------------
    def add_version(self, version: SiteStateVersion) -> None:
        self._put("versions", version.id, {}, version.to_dict())

    def versions(self) -> list[dict]:
        return [json.loads(r[0]) for r in self.db.execute("SELECT json FROM versions")]

    def version(self, version_id: str) -> dict | None:
        return self._get("versions", version_id)

    # -- provenance --------------------------------------------------------
    def trace(self, claim_id: str) -> dict[str, Any]:
        """Follow a claim back to its activity, evidence, observations and sensors."""
        claim = self.claim(claim_id)
        if claim is None:
            raise KeyError(f"unknown claim {claim_id}")
        activity = self.activity(claim["activity_id"]) or {}
        evidence: list[dict] = []
        sensors: dict[str, dict] = {}
        seen: set[str] = set()

        def visit_evidence(ev_ids: Iterable[str]) -> None:
            for ev_id in ev_ids:
                if ev_id in seen:
                    continue
                seen.add(ev_id)
                ev = self.evidence(ev_id)
                if not ev:
                    continue
                evidence.append(ev)
                obs_id = ev.get("observation_id")
                if obs_id:
                    obs = self._get("observations", obs_id)
                    if obs:
                        sensor = self.sensor(obs["sensor_id"])
                        if sensor:
                            sensors[sensor["id"]] = sensor
                # derived evidence: walk into the producing activity's inputs
                act_id = ev.get("activity_id")
                if act_id:
                    act = self.activity(act_id)
                    if act:
                        visit_evidence(act.get("input_evidence_ids", []))

        visit_evidence(claim.get("evidence_ids", []))
        visit_evidence(activity.get("input_evidence_ids", []))
        return {
            "claim": claim,
            "activity": activity,
            "evidence": evidence,
            "sensors": list(sensors.values()),
        }
