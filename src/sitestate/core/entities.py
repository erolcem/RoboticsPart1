"""Core data entities of the Site State Platform.

These mirror section 12 of the proposal: Sensor, Carrier (described on the
Mission), Mission, Observation, Evidence asset, Processing activity, Claim
and Site-state version. Every derived Claim stays linked to the evidence
and processing activity that produced it (provenance), and carries an
explicit confidence plus uncertainty payload.
"""

from __future__ import annotations

import datetime as _dt
import math
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


@dataclass
class Pose2D:
    """A 2D pose with 1-sigma uncertainty (geometric uncertainty dimension)."""

    x: float
    y: float
    theta: float
    sigma_xy: float = 0.0
    sigma_theta: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, float]) -> "Pose2D":
        return Pose2D(**d)

    def distance_to(self, other: "Pose2D") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass
class SensorManifest:
    """What a sensor adapter must declare about itself (proposal section 9.1).

    An adapter does more than expose a data stream: it declares identity,
    data types, units, timing, expected accuracy, mounting, calibration and
    known limitations, so the platform can judge whether a configuration is
    suitable for a requested output.
    """

    name: str
    sensor_type: str  # e.g. "lidar_2d", "depth_camera", "odometry", "fiducial_camera"
    data_types: list[str]  # observation data types this sensor emits
    units: dict[str, str] = field(default_factory=dict)
    timestamp_source: str = "carrier_clock"
    expected_accuracy: dict[str, float] = field(default_factory=dict)
    mounting: dict[str, float] = field(default_factory=dict)  # transform on carrier
    calibration_version: str = "factory-v1"
    calibration_date: str = ""
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Sensor:
    id: str
    manifest: SensorManifest
    health: dict[str, Any] = field(default_factory=dict)
    subscribed_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "manifest": self.manifest.to_dict(),
            "health": self.health,
            "subscribed_at": self.subscribed_at,
        }


@dataclass
class Mission:
    id: str
    name: str
    carrier: dict[str, Any]  # type: robot/trolley/backpack/... plus description
    operator: str = ""
    area: str = ""
    started_at: str = ""
    ended_at: str = ""
    configuration: dict[str, Any] = field(default_factory=dict)
    sensor_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Observation:
    """A timestamped measurement with frame, source and quality metadata."""

    id: str
    mission_id: str
    sensor_id: str
    t: float  # mission-relative time (s)
    data_type: str
    frame: str = "carrier"
    quality: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceAsset:
    """A stored artefact: scan, image, grid, trajectory segment, ..."""

    id: str
    kind: str
    path: str  # relative to the ledger's blob root
    sha256: str
    observation_id: str = ""  # empty for derived assets
    activity_id: str = ""  # set when produced by a processing activity
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessingActivity:
    """A specific run of a plug-in with parameters and versions."""

    id: str
    plugin: str
    plugin_version: str
    mission_id: str
    params: dict[str, Any] = field(default_factory=dict)
    input_evidence_ids: list[str] = field(default_factory=list)
    input_claim_ids: list[str] = field(default_factory=list)
    started_at: str = ""
    ended_at: str = ""
    status: str = "running"  # running | succeeded | failed
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Claim:
    """A derived statement about geometry, an entity, state or change.

    Claims are interpretation, kept strictly separate from observation.
    They carry confidence, link back to their evidence and activity, and
    are never deleted: reprocessing supersedes rather than overwrites.
    """

    id: str
    kind: str  # registration | trajectory | occupancy_geometry | coverage | change | ...
    mission_id: str
    activity_id: str
    payload: dict[str, Any]
    confidence: float
    evidence_ids: list[str] = field(default_factory=list)
    subject: str = ""  # optional stable key, e.g. "grid:main" or "change:region-3"
    status: str = "accepted"  # accepted | competing | superseded | rejected
    observed_at: str = ""  # freshness: when the supporting evidence was captured
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SiteStateVersion:
    """A consistent snapshot of accepted (and competing) claims at a time."""

    id: str
    label: str
    claim_ids: list[str]
    mission_ids: list[str]
    parent_id: str = ""
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
