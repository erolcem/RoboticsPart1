"""Plug-in boundaries (proposal section 10).

Three replaceable boundaries, each with a stable contract:

* SensorAdapter  - a new sensor connects through a documented adapter
                   without changing the world-model core.
* ProcessingPlugin - a new perception/reconstruction model deploys through
                   a manifest-driven interface and can be compared with,
                   and supersede, prior versions.
* OutputAdapter  - new reports/exports consume the site state without
                   modifying acquisition code.

Previously captured evidence can always be reinterpreted with improved
plug-ins without returning to the site (reprocessing).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from ..core.entities import Claim, SensorManifest, new_id, now_iso

if TYPE_CHECKING:
    from ..ledger.ledger import ObservationLedger


@dataclass
class Sample:
    """One measurement produced by a sensor adapter at time t."""

    data_type: str
    payload: dict[str, Any]  # named arrays / scalars, stored as evidence
    quality: dict[str, Any] = field(default_factory=dict)
    frame: str = "carrier"


class SensorAdapter(ABC):
    """Contract every sensor must fulfil to subscribe to the platform."""

    @property
    @abstractmethod
    def manifest(self) -> SensorManifest: ...

    @abstractmethod
    def health_check(self) -> dict[str, Any]:
        """Return {'ok': bool, 'notes': [...]}. Called at subscription and
        before each mission; a failing sensor is flagged, not silently used."""

    @abstractmethod
    def sample(self, t: float) -> list[Sample]:
        """Produce zero or more samples for mission-relative time t."""


@dataclass
class PluginManifest:
    """Manifest for a processing plug-in (proposal section 9.3)."""

    name: str
    version: str
    consumes: list[str]  # observation data types and/or claim kinds required
    produces: list[str]  # claim kinds produced
    mode: str = "offline"  # offline | online
    description: str = ""
    compute: str = "cpu-light"
    validation: str = "unvalidated"
    cross_mission: bool = False  # needs a baseline_mission_id to compare against

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "consumes": self.consumes,
            "produces": self.produces,
            "mode": self.mode,
            "description": self.description,
            "compute": self.compute,
            "validation": self.validation,
            "cross_mission": self.cross_mission,
        }


class ProcessingContext:
    """Handed to a plug-in run; records everything it reads and emits so the
    platform can build the provenance graph without trusting the plug-in."""

    def __init__(
        self,
        ledger: "ObservationLedger",
        mission_id: str,
        activity_id: str,
        project: dict[str, Any],
    ):
        self.ledger = ledger
        self.mission_id = mission_id
        self.activity_id = activity_id
        self.project = project
        self.emitted_claims: list[Claim] = []
        self.read_evidence_ids: list[str] = []
        self.read_claim_ids: list[str] = []
        self.notes: list[str] = []

    # -- reading (tracked for provenance) ---------------------------------
    def observations(self, data_type: str, mission_id: str | None = None) -> list[dict]:
        return self.ledger.observations(mission_id or self.mission_id, data_type)

    def evidence_for(self, observation_id: str) -> list[dict]:
        assets = self.ledger.evidence_for_observation(observation_id)
        self.read_evidence_ids.extend(a["id"] for a in assets)
        return assets

    def payload(self, evidence_id: str) -> dict[str, np.ndarray]:
        if evidence_id not in self.read_evidence_ids:
            self.read_evidence_ids.append(evidence_id)
        return self.ledger.evidence_payload(evidence_id)

    def claims(self, kind: str, mission_id: str | None = None) -> list[dict]:
        found = self.ledger.claims(
            mission_id=mission_id or self.mission_id, kind=kind, status="accepted"
        )
        self.read_claim_ids.extend(c["id"] for c in found)
        return found

    # -- writing -----------------------------------------------------------
    def store_derived(self, kind: str, payload: dict[str, Any], meta: dict | None = None) -> str:
        asset = self.ledger.store_evidence(
            kind, payload, activity_id=self.activity_id, meta=meta
        )
        return asset.id

    def emit_claim(
        self,
        kind: str,
        payload: dict[str, Any],
        confidence: float,
        evidence_ids: list[str] | None = None,
        subject: str = "",
        status: str = "accepted",
        observed_at: str = "",
    ) -> Claim:
        if not observed_at:
            # freshness reflects when the evidence was captured, not when it
            # was processed - reprocessing old data must not look "fresh"
            mission = self.ledger.mission(self.mission_id) or {}
            observed_at = mission.get("ended_at") or now_iso()
        claim = Claim(
            id=new_id("clm"),
            kind=kind,
            mission_id=self.mission_id,
            activity_id=self.activity_id,
            payload=payload,
            confidence=float(confidence),
            evidence_ids=evidence_ids or [],
            subject=subject,
            status=status,
            observed_at=observed_at,
        )
        self.emitted_claims.append(claim)
        return claim

    def note(self, message: str) -> None:
        self.notes.append(message)


class ProcessingPlugin(ABC):
    @property
    @abstractmethod
    def manifest(self) -> PluginManifest: ...

    @abstractmethod
    def run(self, ctx: ProcessingContext, **params: Any) -> None:
        """Read via ctx, emit claims/derived evidence via ctx."""


class OutputAdapter(ABC):
    """Translates the internal model into customer/machine-facing formats."""

    name: str = "output"

    @abstractmethod
    def render(
        self,
        ledger: "ObservationLedger",
        version: dict[str, Any],
        out_dir: Path,
    ) -> list[Path]:
        """Write output files for a site-state version; return written paths."""
