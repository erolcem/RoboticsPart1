"""Versioned Site State Model (proposal section 9.4).

Holds the current best-supported claims. Claims stay linked to evidence,
competing interpretations can coexist, nothing is deleted, and versions
let users reconstruct what the system believed at any point in time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.entities import SiteStateVersion, new_id

if TYPE_CHECKING:
    from ..ledger.ledger import ObservationLedger


class SiteStateModel:
    def __init__(self, ledger: "ObservationLedger"):
        self.ledger = ledger

    def current_claims(self, mission_ids: list[str] | None = None) -> list[dict]:
        claims = [
            c
            for c in self.ledger.claims(status="accepted")
            if mission_ids is None or c["mission_id"] in mission_ids
        ]
        competing = [
            c
            for c in self.ledger.claims(status="competing")
            if mission_ids is None or c["mission_id"] in mission_ids
        ]
        return claims + competing

    def supersede(self, mission_id: str, kind: str, subject: str, new_activity_id: str) -> int:
        """Mark previous accepted claims on the same subject as superseded.

        Called when a plug-in (possibly a newer version) re-derives the same
        subject for the same mission - the reprocessing story: old claims are
        retained for history, never deleted.
        """
        n = 0
        for c in self.ledger.claims(mission_id=mission_id, kind=kind, status="accepted"):
            if c.get("subject") == subject and c["activity_id"] != new_activity_id:
                self.ledger.set_claim_status(c["id"], "superseded")
                n += 1
        return n

    def commit_version(
        self, label: str, mission_ids: list[str] | None = None
    ) -> SiteStateVersion:
        claims = self.current_claims(mission_ids)
        versions = self.ledger.versions()
        parent = versions[-1]["id"] if versions else ""
        version = SiteStateVersion(
            id=new_id("ver"),
            label=label,
            claim_ids=[c["id"] for c in claims],
            mission_ids=sorted({c["mission_id"] for c in claims}),
            parent_id=parent,
        )
        self.ledger.add_version(version)
        return version
