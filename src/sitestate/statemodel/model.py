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

    def integrate(self, claim, plugin_name: str) -> str:
        """Fuse a freshly emitted claim into the model and return its status.

        Rules (proposal section 9.4 - claims can coexist when models disagree):
        * same subject re-derived by the SAME plug-in (a re-run, possibly a
          newer version): previous claims become 'superseded' - retained for
          history, never deleted;
        * same subject already claimed by a DIFFERENT plug-in: the new claim
          enters as 'competing' - both interpretations coexist until a human
          review accepts one and rejects the other.
        """
        if claim.subject and claim.status == "accepted":
            for c in self.ledger.claims(
                mission_id=claim.mission_id, kind=claim.kind, status="accepted"
            ):
                if c.get("subject") != claim.subject or c["activity_id"] == claim.activity_id:
                    continue
                other_plugin = (self.ledger.activity(c["activity_id"]) or {}).get("plugin")
                if other_plugin == plugin_name:
                    self.ledger.set_claim_status(c["id"], "superseded")
                else:
                    claim.status = "competing"
        self.ledger.add_claim(claim)
        return claim.status

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
