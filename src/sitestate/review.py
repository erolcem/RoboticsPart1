"""Human review workflow (non-functional requirement: uncertain or
conflicting claims can be inspected, accepted, rejected or scheduled for
re-capture).

Reviews never edit claims in place: each action is an audit-trail row in
the ledger, and accept/reject only flips the claim's status. The evidence
and the original interpretation remain traceable forever.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ledger.ledger import ObservationLedger

# claim kinds a human decision typically hangs on
DECISION_KINDS = ("change", "deviation", "entity", "alignment_check", "calibration_check")


class ReviewQueue:
    def __init__(self, ledger: "ObservationLedger"):
        self.ledger = ledger

    def pending(self, min_confidence: float = 0.7) -> list[dict]:
        """Claims needing a human eye: every competing claim, plus accepted
        decision-relevant claims below the confidence threshold - excluding
        anything already reviewed."""
        reviewed = {r["claim_id"] for r in self.ledger.reviews()}
        out: list[dict] = []
        for c in self.ledger.claims(status="competing"):
            if c["id"] not in reviewed:
                out.append(c)
        for c in self.ledger.claims(status="accepted"):
            if (
                c["id"] not in reviewed
                and c["kind"] in DECISION_KINDS
                and c["confidence"] < min_confidence
            ):
                out.append(c)
        return sorted(out, key=lambda c: c["confidence"])

    def accept(self, claim_id: str, reviewer: str, note: str = "") -> None:
        self.ledger.set_claim_status(claim_id, "accepted")
        self.ledger.add_review(claim_id, "accepted", reviewer, note)

    def reject(self, claim_id: str, reviewer: str, note: str = "") -> None:
        self.ledger.set_claim_status(claim_id, "rejected")
        self.ledger.add_review(claim_id, "rejected", reviewer, note)

    def request_recapture(self, claim_id: str, reviewer: str, note: str = "") -> None:
        """Flag the claim's area for re-capture on the next mission; the
        claim keeps its status - the request is planning metadata."""
        self.ledger.add_review(claim_id, "recapture_requested", reviewer, note)

    def recapture_requests(self) -> list[dict]:
        requests = [r for r in self.ledger.reviews() if r["action"] == "recapture_requested"]
        for r in requests:
            claim = self.ledger.claim(r["claim_id"]) or {}
            r["region"] = (claim.get("payload") or {}).get("bbox")
            r["claim_kind"] = claim.get("kind")
        return requests
