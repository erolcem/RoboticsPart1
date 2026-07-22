"""Site-state query API: the machine-readable service other applications
build on. Ask about any project-frame coordinate and get the platform's
current belief WITH its limits: occupancy, coverage, traversability,
freshness (temporal uncertainty with per-kind expiry), confidence and the
claim ids behind the answer - never a bare yes/no.
"""

from __future__ import annotations

import datetime as _dt
import math
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .ledger.ledger import ObservationLedger

# expected change rate per claim kind, as a freshness half-life-ish tau (s);
# construction sites churn, so geometry goes stale in days, not months
DEFAULT_EXPIRY_S = {
    "occupancy_geometry": 7 * 86400.0,
    "traversability": 2 * 86400.0,
    "change": 3 * 86400.0,
    "entity": 7 * 86400.0,
    "progress": 7 * 86400.0,
}

_OCC_NAMES = {-1: "unknown", 0: "free", 1: "occupied"}
_COV_NAMES = {0: "unobserved", 1: "insufficient", 2: "observed"}
_TRAV_NAMES = {0: "traversable", 1: "inflated", 2: "obstacle", 3: "unknown"}


def _age_seconds(iso: str) -> float:
    try:
        then = _dt.datetime.fromisoformat(iso)
    except ValueError:
        return math.inf
    return (_dt.datetime.now(_dt.timezone.utc) - then).total_seconds()


class SiteStateQuery:
    def __init__(
        self,
        ledger: "ObservationLedger",
        version_id: str = "",
        expiry_s: dict[str, float] | None = None,
    ):
        versions = ledger.versions()
        if version_id:
            version = ledger.version(version_id)
            if version is None:
                raise KeyError(f"unknown version {version_id}")
        elif versions:
            version = versions[-1]
        else:
            raise RuntimeError("no committed site-state version to query")
        self.ledger = ledger
        self.version = version
        self.expiry_s = {**DEFAULT_EXPIRY_S, **(expiry_s or {})}
        self.claims = [c for c in (ledger.claim(cid) for cid in version["claim_ids"]) if c]
        # mission recency breaks observed_at ties (captures seconds apart)
        self._mission_rank = {m["id"]: i for i, m in enumerate(ledger.missions())}

    def _latest(self, kind: str) -> dict | None:
        of_kind = [c for c in self.claims if c["kind"] == kind and c["status"] == "accepted"]
        if not of_kind:
            return None
        return max(
            of_kind,
            key=lambda c: (c["observed_at"], self._mission_rank.get(c["mission_id"], -1)),
        )

    def _grid_lookup(self, claim: dict | None, array_key: str, x: float, y: float):
        if claim is None:
            return None, None
        g = self.ledger.evidence_payload(claim["payload"]["evidence_id"])
        i = int((x - float(g["x0"])) / float(g["res"]))
        j = int((y - float(g["y0"])) / float(g["res"]))
        arr = g[array_key]
        if 0 <= j < arr.shape[0] and 0 <= i < arr.shape[1]:
            return int(arr[j, i]), claim
        return None, claim

    def freshness(self, kind: str) -> dict[str, Any]:
        claim = self._latest(kind)
        if claim is None:
            return {"available": False}
        age = _age_seconds(claim["observed_at"])
        tau = self.expiry_s.get(kind, 7 * 86400.0)
        return {
            "available": True,
            "observed_at": claim["observed_at"],
            "age_s": age,
            "freshness": math.exp(-age / tau),
        }

    def at_point(self, x: float, y: float) -> dict[str, Any]:
        occ_val, occ_claim = self._grid_lookup(self._latest("occupancy_geometry"), "occ", x, y)
        cov_val, cov_claim = self._grid_lookup(self._latest("coverage"), "coverage", x, y)
        trav_val, trav_claim = self._grid_lookup(self._latest("traversability"), "classes", x, y)

        nearby = [
            {
                "kind": c["kind"],
                "id": c["id"],
                "confidence": c["confidence"],
                "payload": {
                    k: c["payload"][k]
                    for k in ("change_type", "deviation_type", "top_class", "bbox")
                    if k in c["payload"]
                },
            }
            for c in self.claims
            if c["status"] in ("accepted", "competing")
            and isinstance(c["payload"].get("bbox"), list)
            and c["payload"]["bbox"][0] <= x <= c["payload"]["bbox"][2]
            and c["payload"]["bbox"][1] <= y <= c["payload"]["bbox"][3]
        ]

        return {
            "point": [x, y],
            "version": self.version["id"],
            "occupancy": _OCC_NAMES.get(occ_val, "no_data"),
            "coverage": _COV_NAMES.get(cov_val, "no_data"),
            "traversability": _TRAV_NAMES.get(trav_val, "no_data"),
            "freshness": self.freshness("occupancy_geometry"),
            "confidence": {
                "occupancy": occ_claim["confidence"] if occ_claim else None,
                "coverage": cov_claim["confidence"] if cov_claim else None,
                "traversability": trav_claim["confidence"] if trav_claim else None,
            },
            "claims_here": nearby,
            "sources": {
                "occupancy": occ_claim["id"] if occ_claim else None,
                "coverage": cov_claim["id"] if cov_claim else None,
                "traversability": trav_claim["id"] if trav_claim else None,
            },
        }

    def summary(self) -> dict[str, Any]:
        by_kind: dict[str, int] = {}
        for c in self.claims:
            by_kind[c["kind"]] = by_kind.get(c["kind"], 0) + 1
        return {
            "version": self.version["id"],
            "label": self.version["label"],
            "missions": self.version["mission_ids"],
            "claims_by_kind": by_kind,
            "competing_claims": sum(1 for c in self.claims if c["status"] == "competing"),
            "freshness": {k: self.freshness(k) for k in self.expiry_s},
        }
