"""Capture planning: close the loop from "what we failed to see" back to
"where to send the carrier next".

Inputs: the latest coverage and traversability grids of a version, plus
any human recapture requests from the review queue. Poorly observed
regions become capture targets; each target gets the nearest traversable
viewpoint; viewpoints are ordered into a greedy nearest-neighbour tour.
The output is a plain JSON plan an operator (or, later, a navigation
stack fed by the costmap export) can follow.
"""

from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

import numpy as np

from .processing.grid import connected_regions, region_stats
from .query import SiteStateQuery
from .review import ReviewQueue
from .design.zones import zone_of

if TYPE_CHECKING:
    from .ledger.ledger import ObservationLedger

UNOBSERVED, INSUFFICIENT, OBSERVED = 0, 1, 2  # coverage codes
TRAVERSABLE = 0  # traversability class code


def propose_capture_plan(
    ledger: "ObservationLedger",
    version_id: str = "",
    min_target_cells: int = 12,
    max_targets: int = 12,
    sensor_reach_m: float = 4.0,
) -> dict[str, Any]:
    q = SiteStateQuery(ledger, version_id)
    cov_claim = q._latest("coverage")
    trav_claim = q._latest("traversability")
    if cov_claim is None:
        raise RuntimeError("no coverage claim in this version; run the pipeline first")
    cov_grid = ledger.evidence_payload(cov_claim["payload"]["evidence_id"])
    cov = cov_grid["coverage"]
    x0, y0, res = float(cov_grid["x0"]), float(cov_grid["y0"]), float(cov_grid["res"])

    trav = None
    if trav_claim is not None:
        trav = ledger.evidence_payload(trav_claim["payload"]["evidence_id"])["classes"]

    project = _load_project(ledger)
    zones = project.get("zones")

    # capture targets: contiguous poorly-observed areas, biggest first.
    # Frontier filter: only areas bordering traversable space are targets -
    # the void outside the building or a sealed interior can never be
    # captured and must not generate waypoints.
    from .processing.grid import dilate

    poorly = (cov == UNOBSERVED) | (cov == INSUFFICIENT)
    if trav is not None:
        poorly &= dilate(trav == TRAVERSABLE, 3)
    targets: list[dict[str, Any]] = []
    for cells in sorted(connected_regions(poorly), key=len, reverse=True):
        if len(cells) < min_target_cells or len(targets) >= max_targets:
            continue
        stats = region_stats(cells, x0, y0, res)
        targets.append(
            {
                "reason": "insufficient_coverage",
                "zone": zone_of(zones, *stats["centroid"]),
                **stats,
            }
        )

    # human recapture requests take priority over coverage gaps
    for req in ReviewQueue(ledger).recapture_requests():
        if not req.get("region"):
            continue
        bx0, by0, bx1, by1 = req["region"]
        targets.insert(
            0,
            {
                "reason": "recapture_requested",
                "claim_id": req["claim_id"],
                "requested_by": req["reviewer"],
                "note": req["note"],
                "zone": zone_of(zones, (bx0 + bx1) / 2, (by0 + by1) / 2),
                "centroid": [(bx0 + bx1) / 2, (by0 + by1) / 2],
                "bbox": req["region"],
                "n_cells": 0,
                "area_m2": (bx1 - bx0) * (by1 - by0),
            },
        )

    # viewpoint per target: nearest traversable cell to the target centroid
    waypoints: list[dict[str, Any]] = []
    if trav is not None and (trav == TRAVERSABLE).any():
        tj, ti = np.nonzero(trav == TRAVERSABLE)
        tx = x0 + (ti + 0.5) * res
        ty = y0 + (tj + 0.5) * res
        for target in targets:
            cx, cy = target["centroid"]
            k = int(np.argmin((tx - cx) ** 2 + (ty - cy) ** 2))
            waypoints.append(
                {
                    "x": float(tx[k]),
                    "y": float(ty[k]),
                    "look_at": [cx, cy],
                    "reason": target["reason"],
                    "zone": target["zone"],
                }
            )
    else:
        waypoints = [
            {"x": t["centroid"][0], "y": t["centroid"][1], "look_at": t["centroid"],
             "reason": t["reason"], "zone": t["zone"]}
            for t in targets
        ]

    # expected information gain per viewpoint: poorly-observed cells within
    # sensor reach (the active-SLAM criterion; see docs/sota-review.md §6)
    pj, pi = np.nonzero(poorly)
    px = x0 + (pi + 0.5) * res
    py = y0 + (pj + 0.5) * res
    for wp in waypoints:
        within = (px - wp["x"]) ** 2 + (py - wp["y"]) ** 2 <= sensor_reach_m**2
        wp["expected_gain_cells"] = int(within.sum())

    # tour: human recapture requests keep priority; the rest are ordered
    # greedily by information gain per metre of travel, so the first stops
    # of the route buy the most map
    ordered: list[dict[str, Any]] = []
    remaining = waypoints[:]
    here = (waypoints[0]["x"], waypoints[0]["y"]) if waypoints else (0.0, 0.0)
    while remaining:
        k = max(
            range(len(remaining)),
            key=lambda i: (
                remaining[i]["reason"] == "recapture_requested",
                (remaining[i].get("expected_gain_cells", 0) + 1)
                / (1.0 + math.hypot(remaining[i]["x"] - here[0],
                                    remaining[i]["y"] - here[1])),
            ),
        )
        nxt = remaining.pop(k)
        ordered.append(nxt)
        here = (nxt["x"], nxt["y"])

    return {
        "schema": "sitestate/capture-plan@2.0",
        "version": q.version["id"],
        "n_targets": len(targets),
        "targets": targets,
        "waypoints": ordered,
        "notes": [
            "waypoints are traversable cells nearest to each poorly-observed region",
            "ordered by expected information gain per metre of travel",
            "recapture_requested targets originate from human review and come first",
        ],
    }


def _load_project(ledger: "ObservationLedger") -> dict[str, Any]:
    import json

    f = ledger.root / "project.json"
    return json.loads(f.read_text()) if f.exists() else {}
