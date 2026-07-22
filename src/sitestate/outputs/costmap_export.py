"""Robot-facing output adapter (Phase 4): navigation costmap and a simple
planning scene, consumed by navigation stacks and task planners.

Deliberately plain files (npz + JSON) rather than ROS messages: a ROS 2
bridge node can trivially republish these, but non-ROS consumers are not
forced to link ROS - the interoperability requirement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..plugins.base import OutputAdapter


class RobotCostmapExport(OutputAdapter):
    name = "robot-costmap"

    def render(self, ledger, version: dict[str, Any], out_dir: Path) -> list[Path]:
        claims = [c for c in (ledger.claim(cid) for cid in version["claim_ids"]) if c]
        travs = [
            c for c in claims if c["kind"] == "traversability" and c["status"] == "accepted"
        ]
        if not travs:
            raise RuntimeError(
                "version has no accepted traversability claim; run the "
                "traversability-analysis plug-in first"
            )
        # mission recency breaks observed_at ties (captures seconds apart)
        rank = {m["id"]: i for i, m in enumerate(ledger.missions())}
        trav = max(travs, key=lambda c: (c["observed_at"], rank.get(c["mission_id"], -1)))
        grid = ledger.evidence_payload(trav["payload"]["evidence_id"])

        npz_path = out_dir / "costmap.npz"
        np.savez_compressed(
            npz_path,
            cost=grid["cost"],
            classes=grid["classes"],
            x0=grid["x0"],
            y0=grid["y0"],
            res=grid["res"],
        )
        meta = {
            "schema": "sitestate/costmap@0.1",
            "frame": "project",
            "cost_convention": "0 traversable, 253 inflated, 254 lethal, 255 unknown",
            "robot_radius_m": trav["payload"]["robot_radius_m"],
            "origin": [float(grid["x0"]), float(grid["y0"])],
            "resolution_m": float(grid["res"]),
            "shape_cells": list(grid["cost"].shape),
            "confidence": trav["confidence"],
            "observed_at": trav["observed_at"],
            "source_claim": trav["id"],
            "fractions": trav["payload"]["fractions"],
        }
        meta_path = out_dir / "costmap.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        # planning scene: labelled obstacles for task planning
        entities = [
            c for c in claims if c["kind"] == "entity" and c["status"] in ("accepted", "competing")
        ]
        scene = {
            "schema": "sitestate/planning-scene@0.1",
            "frame": "project",
            "version": version["id"],
            "obstacles": [
                {
                    "claim_id": c["id"],
                    "status": c["status"],
                    "top_class": c["payload"]["top_class"],
                    "class_probs": c["payload"]["class_probs"],
                    "bbox": c["payload"]["bbox"],
                    "centroid": c["payload"]["centroid"],
                    "confidence": c["confidence"],
                    "observed_at": c["observed_at"],
                }
                for c in entities
            ],
        }
        scene_path = out_dir / "planning_scene.json"
        scene_path.write_text(json.dumps(scene, indent=2))
        return [npz_path, meta_path, scene_path]
