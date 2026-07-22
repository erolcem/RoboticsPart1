"""Scene-graph output adapter (v2.1).

Exports the site state as a layered scene graph in the spirit of
Hydra/Khronos (docs/sota-review.md §6): site → zones → entities/assets,
with typed edges (`in_zone`, `changed_since`, `deviates_from_plan`) and
the temporal dimension carried by claim `observed_at` timestamps and the
version chain. Task planners and navigation stacks consume this instead
of raw grids when they need *things and rooms*, not cells - and every
node keeps its claim id, so the full evidence chain stays one
`/api/claims/<id>/trace` away.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..plugins.base import OutputAdapter


class SceneGraphExport(OutputAdapter):
    name = "scene-graph"

    def render(self, ledger, version: dict[str, Any], out_dir: Path) -> list[Path]:
        claims = [c for c in (ledger.claim(cid) for cid in version["claim_ids"]) if c]
        project_file = ledger.root / "project.json"
        project = json.loads(project_file.read_text()) if project_file.exists() else {}

        def of_kind(kind: str) -> list[dict]:
            return [c for c in claims
                    if c["kind"] == kind and c["status"] in ("accepted", "competing")]

        def node(c: dict, extra_keys: tuple[str, ...]) -> dict[str, Any]:
            return {
                "claim_id": c["id"],
                "status": c["status"],
                "confidence": c["confidence"],
                "observed_at": c["observed_at"],
                "centroid": c["payload"].get("centroid"),
                "bbox": c["payload"].get("bbox"),
                **{k: c["payload"][k] for k in extra_keys if k in c["payload"]},
            }

        entities = [node(c, ("top_class", "class_probs", "zone", "extent_m"))
                    for c in of_kind("entity")]
        assets = [node(c, ("tag", "position", "position_sigma_m", "zone"))
                  for c in of_kind("asset")]
        changes = [node(c, ("change_type", "zone", "baseline_mission_id",
                            "likely_registration_artifact"))
                   for c in of_kind("change")]
        deviations = [node(c, ("deviation_type", "zone", "plan_name"))
                      for c in of_kind("deviation")]

        edges: list[dict[str, str]] = []
        for group in (entities, assets, changes, deviations):
            for n in group:
                if n.get("zone"):
                    edges.append({"type": "in_zone", "from": n["claim_id"],
                                  "to": n["zone"]})
        for n in changes:
            if n.get("baseline_mission_id"):
                edges.append({"type": "changed_since", "from": n["claim_id"],
                              "to": n["baseline_mission_id"]})
        for n in deviations:
            edges.append({"type": "deviates_from_plan", "from": n["claim_id"],
                          "to": n.get("plan_name", "design")})

        graph = {
            "schema": "sitestate/scene-graph@1.0",
            "version": version["id"],
            "layers": {
                "site": {
                    "name": project.get("name", "unnamed site"),
                    "bounds": project.get("bounds"),
                },
                "zones": [
                    {"name": name, "bbox": bbox}
                    for name, bbox in (project.get("zones") or {}).items()
                ],
                "entities": entities,
                "assets": assets,
                "changes": changes,
                "deviations": deviations,
            },
            "edges": edges,
            "notes": [
                "layered scene graph: site -> zones -> entities/assets, with "
                "change/deviation nodes carrying the temporal dimension",
                "every node keeps its claim_id: provenance via ledger.trace()",
            ],
        }
        path = out_dir / "scene_graph.json"
        path.write_text(json.dumps(graph, indent=2))
        return [path]
