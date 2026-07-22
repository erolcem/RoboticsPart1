"""Machine-readable package export (JSON).

The consumer-facing contract deliberately does not expose internal ROS or
storage details: it is plain JSON referencing evidence files by id/path,
with provenance chains resolved for every claim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..plugins.base import OutputAdapter


class JsonPackageExport(OutputAdapter):
    name = "json-package"

    def render(self, ledger, version: dict[str, Any], out_dir: Path) -> list[Path]:
        claims = [ledger.claim(cid) for cid in version["claim_ids"]]
        claims = [c for c in claims if c]
        package = {
            "schema": "sitestate/package@0.1",
            "version": version,
            "missions": [ledger.mission(m) for m in version["mission_ids"]],
            "sensors": ledger.sensors(),
            "claims": claims,
            "provenance": {c["id"]: self._trace_summary(ledger, c["id"]) for c in claims},
        }
        path = out_dir / "package.json"
        path.write_text(json.dumps(package, indent=2))
        return [path]

    @staticmethod
    def _trace_summary(ledger, claim_id: str) -> dict[str, Any]:
        trace = ledger.trace(claim_id)
        act = trace["activity"]
        return {
            "plugin": act.get("plugin"),
            "plugin_version": act.get("plugin_version"),
            "params": act.get("params", {}),
            "evidence": [
                {"id": e["id"], "kind": e["kind"], "path": e["path"], "sha256": e["sha256"]}
                for e in trace["evidence"]
            ],
            "sensors": [
                {
                    "id": s["id"],
                    "name": s["manifest"]["name"],
                    "type": s["manifest"]["sensor_type"],
                    "calibration_version": s["manifest"]["calibration_version"],
                }
                for s in trace["sensors"]
            ],
        }
