"""Human-reviewable HTML report: the lightweight viewer of the MVP.

One self-contained file showing occupancy maps, trajectory, coverage,
changes with confidence, uncertainty/registration quality, sensor
manifests and evidence provenance.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import numpy as np

from ..plugins.base import OutputAdapter
from .svg import bbox_overlay, grid_svg, markers_overlay, trajectory_overlay

_OCC_COLORS = {1: "#334155", 0: "#e7edf5", -1: "#c9ced6"}
_COV_COLORS = {2: "#b5e3c0", 1: "#fbe3a2", 0: "#f3b8b8"}

_CSS = """
body{font-family:system-ui,sans-serif;margin:2rem auto;max-width:1080px;padding:0 1rem;
     color:#1e293b;background:#fff}
h1{font-size:1.5rem} h2{font-size:1.15rem;margin-top:2.2rem;border-bottom:1px solid #e2e8f0;
     padding-bottom:.3rem}
table{border-collapse:collapse;width:100%;font-size:.85rem;margin:.8rem 0}
th,td{border:1px solid #e2e8f0;padding:.35rem .6rem;text-align:left;vertical-align:top}
th{background:#f8fafc}
.maps{display:flex;flex-wrap:wrap;gap:1.2rem}
.map{flex:1 1 460px;min-width:320px}
.map h3{font-size:.95rem;margin:.3rem 0}
.legend{font-size:.78rem;color:#475569;margin:.3rem 0}
.legend span{display:inline-block;width:.85rem;height:.85rem;border-radius:3px;
     vertical-align:-2px;margin:0 .25rem 0 .8rem}
.badge{display:inline-block;padding:.05rem .5rem;border-radius:999px;font-size:.75rem}
.hi{background:#dcfce7;color:#166534}.mid{background:#fef9c3;color:#854d0e}
.lo{background:#fee2e2;color:#991b1b}
.small{font-size:.78rem;color:#64748b}
code{background:#f1f5f9;padding:.05rem .3rem;border-radius:4px;font-size:.8rem}
"""


def _conf_badge(c: float) -> str:
    cls = "hi" if c >= 0.75 else ("mid" if c >= 0.45 else "lo")
    return f'<span class="badge {cls}">{c:.2f}</span>'


class HtmlReport(OutputAdapter):
    name = "html-report"

    def render(self, ledger, version: dict[str, Any], out_dir: Path) -> list[Path]:
        claims = [c for c in (ledger.claim(cid) for cid in version["claim_ids"]) if c]
        missions = {m: ledger.mission(m) for m in version["mission_ids"]}
        by_kind: dict[str, list[dict]] = {}
        for c in claims:
            by_kind.setdefault(c["kind"], []).append(c)

        parts: list[str] = [f"<style>{_CSS}</style>"]
        parts.append("<h1>Site State Capture Package</h1>")
        parts.append(
            f'<p class="small">Version <code>{version["id"]}</code> — {html.escape(version["label"])} '
            f'— committed {version["created_at"]}</p>'
        )

        # missions
        parts.append("<h2>Missions</h2><table><tr><th>Mission</th><th>Carrier</th>"
                     "<th>Operator</th><th>Start</th><th>End</th><th>Sensors</th></tr>")
        for m in missions.values():
            if not m:
                continue
            parts.append(
                f"<tr><td>{html.escape(m['name'])}<br><code>{m['id']}</code></td>"
                f"<td>{html.escape(str(m['carrier'].get('type', '?')))}</td>"
                f"<td>{html.escape(m.get('operator', ''))}</td>"
                f"<td>{m.get('started_at', '')}</td><td>{m.get('ended_at', '')}</td>"
                f"<td>{len(m.get('sensor_ids', []))}</td></tr>"
            )
        parts.append("</table>")

        # maps per mission occupancy
        parts.append("<h2>Registered spatial model</h2><div class='maps'>")
        mission_order = sorted(
            by_kind.get("occupancy_geometry", []), key=lambda c: c["observed_at"]
        )
        grids: dict[str, dict] = {}
        for geom in mission_order:
            grid = ledger.evidence_payload(geom["payload"]["evidence_id"])
            grids[geom["mission_id"]] = grid
            ny = grid["occ"].shape[0]
            overlays = ""
            for traj in by_kind.get("trajectory", []):
                if traj["mission_id"] == geom["mission_id"]:
                    tp = ledger.evidence_payload(traj["payload"]["evidence_id"])
                    overlays += trajectory_overlay(
                        tp["poses"], float(grid["x0"]), float(grid["y0"]), ny, float(grid["res"])
                    )
            mname = (missions.get(geom["mission_id"]) or {}).get("name", geom["mission_id"])
            parts.append(
                f"<div class='map'><h3>{html.escape(mname)} — occupancy "
                f"(confidence {_conf_badge(geom['confidence'])})</h3>"
                + grid_svg(grid["occ"], _OCC_COLORS, float(grid["x0"]), float(grid["y0"]),
                           float(grid["res"]), overlays=overlays)
                + "<div class='legend'>occupied<span style='background:#334155'></span>"
                  "free<span style='background:#e7edf5'></span>"
                  "unknown<span style='background:#c9ced6'></span>"
                  "trajectory<span style='background:#2563eb'></span></div></div>"
            )
        parts.append("</div>")

        # coverage
        if by_kind.get("coverage"):
            parts.append("<h2>Coverage — what was actually seen</h2><div class='maps'>")
            for cov in by_kind["coverage"]:
                grid = ledger.evidence_payload(cov["payload"]["evidence_id"])
                fr = cov["payload"]["fractions"]
                mname = (missions.get(cov["mission_id"]) or {}).get("name", cov["mission_id"])
                parts.append(
                    f"<div class='map'><h3>{html.escape(mname)} — coverage</h3>"
                    + grid_svg(grid["coverage"], _COV_COLORS, float(grid["x0"]),
                               float(grid["y0"]), float(grid["res"]))
                    + f"<div class='legend'>observed {fr['observed']:.0%}"
                      "<span style='background:#b5e3c0'></span>"
                      f"insufficient {fr['insufficient']:.0%}"
                      "<span style='background:#fbe3a2'></span>"
                      f"unobserved {fr['unobserved']:.0%}"
                      "<span style='background:#f3b8b8'></span></div></div>"
                )
            parts.append("</div>")

        # change set
        changes = by_kind.get("change", [])
        summary = (by_kind.get("change_summary") or [None])[0]
        parts.append("<h2>Change set</h2>")
        if summary:
            parts.append(
                f"<p>{summary['payload']['n_regions']} change region(s) detected; "
                f"{summary['payload']['comparable_fraction']:.0%} of the area was observed "
                f"well enough in both missions to be compared. Areas outside that fraction "
                f"are reported as <em>not comparable</em>, never as unchanged.</p>"
            )
        if changes:
            cur_mission = changes[0]["mission_id"]
            grid = grids.get(cur_mission)
            if grid is not None:
                ny = grid["occ"].shape[0]
                overlays = ""
                for idx, ch in enumerate(changes, 1):
                    color = "#16a34a" if ch["payload"]["change_type"] == "appeared" else "#dc2626"
                    overlays += bbox_overlay(
                        ch["payload"]["bbox"], float(grid["x0"]), float(grid["y0"]), ny,
                        float(grid["res"]), color=color, label=f"#{idx}",
                    )
                parts.append(
                    "<div class='map'>"
                    + grid_svg(grid["occ"], _OCC_COLORS, float(grid["x0"]), float(grid["y0"]),
                               float(grid["res"]), overlays=overlays)
                    + "<div class='legend'>appeared<span style='background:#16a34a'></span>"
                      "disappeared<span style='background:#dc2626'></span></div></div>"
                )
            parts.append(
                "<table><tr><th>#</th><th>Type</th><th>Location (m)</th><th>Area</th>"
                "<th>Confidence</th><th>Evidence</th></tr>"
            )
            for idx, ch in enumerate(changes, 1):
                p = ch["payload"]
                ev_links = ", ".join(f"<code>{e}</code>" for e in ch["evidence_ids"])
                parts.append(
                    f"<tr><td>{idx}</td><td>{p['change_type']}</td>"
                    f"<td>({p['centroid'][0]:.1f}, {p['centroid'][1]:.1f})</td>"
                    f"<td>{p['area_m2']:.2f} m²</td><td>{_conf_badge(ch['confidence'])}</td>"
                    f"<td class='small'>{ev_links}</td></tr>"
                )
            parts.append("</table>")
        else:
            parts.append("<p>No change claims in this version.</p>")

        # uncertainty & registration quality
        parts.append("<h2>Uncertainty and registration quality</h2>")
        parts.append("<table><tr><th>Mission</th><th>Registration RMSE</th>"
                     "<th>Control points</th><th>Max pose σ</th><th>Confidence</th></tr>")
        for reg in by_kind.get("registration", []):
            traj = next(
                (t for t in by_kind.get("trajectory", []) if t["mission_id"] == reg["mission_id"]),
                None,
            )
            mname = (missions.get(reg["mission_id"]) or {}).get("name", reg["mission_id"])
            parts.append(
                f"<tr><td>{html.escape(mname)}</td>"
                f"<td>{reg['payload'].get('rmse_m', float('nan')) * 100:.1f} cm</td>"
                f"<td>{reg['payload'].get('n_control_points', 0)}</td>"
                f"<td>{(traj['payload']['max_pose_sigma_m'] * 100 if traj else 0):.1f} cm</td>"
                f"<td>{_conf_badge(reg['confidence'])}</td></tr>"
            )
        parts.append("</table>")
        parts.append(
            "<p class='small'>Geometric, registration, coverage and freshness uncertainty "
            "are reported separately: a single percentage would hide which decisions each "
            "affects.</p>"
        )

        # sensors
        parts.append("<h2>Sensors and calibration</h2>")
        parts.append("<table><tr><th>Sensor</th><th>Type</th><th>Data types</th>"
                     "<th>Calibration</th><th>Declared accuracy</th><th>Limitations</th></tr>")
        for s in ledger.sensors():
            m = s["manifest"]
            parts.append(
                f"<tr><td>{html.escape(m['name'])}<br><code>{s['id']}</code></td>"
                f"<td>{m['sensor_type']}</td><td>{', '.join(m['data_types'])}</td>"
                f"<td>{m['calibration_version']}</td>"
                f"<td class='small'>{html.escape(str(m['expected_accuracy']))}</td>"
                f"<td class='small'>{html.escape('; '.join(m['limitations']))}</td></tr>"
            )
        parts.append("</table>")

        # provenance appendix
        parts.append("<h2>Evidence & processing provenance</h2>")
        parts.append("<table><tr><th>Claim</th><th>Kind</th><th>Plug-in</th>"
                     "<th>Evidence assets</th><th>Source sensors</th></tr>")
        for c in claims:
            tr = ledger.trace(c["id"])
            sensors = ", ".join(s["manifest"]["name"] for s in tr["sensors"]) or "—"
            parts.append(
                f"<tr><td><code>{c['id']}</code></td><td>{c['kind']}</td>"
                f"<td>{tr['activity'].get('plugin', '?')} "
                f"v{tr['activity'].get('plugin_version', '?')}</td>"
                f"<td>{len(tr['evidence'])}</td><td class='small'>{html.escape(sensors)}</td></tr>"
            )
        parts.append("</table>")
        parts.append(
            "<p class='small'>Every claim above can be traced to raw evidence files "
            "(content-addressed by SHA-256), the processing activity, its parameters and "
            "plug-in version, and the sensor manifests including calibration.</p>"
        )

        path = out_dir / "report.html"
        path.write_text(
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>Site State Capture Package</title></head><body>"
            + "".join(parts)
            + "</body></html>"
        )
        return [path]
