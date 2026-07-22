"""Human-reviewable HTML report: the lightweight viewer of the MVP.

One self-contained file showing occupancy maps, trajectory, coverage,
changes with confidence, uncertainty/registration quality, sensor
manifests and evidence provenance.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from ..plugins.base import OutputAdapter
from .png import data_uri, depth_strip
from .svg import bbox_overlay, grid_svg, markers_overlay, trajectory_overlay

_OCC_COLORS = {1: "#334155", 0: "#e7edf5", -1: "#c9ced6"}
_COV_COLORS = {2: "#b5e3c0", 1: "#fbe3a2", 0: "#f3b8b8"}
_TRAV_COLORS = {0: "#a7d3a9", 1: "#f5c76e", 2: "#334155", 3: "#d5d9df"}

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
.flag{background:#fef3c7;color:#92400e;border-radius:6px;padding:.05rem .45rem;font-size:.72rem}
.strip{image-rendering:pixelated;border:1px solid #cbd5e1;border-radius:3px;
     height:26px;margin:1px 3px 1px 0;vertical-align:middle}
.layers{margin:.4rem 0;font-size:.8rem;color:#475569}
.layers label{margin-right:1rem;cursor:pointer}
"""

_LAYER_SCRIPT = """
<script>
function toggleLayer(cls, on) {
  document.querySelectorAll('.' + cls).forEach(el => {
    el.style.display = on ? '' : 'none';
  });
}
</script>
"""


def _conf_badge(c: float) -> str:
    cls = "hi" if c >= 0.75 else ("mid" if c >= 0.45 else "lo")
    return f'<span class="badge {cls}">{c:.2f}</span>'


class HtmlReport(OutputAdapter):
    name = "html-report"

    def render(self, ledger, version: dict[str, Any], out_dir: Path) -> list[Path]:
        claims = [c for c in (ledger.claim(cid) for cid in version["claim_ids"]) if c]
        missions = {m: ledger.mission(m) for m in version["mission_ids"]}
        project_file = ledger.root / "project.json"
        project = json.loads(project_file.read_text()) if project_file.exists() else {}
        control_points = {
            name: (xy[0], xy[1]) for name, xy in (project.get("control_points") or {}).items()
        }
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
            if control_points:
                overlays += markers_overlay(
                    control_points, float(grid["x0"]), float(grid["y0"]), ny,
                    float(grid["res"]),
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
                  "trajectory<span style='background:#2563eb'></span>"
                  "control points<span style='background:#7c3aed'></span></div></div>"
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
            # per-zone coverage: the site team's rooms, not our coordinates
            zoned = [c for c in by_kind["coverage"] if c["payload"].get("by_zone")]
            if zoned:
                parts.append("<table><tr><th>Mission</th><th>Zone</th><th>Observed</th>"
                             "<th>Insufficient</th><th>Unobserved</th></tr>")
                for cov in zoned:
                    mname = (missions.get(cov["mission_id"]) or {}).get(
                        "name", cov["mission_id"])
                    for zone, zf in cov["payload"]["by_zone"].items():
                        parts.append(
                            f"<tr><td>{html.escape(mname)}</td><td>{html.escape(zone)}</td>"
                            f"<td>{zf['observed']:.0%}</td><td>{zf['insufficient']:.0%}</td>"
                            f"<td>{zf['unobserved']:.0%}</td></tr>"
                        )
                parts.append("</table>")

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
                overlays = "<g class='layer-changes'>"
                for idx, ch in enumerate(changes, 1):
                    color = "#16a34a" if ch["payload"]["change_type"] == "appeared" else "#dc2626"
                    overlays += bbox_overlay(
                        ch["payload"]["bbox"], float(grid["x0"]), float(grid["y0"]), ny,
                        float(grid["res"]), color=color, label=f"#{idx}",
                    )
                overlays += "</g><g class='layer-traj'>"
                for traj in by_kind.get("trajectory", []):
                    if traj["mission_id"] == cur_mission:
                        tp = ledger.evidence_payload(traj["payload"]["evidence_id"])
                        overlays += trajectory_overlay(
                            tp["poses"], float(grid["x0"]), float(grid["y0"]), ny,
                            float(grid["res"]),
                        )
                overlays += "</g>"
                parts.append(
                    "<div class='map'>"
                    "<div class='layers'>"
                    "<label><input type='checkbox' checked "
                    "onchange=\"toggleLayer('layer-changes', this.checked)\"> changes</label>"
                    "<label><input type='checkbox' checked "
                    "onchange=\"toggleLayer('layer-traj', this.checked)\"> trajectory</label>"
                    "</div>"
                    + grid_svg(grid["occ"], _OCC_COLORS, float(grid["x0"]), float(grid["y0"]),
                               float(grid["res"]), overlays=overlays)
                    + "<div class='legend'>appeared<span style='background:#16a34a'></span>"
                      "disappeared<span style='background:#dc2626'></span></div></div>"
                )
            parts.append(
                "<table><tr><th>#</th><th>Type</th><th>Zone</th><th>Location (m)</th>"
                "<th>Area</th><th>Confidence</th><th>Supporting imagery</th></tr>"
            )
            for idx, ch in enumerate(changes, 1):
                p = ch["payload"]
                flag = (" <span class='flag'>possible registration artifact</span>"
                        if p.get("likely_registration_artifact") else "")
                strips = []
                for label, ev_ids in (("now", (p.get("imagery") or {}).get("current", [])),
                                      ("before", (p.get("imagery") or {}).get("baseline", []))):
                    for ev_id in ev_ids[:2]:
                        try:
                            frame = ledger.evidence_payload(ev_id)
                            img = depth_strip(frame["depths"], max_range=8.0)
                            strips.append(
                                f"<img class='strip' title='{label} · depth frame {ev_id}' "
                                f"src='{data_uri(img)}'>"
                            )
                        except (KeyError, FileNotFoundError):
                            continue
                imagery_html = "".join(strips) if strips else "<span class='small'>—</span>"
                parts.append(
                    f"<tr><td>{idx}</td><td>{p['change_type']}{flag}</td>"
                    f"<td>{html.escape(p.get('zone') or '—')}</td>"
                    f"<td>({p['centroid'][0]:.1f}, {p['centroid'][1]:.1f})</td>"
                    f"<td>{p['area_m2']:.2f} m²</td><td>{_conf_badge(ch['confidence'])}</td>"
                    f"<td>{imagery_html}</td></tr>"
                )
            parts.append("</table><p class='small'>Depth strips are the actual sensor "
                         "frames whose field of view covered each region (near = bright); "
                         "every image is an evidence asset traceable in the ledger.</p>")
        else:
            parts.append("<p>No change claims in this version.</p>")

        # deviations from the designed state
        deviations = by_kind.get("deviation", [])
        if deviations or by_kind.get("deviation_summary"):
            parts.append("<h2>Deviations from the designed state</h2>")
            dsum = (by_kind.get("deviation_summary") or [None])[0]
            if dsum:
                parts.append(
                    f"<p>Compared against <em>{html.escape(dsum['payload']['plan_name'])}</em> "
                    f"with a {dsum['payload']['tolerance_m'] * 100:.0f} cm tolerance band: "
                    f"{dsum['payload']['n_regions']} deviation region(s).</p>"
                )
            if deviations:
                grid = grids.get(deviations[0]["mission_id"])
                if grid is not None:
                    ny = grid["occ"].shape[0]
                    overlays = ""
                    for idx, dv in enumerate(deviations, 1):
                        color = ("#d97706" if dv["payload"]["deviation_type"] == "built_not_planned"
                                 else "#7c3aed")
                        overlays += bbox_overlay(
                            dv["payload"]["bbox"], float(grid["x0"]), float(grid["y0"]), ny,
                            float(grid["res"]), color=color, label=f"D{idx}",
                        )
                    parts.append(
                        "<div class='map'>"
                        + grid_svg(grid["occ"], _OCC_COLORS, float(grid["x0"]),
                                   float(grid["y0"]), float(grid["res"]), overlays=overlays)
                        + "<div class='legend'>built, not planned"
                          "<span style='background:#d97706'></span>"
                          "planned, not built<span style='background:#7c3aed'></span></div></div>"
                    )
                parts.append("<table><tr><th>#</th><th>Type</th><th>Location (m)</th>"
                             "<th>Area</th><th>Confidence</th></tr>")
                for idx, dv in enumerate(deviations, 1):
                    p = dv["payload"]
                    parts.append(
                        f"<tr><td>D{idx}</td><td>{p['deviation_type']}</td>"
                        f"<td>({p['centroid'][0]:.1f}, {p['centroid'][1]:.1f})</td>"
                        f"<td>{p['area_m2']:.2f} m²</td>"
                        f"<td>{_conf_badge(dv['confidence'])}</td></tr>"
                    )
                parts.append("</table>")

        # progress vs the designed plan
        progresses = sorted(by_kind.get("progress", []), key=lambda c: c["observed_at"])
        if progresses:
            parts.append("<h2>Progress against the plan</h2>")
            parts.append("<table><tr><th>Mission</th><th>Scope</th><th>Completion</th>"
                         "<th></th><th>Plan observed</th></tr>")
            for pg in progresses:
                mname = (missions.get(pg["mission_id"]) or {}).get("name", pg["mission_id"])
                rows = [("overall", {"completion": pg["payload"]["overall_completion"],
                                     "observed_fraction": pg["payload"]["observed_plan_fraction"]})]
                rows += list(pg["payload"].get("by_zone", {}).items())
                for scope, v in rows:
                    comp = v.get("completion")
                    obs = v.get("observed_fraction")
                    bar = (f"<div style='background:#e2e8f0;border-radius:4px;width:140px'>"
                           f"<div style='background:#60a5fa;height:10px;border-radius:4px;"
                           f"width:{(comp or 0) * 140:.0f}px'></div></div>")
                    parts.append(
                        f"<tr><td>{html.escape(mname)}</td><td>{html.escape(scope)}</td>"
                        f"<td>{f'{comp:.0%}' if comp is not None else 'n/a'}</td><td>{bar}</td>"
                        f"<td class='small'>{f'{obs:.0%}' if obs is not None else 'n/a'}</td></tr>"
                    )
            parts.append("</table><p class='small'>Completion is measured only over "
                         "planned elements whose location was actually observed - a "
                         "coverage gap can never masquerade as demolition.</p>")

        # semantic entities
        entities = by_kind.get("entity", [])
        if entities:
            parts.append("<h2>Entities (semantic interpretation)</h2>")
            parts.append("<table><tr><th>Class</th><th>Class probabilities</th>"
                         "<th>Location (m)</th><th>Extent</th><th>Status</th>"
                         "<th>Confidence</th></tr>")
            for en in entities:
                p = en["payload"]
                probs = ", ".join(f"{k} {v:.2f}" for k, v in
                                  sorted(p["class_probs"].items(), key=lambda kv: -kv[1]))
                parts.append(
                    f"<tr><td>{p['top_class']}</td><td class='small'>{probs}</td>"
                    f"<td>({p['centroid'][0]:.1f}, {p['centroid'][1]:.1f})</td>"
                    f"<td>{p['extent_m'][0]:.1f} × {p['extent_m'][1]:.1f} m</td>"
                    f"<td>{en['status']}</td><td>{_conf_badge(en['confidence'])}</td></tr>"
                )
            parts.append("</table><p class='small'>Classes are probabilities from a "
                         "replaceable model, not facts; competing interpretations from "
                         "different models coexist until reviewed.</p>")

        # traversability (robot-facing view)
        travs = by_kind.get("traversability", [])
        if travs:
            parts.append("<h2>Traversability (robot-ready world model)</h2>"
                         "<div class='maps'>")
            for tv in travs:
                grid = ledger.evidence_payload(tv["payload"]["evidence_id"])
                fr = tv["payload"]["fractions"]
                parts.append(
                    f"<div class='map'><h3>robot radius {tv['payload']['robot_radius_m']} m "
                    f"(confidence {_conf_badge(tv['confidence'])})</h3>"
                    + grid_svg(grid["classes"], _TRAV_COLORS, float(grid["x0"]),
                               float(grid["y0"]), float(grid["res"]))
                    + f"<div class='legend'>traversable {fr['traversable']:.0%}"
                      "<span style='background:#a7d3a9'></span>"
                      f"inflated<span style='background:#f5c76e'></span>"
                      f"obstacle<span style='background:#334155'></span>"
                      f"unknown {fr['unknown']:.0%}"
                      "<span style='background:#d5d9df'></span></div></div>"
                )
            parts.append("</div><p class='small'>Unobserved space is non-traversable "
                         "by policy - the robot is never told an unseen area is clear.</p>")

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

        # pose refinement stats
        refines = by_kind.get("pose_corrections", [])
        if refines:
            parts.append("<table><tr><th>Mission</th><th>Scans refined (ICP)</th>"
                         "<th>Mean pose correction</th><th>Mean scan-fit RMSE</th></tr>")
            for rf in refines:
                p = rf["payload"]
                mname = (missions.get(rf["mission_id"]) or {}).get("name", rf["mission_id"])
                rmse = p.get("mean_icp_rmse_m")
                parts.append(
                    f"<tr><td>{html.escape(mname)}</td>"
                    f"<td>{p['n_refined']}/{p['n_scans']}</td>"
                    f"<td>{p['mean_correction_m'] * 100:.1f} cm</td>"
                    f"<td>{(rmse * 100 if rmse else 0):.1f} cm</td></tr>"
                )
            parts.append("</table>")

        # QA: cross-mission alignment + per-sensor calibration
        aligns = by_kind.get("alignment_check", [])
        calibs = by_kind.get("calibration_check", [])
        if aligns or calibs:
            parts.append("<h2>Quality assurance — the platform checking itself</h2>")
        if aligns:
            parts.append("<table><tr><th>Check</th><th>Residual translation</th>"
                         "<th>Residual rotation</th><th>Verdict</th></tr>")
            for al in aligns:
                p = al["payload"]
                verdict = ("<span class='badge hi'>within tolerance</span>"
                           if p.get("within_tolerance")
                           else "<span class='badge lo'>NEEDS REVIEW</span>")
                parts.append(
                    "<tr><td>cross-mission map alignment (ICP)</td>"
                    f"<td>{p.get('residual_translation_m', 0) * 100:.1f} cm</td>"
                    f"<td>{p.get('residual_rotation_deg', 0):.2f}°</td>"
                    f"<td>{verdict}</td></tr>"
                )
            parts.append("</table>")
        if calibs:
            parts.append("<table><tr><th>Sensor</th><th>Calibration</th>"
                         "<th>Systematic bias</th><th>Spread</th><th>Verdict</th></tr>")
            for cb in calibs:
                p = cb["payload"]
                verdict = ("<span class='badge hi'>ok</span>" if p.get("within_tolerance")
                           else "<span class='badge lo'>BIAS DETECTED</span>")
                parts.append(
                    f"<tr><td>{html.escape(p.get('sensor_name', '?'))}</td>"
                    f"<td>{html.escape(p.get('calibration_version', '?'))}</td>"
                    f"<td>{p.get('bias_magnitude_m', 0) * 100:.1f} cm</td>"
                    f"<td>{p.get('residual_spread_m', 0) * 100:.1f} cm</td>"
                    f"<td>{verdict}</td></tr>"
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

        from .. import __version__
        from ..core.entities import now_iso

        parts.append(
            f"<p class='small' style='margin-top:3rem;border-top:1px solid #e2e8f0;"
            f"padding-top:.6rem'>Generated by Site State Platform v{__version__} "
            f"on {now_iso()} — report contents are reproducible from the ledger "
            f"and this site-state version.</p>"
        )
        path = out_dir / "report.html"
        path.write_text(
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>Site State Capture Package</title></head><body>"
            + "".join(parts)
            + _LAYER_SCRIPT
            + "</body></html>"
        )
        return [path]
