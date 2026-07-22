"""Read-only HTTP API + live viewer over a project's site state (stdlib).

Endpoints:
  GET /                      - index page with links
  GET /viewer                - interactive map: click anywhere to query the
                               site state at that point (uses /api/query)
  GET /report                - the HTML report for the served version
  GET /package.json          - the machine-readable package export
  GET /api/summary           - version summary + freshness per claim kind
  GET /api/missions          - all missions
  GET /api/claims?kind=&status=  - claims in the served version
  GET /api/claims/<id>       - one claim
  GET /api/claims/<id>/trace - full provenance chain of a claim
  GET /api/query?x=..&y=..   - site-state belief at a project-frame point
  GET /api/plan              - proposed next capture plan (coverage gaps +
                               review recapture requests)

Local development server: no auth, no writes, binds 127.0.0.1. Put a real
gateway in front before exposing project data beyond localhost
(security/ownership NFR).
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .ledger.ledger import ObservationLedger
from .outputs import HtmlReport, JsonPackageExport
from .outputs.svg import bbox_overlay, grid_svg, trajectory_overlay
from .planning import propose_capture_plan
from .query import SiteStateQuery

_INDEX = """<html><body style="font-family:sans-serif;margin:3rem">
<h1>Site State Platform</h1><ul>
<li><a href="/viewer">Interactive viewer (click to query)</a></li>
<li><a href="/report">Human-reviewable report</a></li>
<li><a href="/package.json">Machine-readable package</a></li>
<li><a href="/api/summary">/api/summary</a></li>
<li><a href="/api/missions">/api/missions</a></li>
<li><a href="/api/claims">/api/claims</a></li>
<li>/api/claims/&lt;id&gt;[/trace]</li>
<li><a href="/api/query?x=7.0&amp;y=4.0">/api/query?x=7.0&amp;y=4.0</a></li>
<li><a href="/api/plan">/api/plan</a> (next capture proposal)</li>
</ul></body></html>"""

_OCC_COLORS = {1: "#334155", 0: "#e7edf5", -1: "#c9ced6"}


def _build_viewer(ledger: ObservationLedger, version: dict) -> str:
    """One self-contained interactive page: latest occupancy map with change
    overlays; clicking queries /api/query and shows the belief at the point."""
    claims = [c for c in (ledger.claim(cid) for cid in version["claim_ids"]) if c]
    geoms = [c for c in claims
             if c["kind"] == "occupancy_geometry" and c["status"] == "accepted"]
    if not geoms:
        return "<html><body>No occupancy geometry in this version yet.</body></html>"
    geom = max(geoms, key=lambda c: c["observed_at"])
    grid = ledger.evidence_payload(geom["payload"]["evidence_id"])
    ny, nx = grid["occ"].shape
    x0, y0, res = float(grid["x0"]), float(grid["y0"]), float(grid["res"])

    overlays = ""
    for traj in claims:
        if traj["kind"] == "trajectory" and traj["mission_id"] == geom["mission_id"]:
            tp = ledger.evidence_payload(traj["payload"]["evidence_id"])
            overlays += trajectory_overlay(tp["poses"], x0, y0, ny, res)
    for ch in claims:
        if ch["kind"] == "change" and ch["status"] == "accepted":
            color = "#16a34a" if ch["payload"]["change_type"] == "appeared" else "#dc2626"
            overlays += bbox_overlay(ch["payload"]["bbox"], x0, y0, ny, res, color=color)
    overlays += "<circle id='probe' r='6' fill='none' stroke='#0ea5e9' stroke-width='3' visibility='hidden'/>"

    svg = grid_svg(grid["occ"], _OCC_COLORS, x0, y0, res, overlays=overlays)
    h_px = ny * res * 30.0
    return f"""<html><head><meta charset='utf-8'><title>Site State Viewer</title>
<style>
body{{font-family:system-ui,sans-serif;margin:1.5rem;display:flex;gap:1.5rem;flex-wrap:wrap}}
#map{{flex:2 1 560px;max-width:900px}}
#panel{{flex:1 1 280px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
       padding:1rem;font-size:.85rem;min-height:200px}}
#panel h2{{margin-top:0;font-size:1rem}}
dt{{color:#64748b;margin-top:.5rem}} dd{{margin:0;font-weight:600}}
.hint{{color:#64748b;font-size:.8rem}}
</style></head><body>
<div id="map">
<h1 style="font-size:1.2rem">Site State — click the map to query</h1>
<div id="svgwrap">{svg}</div>
<p class="hint">Version <code>{version["id"]}</code>. Green = appeared, red = disappeared,
blue = trajectory. The answer always includes coverage, freshness and the claim ids
behind it.</p>
</div>
<div id="panel"><h2>Point query</h2><div id="out" class="hint">Click anywhere on the map.</div></div>
<script>
const X0={x0}, Y0={y0}, RES={res}, NY={ny}, PXM=30.0, HPX={h_px};
const svg = document.querySelector('#svgwrap svg');
svg.style.cursor = 'crosshair';
svg.addEventListener('click', async (e) => {{
  const r = svg.getBoundingClientRect();
  const vb = svg.viewBox.baseVal;
  const px = (e.clientX - r.left) * (vb.width / r.width);
  const py = (e.clientY - r.top) * (vb.height / r.height);
  const wx = X0 + px / PXM;
  const wy = Y0 + (HPX - py) / PXM;
  const probe = document.getElementById('probe');
  probe.setAttribute('cx', px); probe.setAttribute('cy', py);
  probe.setAttribute('visibility', 'visible');
  const res_ = await fetch(`/api/query?x=${{wx.toFixed(2)}}&y=${{wy.toFixed(2)}}`);
  const d = await res_.json();
  const claims = (d.claims_here || []).map(c =>
      `<li>${{c.kind}}${{c.payload.change_type ? ' · ' + c.payload.change_type : ''}}` +
      `${{c.payload.top_class ? ' · ' + c.payload.top_class : ''}}` +
      ` (conf ${{(c.confidence ?? 0).toFixed(2)}})</li>`).join('') || '<li>none</li>';
  document.getElementById('out').innerHTML = `
    <dl>
    <dt>point</dt><dd>(${{wx.toFixed(2)}}, ${{wy.toFixed(2)}}) m
      ${{d.zone ? '· ' + d.zone : ''}}</dd>
    <dt>occupancy</dt><dd>${{d.occupancy}}</dd>
    <dt>coverage</dt><dd>${{d.coverage}}</dd>
    <dt>traversability</dt><dd>${{d.traversability}}</dd>
    <dt>freshness</dt><dd>${{d.freshness && d.freshness.freshness !== undefined
        ? (d.freshness.freshness*100).toFixed(0) + '%' : 'n/a'}}</dd>
    <dt>claims at this point</dt><dd><ul style="margin:.2rem 0 0 1rem;padding:0">${{claims}}</ul></dd>
    <dt>occupancy source claim</dt><dd style="font-weight:400"><code>${{(d.sources||{{}}).occupancy || '—'}}</code></dd>
    </dl>`;
}});
</script></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    ledger: ObservationLedger
    version: dict
    exports_dir: Path
    viewer_html: str

    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        from .ledger.ledger import _jsonable

        # strict JSON on the wire: non-finite floats (e.g. an unbounded
        # freshness age) must never leak as invalid NaN/Infinity literals
        body = json.dumps(_jsonable(obj), indent=2, allow_nan=False).encode()
        self._send(code, body, "application/json")

    def do_GET(self):  # noqa: N802 (stdlib API)
        url = urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        q = parse_qs(url.query)
        try:
            if not parts:
                self._send(200, _INDEX.encode(), "text/html")
            elif parts == ["healthz"]:
                self._json({"status": "ok", "version": _package_version(),
                            "site_state_version": self.version["id"]})
            elif parts == ["version"]:
                self._json({"sitestate": _package_version(),
                            "site_state_version": self.version["id"],
                            "label": self.version.get("label", "")})
            elif parts == ["viewer"]:
                self._send(200, self.viewer_html.encode(), "text/html")
            elif parts == ["report"]:
                self._send(200, (self.exports_dir / "report.html").read_bytes(), "text/html")
            elif parts == ["package.json"]:
                self._send(
                    200, (self.exports_dir / "package.json").read_bytes(), "application/json"
                )
            elif parts == ["api", "summary"]:
                self._json(SiteStateQuery(self.ledger, self.version["id"]).summary())
            elif parts == ["api", "missions"]:
                self._json(self.ledger.missions())
            elif parts == ["api", "plan"]:
                self._json(propose_capture_plan(self.ledger, self.version["id"]))
            elif parts == ["api", "claims"]:
                claims = [
                    c
                    for c in (self.ledger.claim(cid) for cid in self.version["claim_ids"])
                    if c
                    and (not q.get("kind") or c["kind"] == q["kind"][0])
                    and (not q.get("status") or c["status"] == q["status"][0])
                ]
                self._json(claims)
            elif len(parts) >= 3 and parts[:2] == ["api", "claims"]:
                claim = self.ledger.claim(parts[2])
                if claim is None:
                    self._json({"error": f"unknown claim {parts[2]}"}, 404)
                elif len(parts) == 4 and parts[3] == "trace":
                    self._json(self.ledger.trace(parts[2]))
                else:
                    self._json(claim)
            elif parts == ["api", "query"]:
                x, y = float(q["x"][0]), float(q["y"][0])
                self._json(SiteStateQuery(self.ledger, self.version["id"]).at_point(x, y))
            else:
                self._json({"error": "not found"}, 404)
        except (KeyError, ValueError, FileNotFoundError, RuntimeError) as exc:
            self._json({"error": str(exc)}, 400)


def _package_version() -> str:
    from . import __version__

    return __version__


def make_server(
    project_dir: str | Path, port: int = 8752, version_id: str = "",
    host: str = "127.0.0.1",
) -> ThreadingHTTPServer:
    """Build (but do not start) the server; port 0 picks a free port."""
    ledger = ObservationLedger(Path(project_dir))
    versions = ledger.versions()
    if not versions:
        raise RuntimeError("no committed site-state version in this project")
    version = ledger.version(version_id) if version_id else versions[-1]
    if version is None:
        raise KeyError(f"unknown version {version_id}")

    exports = Path(project_dir) / "exports"
    exports.mkdir(exist_ok=True)
    HtmlReport().render(ledger, version, exports)
    JsonPackageExport().render(ledger, version, exports)

    _Handler.ledger = ledger
    _Handler.version = version
    _Handler.exports_dir = exports
    _Handler.viewer_html = _build_viewer(ledger, version)
    return ThreadingHTTPServer((host, port), _Handler)


def serve(project_dir: str | Path, port: int = 8752, version_id: str = "",
          host: str = "127.0.0.1") -> None:
    server = make_server(project_dir, port, version_id, host=host)
    if host not in ("127.0.0.1", "localhost"):
        print("WARNING: binding beyond localhost - this server has no "
              "authentication; front it with a gateway before exposing "
              "project data (security/ownership NFR).")
    print(f"serving on http://{host}:{server.server_address[1]} "
          f"(viewer: /viewer, API: /api/..., health: /healthz)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
