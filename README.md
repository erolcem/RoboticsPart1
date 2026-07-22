# Site State Platform (v0.1)

A plug-and-play sensing platform that converts evidence from robots and
other sensor carriers into a **versioned, uncertainty-aware model of a
changing construction site** — the software described in
*Construction Site State Platform — Initial Deliverable Proposal v0.1*.

Sensors **subscribe** to the platform through documented adapters; their
observations are recorded immutably in an evidence ledger; replaceable
processing plug-ins turn evidence into claims (geometry, coverage,
changes) that always carry confidence and full provenance; output
adapters export human reports and machine-readable packages. Nothing is
ever silently overwritten — reprocessing with a better model supersedes
old claims and keeps them for history.

## Quick start

```bash
pip install -e .            # only dependency: numpy
python examples/demo_two_missions.py demo_output
# open demo_output/report.html          (human-reviewable package)
# see  demo_output/package.json         (machine-readable export)
pytest tests/               # end-to-end acceptance tests
```

The demo runs the acceptance scenario from the proposal: capture the same
simulated indoor area on Day 1 and Day 2 (a pallet appears, a partition
moves, a material stack is removed), register both missions to surveyed
control points, detect the changes with confidence, report coverage and
uncertainty, and trace every result back to raw sensor evidence.

## Architecture (proposal §9)

```
 SensorAdapter ──► ObservationLedger ──► ProcessingPlugin ──► SiteStateModel ──► OutputAdapter
 (manifest,        (evidence blobs +     (manifest-driven,    (versioned         (HTML report,
  calibration,      SQLite metadata,      typed claims with    claims, super-     JSON package,
  health checks)    append-only)          provenance)          sede not delete)   later: IFC, USD…)
```

| Layer | Code | Stable contract |
|---|---|---|
| Sensor & carrier adapters | `src/sitestate/sensors/`, contract in `plugins/base.py` | `SensorAdapter`: manifest + health_check + sample |
| Observation ledger | `src/sitestate/ledger/` | append-only evidence (.npz blobs, SHA-256) + SQLite metadata |
| Processing plug-ins | `src/sitestate/processing/` | `ProcessingPlugin`: manifest (consumes/produces) + run(ctx) |
| Site State Model | `src/sitestate/statemodel/` | claims → versions; supersede, never delete |
| Output adapters | `src/sitestate/outputs/` | `OutputAdapter.render(ledger, version, out_dir)` |
| Orchestrator | `src/sitestate/platform.py` | subscribe → run_mission → process → commit_version → export |

## Plug and play (proposal §10)

* **New sensor** — implement `SensorAdapter` (declare manifest: data types,
  units, accuracy, mounting, calibration, limitations; answer health
  checks; emit typed `Sample`s), then `platform.subscribe(adapter)`.
  Nothing downstream changes. A ROS 2 adapter is just another
  implementation of this class; stored contracts stay ROS-free.
* **New model** — implement `ProcessingPlugin` with a manifest declaring
  what it consumes/produces and register it. The platform checks
  suitability against the subscribed sensors, records every run as a
  `ProcessingActivity` (version, params, inputs), and re-runs supersede
  prior claims — so you can reprocess yesterday's evidence with
  tomorrow's model without re-capturing.
* **New output** — implement `OutputAdapter`; it reads only the ledger and
  a site-state version, never acquisition code.
* **Carrier independence** — the carrier is metadata on the mission;
  the pipeline is identical for robot, trolley, backpack or tripod.

## Uncertainty is a first-class output (proposal §11)

Reported separately, never as one merged percentage:

* **Registration** — RMSE against surveyed control points, per-point residuals.
* **Geometric** — pose σ growing with odometry drift, carried into map confidence.
* **Coverage** — per-cell observed / insufficiently observed / unobserved;
  change detection only compares cells both missions actually saw, so a
  missed area is reported as *not comparable*, never as *unchanged*.
* **Semantic/derived** — every claim carries a confidence, and change
  confidence reflects observation strength and region size.
* **Provenance** — every claim traces to evidence assets (content-addressed),
  the processing activity, plug-in version/params, and sensor manifests
  including calibration version. See `ledger.trace(claim_id)`.
* **Degraded operation** — missions with too few control points, failing
  sensors or missing inputs are *flagged* (rejected/failed records), never
  silently guessed.

## Current status vs the proposal roadmap

Implements the software skeleton of **Phase 1–2** (evidence capture,
cross-session alignment, change/coverage/uncertainty package) against a
built-in simulator, so the whole pipeline is testable without hardware.
Real deployment work = writing real `SensorAdapter`s (ROS 2 topics,
vendor SDKs) and stronger processing plug-ins (3D SLAM, ICP refinement,
learned change detection) behind the same contracts. Not yet built:
BIM/IFC alignment (Phase 3), robot-facing outputs (Phase 4), web viewer
beyond the static HTML report.
