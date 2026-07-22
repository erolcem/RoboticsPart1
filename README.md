# Site State Platform — v2.1

A plug-and-play sensing platform that converts evidence from robots and
other sensor carriers into a **versioned, uncertainty-aware model of a
changing construction site** — the software described in
*Construction Site State Platform — Initial Deliverable Proposal v0.1*
(the docx in this repository).

Sensors **subscribe** through documented adapters. Their observations are
recorded immutably in an **observation ledger**. Replaceable **processing
plug-ins** turn evidence into **claims** — geometry, coverage, changes,
plan deviations, semantics, traversability, and the platform's own QA
verdicts — that always carry confidence and full provenance. **Output
adapters** export human reports, JSON packages and robot-facing costmaps;
a **query API, live viewer and capture planner** close the loop back to
the next mission. A **benchmark harness** scores the whole pipeline
against ground truth, so "better model" is a measured statement.

```bash
pip install -e .                      # only runtime dependency: numpy
sitestate demo --out demo_output      # full two-mission demonstration
open demo_output/report.html          # the human-reviewable package
sitestate serve --project demo_output/project_data
#   -> http://127.0.0.1:8752/viewer   # click the map to query the site state
sitestate benchmark --seeds 5         # score against simulation ground truth
pytest tests/                         # 37 end-to-end tests (~50 s)
python examples/custom_asset_tracking.py   # extension tutorial, executable

docker compose up --build             # or fully containerized:
#   -> http://127.0.0.1:8752/viewer   # (see docs/deployment.md)
```

Measured on the built-in benchmark (randomized frame offsets, drift and
noise; see CHANGELOG for the run): **trajectory RMSE ≈ 1.1 cm with the
pose-graph back-end (2.4 cm from control-point registration alone) · map
precision/recall ≈ 1.00/1.00 · change precision/recall = 1.00/1.00 ·
coverage honesty = 1.00 · provenance traceability = 1.00 · plan
completion correctly reads ≈ 0.96 against the seeded unbuilt wall ·
~1.3 s processing per two-mission scenario.** Confidence calibration
(reliability bins + ECE) is measured and reported, not assumed.

v2 is grounded in a two-round state-of-the-art survey —
**[docs/sota-review.md](docs/sota-review.md)** — mapping current research
(factor-graph SLAM back-ends, KISS-ICP-style odometry, **graduated
non-convexity robust estimation**, evidential occupancy grids, **active-SLAM
information-gain planning**, **Hydra/Khronos-style scene graphs**,
SAM-class change detection, scan-to-BIM segmentation, conformal
calibration, 3DGS reality capture) to what is implemented in-core vs.
what enters later through the plug-in seams. Developer docs:
[api-reference](docs/api-reference.md) ·
[data-reference](docs/data-reference.md) ·
[deployment](docs/deployment.md) · [CONTRIBUTING](CONTRIBUTING.md).

---

## 1. The mental model: four ideas everything hangs on

Before reading any code, internalize these — every module is a direct
consequence of them.

**1. Observation ≠ interpretation.** What a sensor measured (an
*Observation* + its *EvidenceAsset*) is stored forever, untouched. What
the system concludes from it (a *Claim*) is a separate object linking
back to its evidence. When a better model ships next year, you rerun it
over the same evidence and get better claims — no re-visiting the site.
This is why the ledger and the state model are different layers.

**2. Claims are never deleted.** Re-running the *same* plug-in marks old
claims `superseded`. A *different* plug-in claiming the same subject makes
the new claim `competing` — both coexist until a human review accepts one.
A version is a snapshot of claim ids, so you can always reconstruct what
the system believed at any point in time.

**3. Uncertainty is multi-dimensional and honest.** Registration RMSE,
pose drift, map decision entropy, per-cell coverage, class probabilities,
freshness decay, and per-sensor calibration bias are reported separately,
because they affect different decisions. Missing data is *flagged*, never
guessed: an unvisited area is "unobserved", not "unchanged"; a robot is
never told unseen space is clear; a suspected registration echo is shown
with a flag, not silently dropped.

**4. Every boundary is a plug-in contract.** Sensors, processing models,
and outputs each implement a small abstract class with a *manifest*
declaring what they are and what they consume/produce. The platform
checks suitability mechanically and orders the pipeline automatically.
Swapping any part touches nothing else — and the robot/carrier is
deliberately just metadata: the durable asset is the software and the
accumulated evidence.

---

## 2. Repository map

```
Construction_Site_State_Platform_...docx   the founding proposal document
pyproject.toml               packaging; `sitestate` CLI entry point
CHANGELOG.md                 release history
Dockerfile / docker-compose.yml / Makefile / .github/workflows/ci.yml
                             deployment + CI (see docs/deployment.md)
CONTRIBUTING.md              working rules: benchmark discipline, contracts
docs/data-reference.md       exact schemas: claims, evidence, APIs, exports
docs/api-reference.md        supported Python surface, call by call
docs/sota-review.md          two-round SOTA survey -> design decisions
docs/deployment.md           bare-metal, Docker and CI deployment guide
examples/demo_two_missions.py  thin wrapper around sitestate.demo
tests/                       37 end-to-end tests (the acceptance criteria)
src/sitestate/
├── core/entities.py         the 8 data entities + SensorManifest
├── ledger/ledger.py         observation ledger: SQLite + .npz blobs +
│                            provenance trace + review audit trail
├── plugins/
│   ├── base.py              THE three contracts: SensorAdapter, ProcessingPlugin,
│   │                        OutputAdapter (+ manifests, ProcessingContext, Sample)
│   └── registry.py          registration + suitability checking
├── sensors/
│   ├── world.py             simulated site: walls, fiducials, line-of-sight,
│   │                        vectorised raycasting; carrier with realistic drift
│   └── sim.py               4 sim sensors (lidar w/ failure mode, odometry,
│                            fiducial cam w/ LOS + bias injection, depth cam)
├── ingest/replay.py         dataset recorder + FileReplayAdapter (hardware bridge)
├── design/
│   ├── floorplan.py         designed-state model (walls-as-segments, rasterizer)
│   └── zones.py             named zones ("Room A") for site-language reporting
├── processing/
│   ├── grid.py              log-odds Grid2D (vectorised rays, probability,
│   │                        decision entropy, evidential D-S masses)
│   ├── icp.py               shared rigid fit + vectorised 2D ICP
│   ├── registration.py      control-point alignment (Kabsch + residuals)
│   ├── refine.py            pose-graph optimization (odometry + landmark +
│   │                        GNC-robust scan-match factors, Gauss-Newton)
│   ├── mapping.py           log-odds occupancy grid + evidential layer
│   ├── coverage.py          per-cell + per-zone observation density
│   ├── change.py            coverage-aware diff + artifact screening +
│   │                        linked depth imagery per change region
│   ├── plan_compare.py      as-built vs floor plan → deviation claims
│   ├── progress.py          per-zone completion of the designed plan
│   ├── semantics.py         entity classification with class probabilities
│   ├── traversability.py    robot drivability + cost grid
│   └── qa.py                registration-verification (cross-mission ICP) +
│                            calibration-check (sensor-frame bias)
├── statemodel/model.py      claim integration (supersede/competing) + versioning
├── review.py                human review queue: accept/reject/request re-capture
├── query.py                 SiteStateQuery: belief-at-a-point with freshness
├── planning.py              capture planner: coverage gaps + recapture requests
│                            → frontier-filtered waypoint tour
├── benchmark.py             ground-truth scoring of the whole pipeline
├── outputs/
│   ├── svg.py               SVG map rendering helpers
│   ├── png.py               pure-stdlib PNG encoder (embedded evidence images)
│   ├── html_report.py       human package: maps, layers, linked imagery, QA
│   ├── json_export.py       machine package with resolved provenance
│   ├── costmap_export.py    robot costmap (npz+json) + planning scene
│   └── scenegraph_export.py layered scene graph (zones/entities/changes)
├── platform.py              SiteStatePlatform + process_all pipeline runner
│                            + load_platform() factory
├── serve.py                 HTTP API + interactive /viewer (stdlib only)
├── cli.py / __main__.py     `sitestate` command-line interface
└── demo.py                  the executable end-to-end scenario (used by tests)
```

## 3. The data model

Eight entities (proposal §12): `Sensor`, `Mission`, `Observation`,
`EvidenceAsset`, `ProcessingActivity`, `Claim`, `SiteStateVersion`, plus
the review audit trail. JSON rows in SQLite with indexed columns; binary
payloads as compressed `.npz` blobs, content-addressed with SHA-256.
Exact field-level schemas for every claim kind, evidence payload, export
format and API endpoint live in **[docs/data-reference.md](docs/data-reference.md)**.

Claim `status` lifecycle:

```
             re-run of same plug-in            different plug-in, same subject
accepted ────────────────────────► superseded          ┌──► competing
    ▲                                                  │        │ human review
    └── ReviewQueue.accept ◄───────────────────────────┘        ▼
    rejected ◄── ReviewQueue.reject                     (accept one, reject other)
```

`subject` is the stable key that makes this work (e.g. `grid:main`,
`change:appeared:13.2,5.9`). **Provenance is not a separate store — it
emerges from the links**: claim → activity (which recorded exactly which
evidence/claim ids the plug-in read through its context) → evidence →
observation → sensor → manifest/calibration. `ledger.trace(claim_id)`
walks the chain, recursively through derived evidence. Plug-ins *must*
read through `ProcessingContext`, so provenance cannot be forgotten.

## 4. The three plug-in contracts (`plugins/base.py`)

**SensorAdapter** — manifest (identity, data types, units, accuracy,
mounting, calibration, limitations) + `health_check()` + `sample(t)`.
The platform polls adapters in a time loop; an adapter controls its own
rate by returning `[]` between frames. A ROS 2 adapter buffers its topic
callbacks and drains them in `sample()`; `FileReplayAdapter` is a working
example of an adapter over recorded data. Unhealthy sensors are excluded
*and recorded as excluded*.

**ProcessingPlugin** — manifest declares `consumes` (observation types
AND claim kinds — this is how pipelines chain), `produces`, and
`cross_mission` (needs a baseline). `run(ctx)` reads via the context
(tracked), emits derived evidence + claims via the context (wrapped in a
`ProcessingActivity`). Unsuitable configurations fail the activity with
an explanation instead of running.

**OutputAdapter** — `render(ledger, version, out_dir)`. Outputs read only
the ledger and a version snapshot, never acquisition code.

## 5. One full run (what `sitestate demo` does)

1. **Project** — coordinate frame, surveyed control points, bounds,
   designed floor plan, named zones (persisted as `project.json`).
2. **Subscribe** — manifests + health recorded; then **capture**: the
   time loop stores every sample as observation + evidence. The sim
   carrier's *estimated* pose drifts from truth (rigid offset + random
   walk), fiducials are occluded by walls (real line-of-sight), so
   registration is genuinely necessary.
3. **`process_all`** — dependency-ordered pipeline, iterating to fixpoint:
   registration (Kabsch on fiducials, RMSE + per-point residuals; <3
   points ⇒ flagged unregisterable, never guessed) → **pose-graph
   optimization** (odometry + landmark + Huber-robust scan-match factors,
   Gauss-Newton; halves trajectory error vs registration alone) →
   **log-odds mapping with an evidential layer** (probability, decision
   entropy, and Dempster–Shafer *ignorance* vs *conflict* per cell) →
   coverage (overall + per-zone) → **calibration check** (sensor-frame
   bias vs surveyed points) → change detection → **registration
   verification** (cross-mission ICP: maps must coincide; residual beyond
   tolerance ⇒ review) → plan comparison → **progress tracking**
   (per-zone plan completion) → semantics → traversability.
4. **Change claims** carry: zone, confidence (map confidence × observation
   strength × region size), a `likely_registration_artifact` flag when a
   region hugs structure in the other mission (the classic sub-cell
   misregistration echo), and the **actual depth frames whose FOV covered
   the region in both missions** — the report renders them inline so a
   reviewer opens the pictures, not just numbers.
5. **Version → exports → loop closure**: HTML report, JSON package,
   costmap + planning scene; `/viewer` for click-to-query;
   `plan-capture` turns remaining coverage gaps (frontier-filtered to
   reachable space) and human recapture requests into an ordered waypoint
   tour for the next mission.

The demo seeds ground truth: a pallet appears, a partition moves, a
material stack disappears, and a planned wall was never built — all
detected, classified and attributed correctly (verified by tests and
benchmark).

## 6. Uncertainty: where each dimension lives

| Dimension (proposal §11) | Where produced | Where visible |
|---|---|---|
| Registration | Kabsch residuals; cross-mission ICP verification | claim payloads, QA table, every downstream confidence |
| Geometric (pose) | carrier σ per observation; pose-graph factor stats | trajectory/pose_corrections claims, report |
| Map quality | log-odds decision entropy | occupancy claim, map confidence |
| Epistemic vs conflicting evidence | Dempster–Shafer ignorance & conflict masses per cell | occupancy grid evidence, mapping claim, change claims (`evidential_conflict`) |
| Confidence calibration | benchmark reliability bins + ECE vs ground truth | `benchmark_report.json` calibration section |
| Coverage | per-cell + per-zone counts | coverage maps/tables; gates change & plan comparison |
| Semantic | class probability distributions | entity claims, planning scene |
| Temporal (freshness) | `observed_at` = capture time | query API exponential decay per claim kind |
| Calibration | sensor-frame bias vs surveyed points | calibration_check claims, QA table |
| Provenance | context-tracked reads | `ledger.trace`, `/api/claims/<id>/trace`, report appendix |

Confidence formulas are deliberately simple, legible heuristics living
inside each plug-in (confidence is the *model's* statement about itself).
The benchmark keeps them honest.

## 7. Interfaces

**CLI**: `demo`, `init` (new project), `missions`, `process` (full
pipeline on a mission), `claims`, `trace`, `query`, `review
list|accept|reject|recapture`, `export --format html|json|costmap`,
`record` (dump portable dataset), `plan-capture`, `benchmark`, `serve`.

**HTTP** (localhost, read-only): `/viewer` (interactive click-to-query
map), `/report`, `/package.json`, `/api/summary`, `/api/missions`,
`/api/claims[...]`, `/api/claims/<id>/trace`, `/api/query?x=&y=`,
`/api/plan`. No auth by design; gateway before exposing further.

**Robot outputs**: `costmap.npz`/`costmap.json` (0 traversable / 253
inflated / 254 lethal / 255 unknown — ROS costmap convention, plain npz),
`planning_scene.json` (labelled obstacles with class probabilities).

**Dataset bridge**: `dataset.json` + `payloads/*.npz` — the contract a
real capture rig writes; replayed via `FileReplayAdapter`. Round-trip is
tested (byte-identical observation counts, same registration quality).

## 8. How to extend it (recipes)

**Start here:** [examples/custom_asset_tracking.py](examples/custom_asset_tracking.py)
is the extension tutorial in executable form — a complete new sensor type
(RFID scanner) plus a new processing plug-in (asset tracking) added
without touching a single core file, run end-to-end with provenance. It
is exercised by the test suite, so it can never rot.

**Add a real sensor** — implement `SensorAdapter` with an honest manifest;
buffer your driver's async data; return it from `sample(t)`. Easiest
hardware path today: make the rig write the dataset format. Everything
downstream (mapping, QA, reports, provenance) works unchanged.

**Replace a model** — new `ProcessingPlugin` with the same `produces`
under its own name: its claims *compete* with the incumbent's on the same
subjects; `sitestate review list` shows the conflict; accepting one
resolves it. Ship v2 under the *same* name: v1 claims become
`superseded`. Then run `sitestate benchmark` — if change P/R or map
precision moved the wrong way, the "better" model isn't.

**Add an output** — implement `OutputAdapter.render` (see
`costmap_export.py`, ~60 lines); register; `sitestate export`.

**Go 3D** — the seams: `grid.py` (→ voxels/TSDF), `icp.py` +
`registration.py` (Kabsch/ICP are dimension-generic), payload
conventions (`pose_est` → 6-DoF), renderers. Ledger, claims, versions,
review, query, planning, serve are dimension-agnostic.

**Add BIM/IFC** — an importer to `FloorPlan` (or a richer successor)
attached to `project["design"]`; plan comparison and semantics consume
it as-is.

## 9. Design decisions and their reasons

* **SQLite + npz files, no services** — a capture package must be
  archivable, diffable and openable in a decade; swap storage behind
  `ObservationLedger` when multi-user demands it.
* **Polling adapters, synchronous core** — determinism and testability
  first; async lives inside adapters.
* **2D grids first** — every contract is representation-independent; 2D
  made the architecture provable quickly, and §8 names the exact 3D seams.
* **Heuristics with visible confidence + a benchmark** — the platform's
  value is the scaffolding (provenance, versioning, review, honest
  uncertainty, measurement); models are the replaceable part.
* **No ROS dependency anywhere** — ROS 2 is welcome at the edge inside an
  adapter; stored contracts and exports stay plain (proposal §10).
* **The simulator is a permanent asset** — it encodes failure modes
  (drift, occlusion, sensor death, stale calibration, wrong initial
  frame) and gives every test and benchmark ground truth. It is the
  regression harness for the hardware era.

## 10. Testing & measurement

- `tests/test_end_to_end.py` — the original acceptance statement.
- `tests/test_full_vision.py` — Phase 3/4 features, competing claims,
  review, replay round-trip.
- `tests/test_v1_features.py` — refinement stats, QA verdicts, LOS
  occlusion, calibration-bias detection, degraded capture → recapture
  planning, benchmark thresholds, PNG validity, server endpoints.
- `tests/test_v2_features.py` — pose graph beats coarse registration,
  evidential masses are valid, progress tracking (Room A < Room B because
  of the unbuilt wall), calibration reporting, health/version probes.
- `sitestate benchmark` — the proposal's §16 measures as numbers with
  mean ± std plus confidence calibration; `benchmark_report.json` for
  tracking over time. CI runs tests on Python 3.10/3.12, a benchmark
  smoke run, and a Docker build + container smoke test.

## 11. License

Deliberately not chosen yet — picking proprietary vs. open-core is a
business decision for the startup; until then all rights are reserved.
(`pyproject.toml` carries the same note.)

## 12. Roadmap position

Phases 0–4 of the proposal exist in software, measured against a
simulator that now models occlusion, drift, sensor failure and stale
calibration. The remaining work is the physical world: a ROS 2 bag →
dataset converter, 3D representations behind the named seams, IFC
import, a richer live viewer, and Phase 5 task modules consuming the
planning scene. None of it requires touching the core — that is the
point of the architecture.
