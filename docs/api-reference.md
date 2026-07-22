# API reference (public Python surface)

The names below are the supported programmatic interface; anything not
listed is internal and may change without notice. Payload/field schemas
live in [data-reference.md](data-reference.md).

## Orchestration — `sitestate`

```python
from sitestate import SiteStatePlatform, load_platform
```

| call | purpose |
|---|---|
| `SiteStatePlatform(root, project=None)` | open/create a project directory; `project` dict is persisted to `project.json`, omit it to load the existing one |
| `load_platform(root)` | open an existing project with every built-in plug-in and output registered |
| `.subscribe(adapter) -> Sensor` / `.unsubscribe(sensor_id)` | attach/detach a `SensorAdapter`; health is checked and recorded |
| `.run_mission(name, carrier, duration, dt=0.5, operator="", area="") -> Mission` | supervised capture: polls every healthy adapter over time |
| `.process(plugin_name, mission_id, **params) -> ProcessingActivity` | run one plug-in; suitability-checked, provenance-recorded, supersede/competing applied |
| `.process_all(mission_id, baseline_mission_id="", params=None) -> list[Activity]` | dependency-ordered pipeline to fixpoint; cross-mission plug-ins get the baseline; unrunnable ones leave `skipped` records |
| `.commit_version(label, mission_ids=None) -> SiteStateVersion` | snapshot current accepted+competing claims |
| `.export(output_name, version_id, out_dir) -> list[Path]` | run an output adapter (`html-report`, `json-package`, `robot-costmap`, `scene-graph`) |
| `.registry` | `PluginRegistry`: `register_processor/or/output`, `suitable_processors`, `missing_inputs` |

## Plug-in contracts — `sitestate.plugins.base`

| class | implement |
|---|---|
| `SensorAdapter` | `manifest -> SensorManifest`, `health_check() -> {"ok", "notes"}`, `sample(t) -> list[Sample]` |
| `ProcessingPlugin` | `manifest -> PluginManifest` (`consumes`, `produces`, `cross_mission`), `run(ctx, **params)` |
| `OutputAdapter` | `name`, `render(ledger, version, out_dir) -> list[Path]` |
| `ProcessingContext` (given to `run`) | read: `observations`, `evidence_for`, `payload`, `claims`; write: `store_derived`, `emit_claim`, `note`. All reads are provenance-tracked. |

## Ledger — `sitestate.ledger.ObservationLedger`

Read side (what outputs/analysis use): `mission(id)`, `missions()`,
`observations(mission_id, data_type=None)`, `sensor(id)`, `sensors()`,
`evidence(id)`, `evidence_payload(id) -> dict[str, ndarray]`,
`claim(id)`, `claims(mission_id=None, kind=None, status=None)`,
`version(id)`, `versions()`, `reviews(claim_id=None)`, and
**`trace(claim_id)`** — the full provenance chain (claim → activity →
evidence → observations → sensors).

## Analysis & workflow

| call | purpose |
|---|---|
| `sitestate.query.SiteStateQuery(ledger, version_id="")` | `.at_point(x, y)` (occupancy/coverage/traversability/zone/freshness/confidences/claims/sources), `.summary()`, `.freshness(kind)` |
| `sitestate.review.ReviewQueue(ledger)` | `.pending(min_confidence=0.7)`, `.accept/.reject/.request_recapture(claim_id, reviewer, note)`, `.recapture_requests()` |
| `sitestate.planning.propose_capture_plan(ledger, version_id="", ...)` | information-gain-ordered waypoint tour over coverage gaps + recapture requests |
| `sitestate.benchmark.run_scenario(seed, root)` / `run_benchmark(seeds, out_dir)` | ground-truth scoring; returns/writes the measures + calibration section |
| `sitestate.serve.make_server(project_dir, port, version_id="", host="127.0.0.1")` / `serve(...)` | HTTP API + viewer (make_server with `port=0` for tests) |
| `sitestate.ingest.export_mission_dataset(ledger, mission_id, out_dir)` / `FileReplayAdapter(dataset_dir, sensor_name)` | portable dataset round-trip |

## Design helpers — `sitestate.design`

`FloorPlan(name, walls)` / `.from_dict` / `.rasterize(x0, y0, nx, ny, res)`;
`zones.zone_of(zones_dict, x, y) -> str`.

## Grid & estimation toolkit — `sitestate.processing`

For plug-in authors: `grid.Grid2D` (`integrate_scan`, `classify`,
`probability`, `evidential`, `decision_entropy`), `grid.dilate`,
`grid.connected_regions`, `grid.region_stats`; `icp.rigid_fit`,
`icp.icp_2d`, `icp.PointHash`, `icp.transform_pose`;
`refine.PoseGraph` (`add_odometry/landmark/absolute`,
`optimize(kernel="gnc"|"huber")`).

## Stability policy

Claim kinds, evidence payload keys, export schemas (`sitestate/*@N.M`)
and the three plug-in ABCs are versioned contracts: additive changes bump
the minor version, breaking changes the major (see CONTRIBUTING.md).
