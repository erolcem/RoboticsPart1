# Data reference

The exact shapes of everything the platform stores and serves. This is the
contract document: adapters, plug-ins and consumers may rely on what is
written here and nothing else.

## Claim kinds

`ProcessingActivity.status` is one of `running | succeeded | failed |
skipped` (skipped = preconditions absent, nothing broke).

Every claim has: `id`, `kind`, `mission_id`, `activity_id`, `payload`,
`confidence` (0–1), `evidence_ids`, `subject`, `status`
(`accepted | competing | superseded | rejected`), `observed_at` (capture
time of the supporting evidence), `created_at`.

| kind | produced by | payload fields (beyond region fields*) |
|---|---|---|
| `registration` | control-point-registration | `rotation` 2×2, `translation` [x,y], `rotation_deg`, `rmse_m`, `n_control_points`, `per_point_residuals_m`, `n_detections` |
| `trajectory` | control-point-registration | `evidence_id` (trajectory grid), `n_poses`, `max_pose_sigma_m`, `path_length_m` |
| `pose_corrections` | pose-refinement (pose-graph) | `evidence_id`, `method`, `robust_kernel` (gnc/huber), `n_scans`, `n_refined`, `factors` {odometry, landmark, scan_match}, `gauss_newton_iterations`, `final_cost`, `mean_correction_m`, `max_correction_m`, `mean_icp_rmse_m` |
| `occupancy_geometry` | occupancy-mapping | `evidence_id` (occupancy grid), `res_m`, `n_scans`, `registration_rmse_m`, `mean_pose_sigma_m`, `decision_entropy`, `mean_ignorance`, `mean_conflict`, `high_conflict_cells`, `used_pose_corrections`, cell counts |
| `coverage` | coverage-analysis | `evidence_id` (coverage grid), `fractions` {observed, insufficient, unobserved}, `by_zone` {zone: fractions}, `strong_obs` |
| `change` | occupancy-change-detection | `change_type` (appeared/disappeared), `baseline_mission_id`, `median_observations`, `structure_adjacency`, `evidential_conflict`, `likely_registration_artifact`, `zone`, `imagery` {current: [ev ids], baseline: [ev ids]} |
| `change_summary` | occupancy-change-detection | `n_regions`, `n_flagged_as_artifact`, `comparable_fraction`, cell counts |
| `deviation` | plan-comparison | `deviation_type` (built_not_planned/planned_not_built), `plan_name`, `zone` |
| `deviation_summary` | plan-comparison | `plan_name`, `n_regions`, `tolerance_m`, cell counts |
| `progress` | progress-tracking | `plan_name`, `overall_completion`, `observed_plan_fraction`, `by_zone` {zone: {completion, observed_fraction, planned_cells, built_cells}}, `tolerance_m` |
| `entity` | semantic-labeling | `top_class`, `class_probs` (sums to 1), `plan_overlap`, `extent_m` [w,h], `zone` |
| `traversability` | traversability-analysis | `evidence_id` (traversability grid), `robot_radius_m`, `fractions` {traversable, inflated, obstacle, unknown} |
| `alignment_check` | registration-verification | `baseline_mission_id`, `residual_translation_m`, `residual_rotation_deg`, `icp_rmse_m`, `n_pairs`, `tolerance_m`, `within_tolerance` |
| `calibration_check` | calibration-check | `sensor_id`, `sensor_name`, `calibration_version`, `bias_m` [x,y] (sensor frame), `bias_magnitude_m`, `residual_spread_m`, `n_detections`, `tolerance_m`, `within_tolerance` |

\* region-shaped claims (`change`, `deviation`, `entity`) also carry
`centroid` [x,y], `bbox` [x0,y0,x1,y1], `n_cells`, `area_m2` in project
frame metres.

## Evidence payloads (.npz)

All arrays; grids are indexed `[j, i]` = `[y, x]`, world position of cell
`(i, j)` centre = `(x0 + (i+0.5)·res, y0 + (j+0.5)·res)`.

| kind | arrays |
|---|---|
| `scan_2d` | `angles` (sensor frame, rad), `ranges` (m), `hit` (int8), `pose_est` [x,y,θ] (mission-estimated frame), `max_range` |
| `pose_estimate` | `pose_est` [x,y,θ], `t` |
| `fiducial_detection` | `fiducial_id` (str), `position_est` [x,y] (mission-est frame), `relative` [x,y] (sensor frame), `pose_est` [x,y,θ] |
| `depth_image` | `angles`, `depths`, `pose_est` |
| `trajectory` (derived) | `poses` (N,3 project frame), `t` (N), `sigma_xy` (N) |
| `pose_corrections` (derived) | `obs_ids` (N str), `pose_before` (N,3), `pose_after` (N,3), `icp_rmse` (N) |
| `occupancy_grid` (derived) | `occ` int8 (−1 unknown / 0 free / 1 occupied), `prob` float32, `hits`, `passes`, `ignorance` float32, `conflict` float32 (Dempster-Shafer masses), `x0`, `y0`, `res` |
| `coverage_grid` (derived) | `coverage` int8 (0 unobserved / 1 insufficient / 2 observed), `x0`, `y0`, `res` |
| `design_grid` (derived) | `plan` int8, `x0`, `y0`, `res` |
| `traversability_grid` (derived) | `classes` int8 (0 traversable / 1 inflated / 2 obstacle / 3 unknown), `cost` uint8 (0 / 253 / 254 lethal / 255 unknown), `x0`, `y0`, `res` |

## project.json

```json
{
  "name": "…",
  "bounds": {"x0": -0.5, "y0": -0.5, "w": 15.0, "h": 10.0},
  "control_points": {"F1": [0.4, 0.4], "…": []},
  "design": {"name": "plan rev A", "walls": [[[x1,y1],[x2,y2]], …]},
  "zones": {"Room A": [x0, y0, x1, y1]},
  "pipeline_params": {"occupancy-mapping": {"res": 0.05}}
}
```

## Dataset format (`sitestate/dataset@0.1`)

```
dataset_dir/
  dataset.json      {schema, mission, sensors: {name: {manifest, health}},
                     observations: [{t, sensor, data_type, frame, quality, payload}]}
  payloads/obs_*.npz
```
Any recorder that writes this is a supported sensor source via
`FileReplayAdapter`. `sitestate record` produces it from a stored mission.

## HTTP API (read-only, localhost)

| endpoint | returns |
|---|---|
| `/healthz`, `/version` | liveness + package/site-state version (deployment probes) |
| `/api/summary` | version, claims by kind, competing count, freshness per kind |
| `/api/missions` | mission records |
| `/api/claims?kind=&status=` | claims in the served version |
| `/api/claims/<id>` / `…/trace` | one claim / its full provenance chain |
| `/api/query?x=&y=` | occupancy, coverage, traversability, zone, freshness, confidences, claims at point, source claim ids |
| `/api/plan` | proposed next capture (targets + waypoint tour) |
| `/viewer`, `/report`, `/package.json` | interactive map, HTML report, JSON package |

## Exports

- `package.json` (`sitestate/package@0.1`): version, missions, sensors,
  claims, resolved provenance per claim.
- `costmap.npz` + `costmap.json` (`sitestate/costmap@0.1`): cost/classes
  grids, origin, resolution, confidence, source claim.
- `planning_scene.json` (`sitestate/planning-scene@0.1`): labelled
  obstacle boxes with class probabilities and status.
- `capture plan` (`sitestate/capture-plan@2.0`): targets + waypoints
  ordered by expected information gain per metre of travel; each
  waypoint carries `expected_gain_cells`.
- `scene_graph.json` (`sitestate/scene-graph@1.0`): layered site ->
  zones -> entities/assets/changes/deviations with typed edges
  (`in_zone`, `changed_since`, `deviates_from_plan`); every node keeps
  its claim id for provenance.
- `benchmark_report.json` (`sitestate/benchmark@2.0`): per-seed runs,
  mean/std summary of all measures, and confidence-calibration section
  (reliability bins + expected calibration error).
