# State-of-the-art review (July 2026) and how v2 responds

A survey of the current research landscape for each subsystem, what v2
implements from it, and what is deliberately left as an adapter/plug-in
seam. Every recommendation is tagged: **[implemented]** in v2,
**[seam]** = the plug-in boundary exists and is documented, waiting for
the heavier model.

## 1. Trajectory estimation / SLAM back-end

**Landscape.** The consensus architecture is factor-graph optimization
(LIO-SAM lineage): odometry factors, scan-match factors, landmark and
loop-closure factors fused in one nonlinear least-squares problem, per
the [LiDAR SLAM survey (arXiv:2311.00276)](https://arxiv.org/pdf/2311.00276)
and [factor-graph LiDAR SLAM](https://www.mdpi.com/1424-8220/21/10/3445).
On the odometry front, [KISS-ICP (2023)](https://medium.com/stachnisslab/kiss-icp-in-defense-of-point-to-point-icp-accurate-and-robust-3d-point-cloud-registration-bd8a21cae3d8)
showed that plain point-to-point ICP, done carefully, is a
state-of-the-art baseline without parameter tuning, and
[Kinematic-ICP (2024)](https://arxiv.org/html/2410.10277v1) improves it
for wheeled robots by constraining the solution to feasible platform
motion. For robustness, plain Huber kernels reduce but do not remove
outlier influence; [switchable constraints](https://nikosuenderhauf.github.io/assets/papers/IROS12-switchableConstraints.pdf)
let the optimizer disable bad loop closures entirely.

**v2 response.**
- **[implemented]** The pose-refinement plug-in (v2.0.0) is now a true
  2D **pose-graph optimizer**: odometry factors between consecutive
  poses, fiducial **landmark factors** (surveyed control points),
  scan-to-map **ICP factors**, solved by Gauss–Newton with a **Huber
  robust kernel** on the scan-match factors. This is the standard
  back-end architecture, scaled to the platform's 2D representation.
- **[implemented]** ICP stays point-to-point with a voxelized reference
  (the KISS-ICP philosophy: simple, no tuning), vectorised in numpy.
- **[seam]** Switchable constraints / graduated non-convexity for loop
  closures, IMU pre-integration, and continuous-time trajectories are
  back-end upgrades behind the same `pose_corrections` claim kind — a
  new plug-in can compete with the built-in one and be judged by the
  benchmark.

## 2. Mapping representation

**Landscape.** Log-odds occupancy remains the workhorse; the research
edge adds **evidential (Dempster–Shafer) occupancy grids** that separate
*ignorance* (no evidence — epistemic) from *conflict* (contradictory
evidence — often dynamics or misregistration), e.g.
[evidential OGM learning (arXiv:2102.12718)](https://arxiv.org/pdf/2102.12718)
and [evidence-theory training data (arXiv:2405.10575)](https://arxiv.org/pdf/2405.10575),
with a direct mapping to subjective-logic opinions
([OGM fundamentals](https://www.emergentmind.com/topics/occupancy-grid-map-ogm)).
For photorealistic capture, **3D Gaussian Splatting** is displacing
photogrammetry in reality-capture workflows
([Bentley](https://blog.bentley.com/software/gaussian-splatting-digital-twin-reality-modeling/),
[heliguy survey](https://www.heliguy.com/blogs/posts/3dgs-and-reality-capture-what-it-means-for-surveying-and-digital-twins/)).

**v2 response.**
- **[implemented]** The occupancy grid now carries an **evidential
  layer**: per-cell belief masses m(occupied), m(free), plus explicit
  **ignorance** (epistemic uncertainty — never observed enough) and
  **conflict** (evidence disagrees — the signature of dynamics, moved
  objects or residual misregistration). Change claims report the
  evidential conflict of their region; the mapping claim reports mean
  ignorance/conflict. This upgrades the platform's core differentiator —
  honest uncertainty — to the current research formulation.
- **[seam]** 3DGS/NeRF reconstruction is an *evidence + output* concern:
  a `gaussian-splat-reconstruction` processing plug-in would consume
  posed imagery evidence and store a splat file as derived evidence; the
  claim/provenance machinery needs no change. Same for TSDF/voxel maps.

## 3. Change detection & progress monitoring

**Landscape.** SOTA compares as-built point clouds against 4D BIM for
element-wise progress
([digital-twin progress monitoring](https://www.sciencedirect.com/science/article/pii/S2666165923001291),
[schedule-driven point-cloud analytics](https://ascelibrary.org/doi/10.1061/9780784485224.050));
foundation models (SAM) are entering point-cloud temporal change
detection ([SAM for construction change detection](https://ascelibrary.org/doi/10.1061/9780784486436.071));
new long-term datasets exist
([iVISION-2DCD](https://arxiv.org/pdf/2607.03553)). Semantics-aided
change detection reduces false positives
([UAV semantic change detection](https://www.sciencedirect.com/science/article/abs/pii/S0926580521005082)).

**v2 response.**
- **[implemented]** A **progress-tracking plug-in**: per-zone and overall
  completion of the designed plan (fraction of observed planned elements
  actually built), the schedule-facing number the literature identifies
  as the customer deliverable. Trended across missions.
- **[implemented]** Change detection already gates on coverage and
  screens registration artifacts; v2 adds region evidential conflict as
  supporting signal.
- **[seam]** SAM-class learned change/segmentation models are exactly
  what the competing-claims mechanism was built for: register a learned
  `change`/`entity` producer beside the geometric one, let both claim,
  review + benchmark decide. The imagery evidence (posed frames) is
  already stored and linked for such models to consume.

## 4. Scan-to-BIM / deviation analysis

**Landscape.** Deep segmentation (Swin3D, PointNeXt, PTV3) leads
scan-to-BIM ([segmentation accuracy study](https://www.mdpi.com/2075-5309/15/7/1126),
[indoor scan-to-BIM framework](https://www.sciencedirect.com/science/article/pii/S2352710225008332));
automated QA flags deviations between cloud and model
([2026 workflow survey](https://ncircletech.com/blogs/unleashing-speed-in-scan-to-bim-how-ai-ml-are-transforming-point-cloud-to-bim-workflows-in-2026)).

**v2 response.** The plan-comparison plug-in implements the geometric
core (tolerance-banded as-built vs designed with coverage gating).
**[seam]** IFC import → richer `FloorPlan`, and learned element
segmentation → `entity` claims; both slot in without core changes.

## 5. Uncertainty calibration

**Landscape.** Conformal prediction is the emerging standard for
distribution-free calibrated uncertainty in robot perception
([conformal extrinsic calibration (arXiv:2501.06878)](https://arxiv.org/abs/2501.06878),
[OCULAR local conformal calibration](https://arxiv.org/abs/2605.13028),
[conformal detection intervals](https://arxiv.org/pdf/2403.07263)).
The prerequisite for any of it is *measuring* calibration.

**v2 response.**
- **[implemented]** The benchmark now measures **confidence calibration**
  of change claims: reliability bins and **expected calibration error
  (ECE)** against ground truth across seeds. A model whose 0.8 means
  "right 80% of the time" is now a testable property, not a hope.
- **[seam]** Conformal wrappers (calibrating per-plug-in confidence on a
  held-out capture set) are a natural `ProcessingPlugin` decorator once
  real-site calibration data exists.

## 6. What v2 deliberately does NOT chase

- **Learned models in-core.** The platform is numpy-only by design; every
  learned model (SAM, Swin3D, 3DGS, learned change detection) enters as a
  *plug-in* with a manifest, competing claims and benchmark scores. That
  is the plug-and-play thesis, and this review confirms the boundaries
  are drawn where the field is moving fastest.
- **3D in this release.** The 2D grid keeps every contract testable and
  benchmarkable in seconds. The 3D seams (`grid.py`, `icp.py`, pose
  payloads) are unchanged from v1 and remain the named migration path.

## Sources

- https://arxiv.org/pdf/2311.00276 · https://www.mdpi.com/1424-8220/21/10/3445
- https://medium.com/stachnisslab/kiss-icp-in-defense-of-point-to-point-icp-accurate-and-robust-3d-point-cloud-registration-bd8a21cae3d8 · https://arxiv.org/html/2410.10277v1 · https://github.com/PRBonn/kinematic-icp
- https://nikosuenderhauf.github.io/assets/papers/IROS12-switchableConstraints.pdf
- https://arxiv.org/pdf/2102.12718 · https://arxiv.org/pdf/2405.10575 · https://www.emergentmind.com/topics/occupancy-grid-map-ogm
- https://ascelibrary.org/doi/10.1061/9780784486436.071 · https://www.sciencedirect.com/science/article/pii/S2666165923001291 · https://ascelibrary.org/doi/10.1061/9780784485224.050 · https://arxiv.org/pdf/2607.03553 · https://www.sciencedirect.com/science/article/abs/pii/S0926580521005082
- https://www.mdpi.com/2075-5309/15/7/1126 · https://www.sciencedirect.com/science/article/pii/S2352710225008332 · https://ncircletech.com/blogs/unleashing-speed-in-scan-to-bim-how-ai-ml-are-transforming-point-cloud-to-bim-workflows-in-2026
- https://arxiv.org/abs/2501.06878 · https://arxiv.org/abs/2605.13028 · https://arxiv.org/pdf/2403.07263
- https://blog.bentley.com/software/gaussian-splatting-digital-twin-reality-modeling/ · https://www.heliguy.com/blogs/posts/3dgs-and-reality-capture-what-it-means-for-surveying-and-digital-twins/
