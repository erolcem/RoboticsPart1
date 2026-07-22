"""Command-line interface.  Entry point: `sitestate` (see pyproject.toml).

    sitestate demo    [--out DIR]
    sitestate init     --project DIR [--template demo|empty]
    sitestate missions --project DIR
    sitestate process  --project DIR MISSION_ID [--baseline MISSION_ID]
    sitestate claims   --project DIR [--kind K] [--status S]
    sitestate trace    --project DIR CLAIM_ID
    sitestate query    --project DIR X Y
    sitestate review   --project DIR list
    sitestate review   --project DIR accept|reject|recapture CLAIM_ID [--note ..]
    sitestate export   --project DIR --format html|json|costmap --out DIR
    sitestate record   --project DIR MISSION_ID --out DIR
    sitestate plan-capture --project DIR [--out FILE]
    sitestate benchmark [--seeds N] [--out DIR]
    sitestate serve    --project DIR [--port N]

`--project` points at the ledger directory (contains ledger.sqlite), e.g.
demo_output/project_data after running the demo.
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys
from pathlib import Path

from .ledger.ledger import ObservationLedger
from .review import ReviewQueue


def _ledger(args) -> ObservationLedger:
    root = Path(args.project)
    if not (root / "ledger.sqlite").exists():
        sys.exit(f"no ledger.sqlite in {root} - is this a project data directory?")
    return ObservationLedger(root)


def _latest_version(ledger: ObservationLedger, version_id: str = "") -> dict:
    if version_id:
        v = ledger.version(version_id)
        if v is None:
            sys.exit(f"unknown version {version_id}")
        return v
    versions = ledger.versions()
    if not versions:
        sys.exit("no committed site-state version in this project")
    return versions[-1]


def cmd_demo(args) -> None:
    from .demo import run_full_demo

    run_full_demo(args.out)


def cmd_init(args) -> None:
    root = Path(args.project)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "project.json").exists():
        sys.exit(f"{root}/project.json already exists - refusing to overwrite")
    if args.template == "demo":
        from .demo import PROJECT

        project = PROJECT
    else:
        project = {
            "name": root.name,
            "bounds": {"x0": 0.0, "y0": 0.0, "w": 20.0, "h": 20.0},
            "control_points": {},
            "zones": {},
        }
    (root / "project.json").write_text(json.dumps(project, indent=2))
    ObservationLedger(root)  # creates ledger.sqlite + evidence dir
    print(f"initialised project '{project['name']}' in {root}")
    if not project.get("control_points"):
        print("note: add surveyed control_points to project.json before capturing -")
        print("      registration refuses to guess without them")


def cmd_process(args) -> None:
    from .platform import load_platform

    platform = load_platform(Path(args.project))
    activities = platform.process_all(
        args.mission_id, baseline_mission_id=args.baseline or ""
    )
    for act in activities:
        print(f"{act.plugin:<28s} {act.status:<10s} {'; '.join(act.notes)}")


def cmd_plan_capture(args) -> None:
    from .planning import propose_capture_plan

    plan = propose_capture_plan(_ledger(args), args.version)
    text = json.dumps(plan, indent=2)
    if args.out:
        Path(args.out).write_text(text)
        print(f"{plan['n_targets']} target(s), {len(plan['waypoints'])} waypoint(s) "
              f"-> {args.out}")
    else:
        print(text)


def cmd_benchmark(args) -> None:
    from .benchmark import run_benchmark

    run_benchmark(seeds=args.seeds, out_dir=args.out or None)


def cmd_missions(args) -> None:
    for m in _ledger(args).missions():
        print(f"{m['id']}  {m['name']:<20s} {m.get('started_at','')}  "
              f"carrier={m['carrier'].get('type','?')}  sensors={len(m.get('sensor_ids',[]))}")


def cmd_claims(args) -> None:
    for c in _ledger(args).claims(kind=args.kind, status=args.status):
        subject = f"  [{c['subject']}]" if c.get("subject") else ""
        print(f"{c['id']}  {c['kind']:<20s} {c['status']:<10s} "
              f"conf={c['confidence']:.2f}  mission={c['mission_id']}{subject}")


def cmd_trace(args) -> None:
    print(json.dumps(_ledger(args).trace(args.claim_id), indent=2))


def cmd_query(args) -> None:
    from .query import SiteStateQuery

    ledger = _ledger(args)
    q = SiteStateQuery(ledger, args.version or _latest_version(ledger)["id"])
    print(json.dumps(q.at_point(args.x, args.y), indent=2))


def cmd_review(args) -> None:
    queue = ReviewQueue(_ledger(args))
    reviewer = args.reviewer or getpass.getuser()
    if args.action == "list":
        pending = queue.pending()
        if not pending:
            print("nothing pending review")
        for c in pending:
            print(f"{c['id']}  {c['kind']:<12s} {c['status']:<10s} "
                  f"conf={c['confidence']:.2f}  {c.get('subject','')}")
        for r in queue.recapture_requests():
            print(f"recapture requested: claim {r['claim_id']} region={r.get('region')} "
                  f"({r['reviewer']}: {r['note']})")
        return
    if not args.claim_id:
        sys.exit("claim id required for accept/reject/recapture")
    if args.action == "accept":
        queue.accept(args.claim_id, reviewer, args.note)
    elif args.action == "reject":
        queue.reject(args.claim_id, reviewer, args.note)
    elif args.action == "recapture":
        queue.request_recapture(args.claim_id, reviewer, args.note)
    print(f"{args.action}: {args.claim_id} by {reviewer}")


def cmd_export(args) -> None:
    from .outputs import HtmlReport, JsonPackageExport
    from .outputs.costmap_export import RobotCostmapExport
    from .outputs.scenegraph_export import SceneGraphExport

    adapters = {
        "html": HtmlReport,
        "json": JsonPackageExport,
        "costmap": RobotCostmapExport,
        "scenegraph": SceneGraphExport,
    }
    ledger = _ledger(args)
    version = _latest_version(ledger, args.version)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = adapters[args.format]().render(ledger, version, out)
    print("\n".join(str(f) for f in files))


def cmd_record(args) -> None:
    from .ingest import export_mission_dataset

    path = export_mission_dataset(_ledger(args), args.mission_id, args.out)
    print(f"dataset written to {path}")


def cmd_serve(args) -> None:
    from .serve import serve

    serve(args.project, port=args.port, version_id=args.version, host=args.host)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="sitestate", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def with_project(sp):
        sp.add_argument("--project", required=True, help="project data dir (has ledger.sqlite)")
        return sp

    sp = sub.add_parser("demo", help="run the built-in two-mission demo")
    sp.add_argument("--out", default="demo_output")
    sp.set_defaults(fn=cmd_demo)

    sp = with_project(sub.add_parser("init", help="initialise a new project directory"))
    sp.add_argument("--template", choices=["demo", "empty"], default="empty")
    sp.set_defaults(fn=cmd_init)

    sp = with_project(sub.add_parser("process", help="run the full plug-in pipeline"))
    sp.add_argument("mission_id")
    sp.add_argument("--baseline", default="")
    sp.set_defaults(fn=cmd_process)

    sp = with_project(sub.add_parser("plan-capture",
                                     help="propose waypoints for the next capture"))
    sp.add_argument("--out", default="")
    sp.add_argument("--version", default="")
    sp.set_defaults(fn=cmd_plan_capture)

    sp = sub.add_parser("benchmark", help="score the pipeline against sim ground truth")
    sp.add_argument("--seeds", type=int, default=3)
    sp.add_argument("--out", default="")
    sp.set_defaults(fn=cmd_benchmark)

    with_project(sub.add_parser("missions", help="list missions")).set_defaults(fn=cmd_missions)

    sp = with_project(sub.add_parser("claims", help="list claims"))
    sp.add_argument("--kind")
    sp.add_argument("--status")
    sp.set_defaults(fn=cmd_claims)

    sp = with_project(sub.add_parser("trace", help="provenance of a claim"))
    sp.add_argument("claim_id")
    sp.set_defaults(fn=cmd_trace)

    sp = with_project(sub.add_parser("query", help="site-state belief at a point"))
    sp.add_argument("x", type=float)
    sp.add_argument("y", type=float)
    sp.add_argument("--version", default="")
    sp.set_defaults(fn=cmd_query)

    sp = with_project(sub.add_parser("review", help="human review of claims"))
    sp.add_argument("action", choices=["list", "accept", "reject", "recapture"])
    sp.add_argument("claim_id", nargs="?", default="")
    sp.add_argument("--note", default="")
    sp.add_argument("--reviewer", default="")
    sp.set_defaults(fn=cmd_review)

    sp = with_project(sub.add_parser("export", help="export a version"))
    sp.add_argument("--format", choices=["html", "json", "costmap", "scenegraph"], required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--version", default="")
    sp.set_defaults(fn=cmd_export)

    sp = with_project(sub.add_parser("record", help="dump a mission as a portable dataset"))
    sp.add_argument("mission_id")
    sp.add_argument("--out", required=True)
    sp.set_defaults(fn=cmd_record)

    sp = with_project(sub.add_parser("serve", help="read-only HTTP API + viewer"))
    sp.add_argument("--port", type=int, default=8752)
    sp.add_argument("--host", default="127.0.0.1",
                    help="bind address; 0.0.0.0 for containers (no auth - gateway required)")
    sp.add_argument("--version", default="")
    sp.set_defaults(fn=cmd_serve)

    args = p.parse_args(argv)
    try:
        args.fn(args)
    except (KeyError, RuntimeError) as exc:
        # domain errors (unknown claim/version id, no committed version, ...)
        # should read as messages, not tracebacks
        sys.exit(f"error: {exc}")
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
