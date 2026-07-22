# Deployment guide

## Local (bare metal / venv)

```bash
pip install -e .          # or: pip install .   (only dependency: numpy)
sitestate init --project /srv/sitestate/projectX      # then edit project.json
sitestate serve --project /srv/sitestate/projectX     # binds 127.0.0.1:8752
```

`sitestate serve` is a read-only, no-auth server by design (the
security/ownership NFR says data access must be *explicit*): bind it to
localhost and put your gateway (nginx/Caddy with auth, or a VPN) in
front. `--host 0.0.0.0` is available for containers and prints a warning.

Probes: `GET /healthz` (liveness + served site-state version),
`GET /version` (package + data versions).

## Docker

```bash
docker build -t sitestate .
# generate demo data into a volume, then serve it:
docker run --rm -v sitestate-data:/data sitestate demo --out /data/demo
docker run -d -p 8752:8752 -v sitestate-data:/data \
    sitestate serve --project /data/demo/project_data --host 0.0.0.0
curl http://127.0.0.1:8752/healthz
```

Or with compose (demo + server, one command):

```bash
docker compose up --build
# -> http://127.0.0.1:8752/viewer
```

The image runs as a non-root user, declares a `HEALTHCHECK` against
`/healthz`, and stores all project data under the `/data` volume — one
volume = one deployment's evidence, which matches the "one directory =
one project = archivable package" storage design.

Verified in this repo's CI (`.github/workflows/ci.yml`): image builds,
demo generates a project inside the container, server answers `/healthz`.

## CI

`.github/workflows/ci.yml` runs on every push/PR:
- pytest across Python 3.10 and 3.12 (33 end-to-end tests),
- a 1-seed benchmark smoke run (thresholds are asserted in the tests),
- Docker build + container smoke test.

## Operational notes

- **Data layout**: everything lives under the project directory
  (`ledger.sqlite`, `evidence/`, `project.json`, `exports/`). Back up the
  directory; it is the whole deployment state.
- **Ingestion**: real rigs write the dataset format
  (docs/data-reference.md) to disk/object storage; an operator (or cron)
  runs `sitestate` with `FileReplayAdapter`-based ingestion, then
  `sitestate process`, `commit`, and the server picks the new version on
  restart (versions are immutable snapshots).
- **Scaling**: the ledger interface is the swap point for
  Postgres/object-storage when multi-user concurrency arrives; the HTTP
  server is read-only so replicas can serve the same directory.
- **Upgrades**: plug-in re-runs supersede rather than overwrite, so
  upgrading the package and re-running `sitestate process` on stored
  missions is safe and auditable; run `sitestate benchmark` before and
  after to verify the upgrade actually improved the measures.
