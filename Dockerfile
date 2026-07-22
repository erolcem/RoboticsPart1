# Site State Platform - deployable image.
#
#   docker build -t sitestate .
#   docker run --rm sitestate demo --out /data/demo          # generate demo data
#   docker run --rm -p 8752:8752 -v sitestate-data:/data \
#       sitestate serve --project /data/demo/project_data --host 0.0.0.0
#
# The server has no authentication: keep it behind a gateway.

FROM python:3.12-slim AS runtime

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# non-root user; /data is the project-data mount point
RUN useradd --create-home sitestate && mkdir /data && chown sitestate:sitestate /data
USER sitestate
VOLUME ["/data"]
EXPOSE 8752

HEALTHCHECK --interval=30s --timeout=3s CMD \
  python -c "import urllib.request,sys; \
  sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8752/healthz', timeout=2).status==200 else 1)" \
  || exit 1

ENTRYPOINT ["sitestate"]
CMD ["--help"]
