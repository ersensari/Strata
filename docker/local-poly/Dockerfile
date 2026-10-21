# Homelab runtime: verified local binaries plus this fork's original app host.
# Prepare artifacts as documented in README.md before building.
FROM node:24.14.0-bookworm-slim AS node
FROM forgeloop-ai-polystrata:0.1.45-poly.2
LABEL forgeloop.managed="true" forgeloop.project="forgeloop-ai" forgeloop.resource="llm-image"
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY serve /app/source/serve
COPY tools /app/source/tools
COPY strata_control /app/source/strata_control
COPY strata_cli /app/source/strata_cli
COPY data /app/source/data
COPY setup.py strata.py CMakeLists.txt /app/source/
COPY apps/host/server /app/host/server
COPY apps/host/dist /app/host/dist
COPY apps/host/node_modules /app/host/node_modules
COPY apps/host/package.json /app/host/package.json
COPY docker/local-poly/artifacts/ /app/engines/
COPY docker/local-poly/start.py /app/start.py
ENV PATH="/opt/strata/.venv/bin:/usr/local/bin:/usr/local/cuda/bin:/usr/bin:/bin" \
    STRATA_HOME=/data/ui STRATA_WORKSPACE=/data/workspace \
    STRATA_NATIVE_REPO=/data/install STRATA_PYTHON=/opt/strata/.venv/bin/python \
    STRATA_REAL_ENGINE=1 STRATA_NO_ACQUIRE=1 STRATA_NO_BUNDLED_ENGINE=1 \
    STRATA_BIND=0.0.0.0 STRATA_PORT=8791 STRATA_APP_VERSION=0.2.2+poly.c7a2fbd7
WORKDIR /app/host
ENTRYPOINT ["/opt/strata/.venv/bin/python", "/app/start.py"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=5m --retries=3 CMD ["/opt/strata/.venv/bin/python", "-c", "import json,urllib.request;d=json.load(urllib.request.urlopen('http://127.0.0.1:8791/api/native',timeout=4));assert any(i['status']=='ready' for i in d.get('instances',[]))"]
