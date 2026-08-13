# The deployable unit.
#
# Two stages. The first installs the ingestion dependencies and builds the
# policy index; the second copies that index into a runtime image that has no
# ZenML in it at all. The separation the architecture claims is therefore also
# true of the artefact: the thing that serves requests cannot run ingestion,
# and cannot write to the index it reads.

FROM python:3.11-slim AS builder

WORKDIR /build

ENV PIP_NO_CACHE_DIR=1 \
    POLICY_REVIEW_RUN_MODE=mock \
    ZENML_ANALYTICS_OPT_IN=false

COPY pyproject.toml ./
COPY config/ ./config/
COPY shared/ ./shared/
COPY agent/ ./agent/
COPY ingestion/ ./ingestion/
COPY cli/ ./cli/
COPY policy_corpus/ ./policy_corpus/

RUN pip install --no-cache-dir -e ".[ingestion]"

# Build the index at image build time. If verification fails or retrieval
# quality has regressed, this layer fails and no image is produced.
RUN python -m cli.ingest --local


FROM python:3.11-slim AS runtime

# Not root. The service reads an index and writes checkpoints; it needs nothing
# else on the filesystem.
RUN useradd --create-home --uid 10001 policy
WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POLICY_REVIEW_RUN_MODE=mock

COPY pyproject.toml ./
COPY config/ ./config/
COPY shared/ ./shared/
COPY agent/ ./agent/
COPY cli/ ./cli/
COPY webapp/ ./webapp/
COPY policy_corpus/ ./policy_corpus/

# The evaluated index from the builder stage. Note what is absent: ingestion/
# is not copied, so this image physically cannot rebuild or modify it.
COPY --from=builder /build/.policy_index/ ./.policy_index/

RUN pip install --no-cache-dir -e ".[web]" \
    && mkdir -p /app/.policy_state \
    && chown -R policy:policy /app

USER policy
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/status', timeout=3).status==200 else 1)"

CMD ["python", "-m", "webapp", "--host", "0.0.0.0", "--port", "8000"]
