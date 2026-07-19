# TriadRank ATS — CPU inference server
#
# Single-stage build. Trained checkpoints (cross-encoder + category encoder) are
# NOT baked into the image; they are fetched at runtime from HuggingFace Hub into
# a persistent HF_HOME volume (see docker-compose.yml + CHECKPOINT_REPO_* env).
# The same cache holds the base transformers tokenizers/models used at startup.
#
# Fail-fast contract (pipeline/inference.py:_load_checkpoint):
#   - missing checkpoint + repo env set   → fetch from HF, load trained weights
#   - missing checkpoint + no repo env   → RAISE (container exits, restarts)
#   - ALLOW_RANDOM_FALLBACK=1            → random untrained weights, /health → 503
# So a broken container can no longer silently serve noise. See backend-verified-usable.

FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache \
    HF_HUB_DOWNLOAD_TIMEOUT=30

# System deps: build-essential for C-extension wheels (pyahocorasick), git for
# huggingface_hub parity. Kept in-image; slim size penalty only.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU torch FIRST so requirements-prod's `torch>=2.0.0` resolves to this already-
# installed CPU wheel instead of pulling the CUDA-bundled one from PyPI (~600MB saved).
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

COPY requirements-prod.txt .
RUN pip install -r requirements-prod.txt \
    && python -m spacy download en_core_web_sm

# App source: models/ (NOTE: the .pt weights are git/docker-ignored; this brings
# the Python package only), pipeline/, api/, config.yaml, data/skill_db.json.
COPY . .

# Fail-loud guard. Tier-3's skill trie (models/skill_trie.py) loads data/skill_db.json.
# If it's absent the extractor degrades to skill_overlap=0 for every candidate — a
# *silent* ranking signal loss, not a crash. A fresh-clone build that forgot to
# regenerate it stops here with a clear message, not in production.
RUN test -f data/skill_db.json \
    || (echo "data/skill_db.json missing — run: python scripts/build_skill_db.py" && exit 1)

# Non-root runtime user. Owns /app and the HF cache volume mount-point.
RUN groupadd -r appgroup && useradd -r -g appgroup appuser \
    && mkdir -p /opt/hf-cache && chown -R appuser:appgroup /app /opt/hf-cache
USER appuser

EXPOSE 8000

# Runtime config (override per `docker run -e` or compose). MODEL_DEVICE is now
# honored by api/main.py lifespan (previously a no-op in config.py).
ENV API_HOST=0.0.0.0 \
    API_PORT=8000 \
    MODEL_DEVICE=cpu \
    UVICORN_WORKERS=1

# /health now returns 503 when the pipeline is degraded (not ready, or running
# random fallback weights), so this healthcheck actually notices a broken server.
# start-period=120s grants time for first-start HF downloads (~1.4GB into the volume).
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://localhost:%s/health' % os.environ.get('API_PORT','8000'), timeout=5).read()" || exit 1

CMD ["sh", "-c", "uvicorn api.main:app --host ${API_HOST} --port ${API_PORT} --workers ${UVICORN_WORKERS}"]
