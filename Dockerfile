# TriadRank ATS - Docker Configuration
# Multi-stage build for optimized image size

# ============================================
# Base Stage
# ============================================
ARG PYTHON_IMAGE=python:3.10-slim
ARG CUDA_IMAGE=nvidia/cuda:11.8-devel-ubuntu20.04
FROM ${PYTHON_IMAGE} AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# ============================================
# Development Stage
# ============================================
FROM base AS dev

# Install development dependencies
COPY requirements-prod.txt requirements-dev.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements-dev.txt \
    && python -m spacy download en_core_web_sm

# Copy source code
COPY . .

# Default command
CMD ["python", "-m", "pip", "install", "-e", "/app"]

# ============================================
# Production Stage
# ============================================
FROM base AS production

# Install production dependencies
COPY requirements-prod.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements-prod.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy source code
COPY . .

# Create non-root user for security
RUN groupadd -r appgroup && useradd -r -g appgroup appuser && \
    chown -R appuser:appgroup /app
USER appuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import os,urllib.request; urllib.request.urlopen('http://localhost:%s/health' % os.environ.get('API_PORT', '8000')).read()"]

# Build args for runtime configuration
ARG API_HOST=0.0.0.0
ARG API_PORT=8000
ARG MODEL_DEVICE=cpu
ARG UVICORN_WORKERS=4

# Environment variables for production
ENV API_HOST="${API_HOST}" \
    API_PORT="${API_PORT}" \
    MODEL_DEVICE="${MODEL_DEVICE}" \
    UVICORN_WORKERS="${UVICORN_WORKERS}"

# Run the API server
CMD ["sh", "-c", "uvicorn api.main:app --host ${API_HOST} --port ${API_PORT} --workers ${UVICORN_WORKERS}"]

# ============================================
# GPU Production Stage (Optional)
# ============================================
FROM ${CUDA_IMAGE} AS gpu-production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118
COPY requirements-prod.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements-prod.txt \
    && pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url ${TORCH_INDEX_URL} \
    && python -m spacy download en_core_web_sm

# Copy source code
COPY . .

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import os,urllib.request; urllib.request.urlopen('http://localhost:%s/health' % os.environ.get('API_PORT', '8000')).read()"]

# Build args for runtime configuration
ARG API_HOST=0.0.0.0
ARG API_PORT=8000
ARG MODEL_DEVICE=cuda
ARG UVICORN_WORKERS=4

# Environment variables for production
ENV API_HOST="${API_HOST}" \
    API_PORT="${API_PORT}" \
    MODEL_DEVICE="${MODEL_DEVICE}" \
    UVICORN_WORKERS="${UVICORN_WORKERS}"

# Run the API server with GPU support
CMD ["sh", "-c", "uvicorn api.main:app --host ${API_HOST} --port ${API_PORT} --workers ${UVICORN_WORKERS}"]

# ============================================
# Default Final Stage (CPU Production)
# ============================================
FROM production AS final

# ============================================
# Build Instructions
# ============================================
# Build CPU version (default):
# docker build -t triadrank-ats:cpu .
#
# Build GPU version:
# docker build -t triadrank-ats:gpu --target gpu-production .
#
# Run CPU version:
# docker run -p 8000:8000 triadrank-ats:cpu
#
# Run GPU version:
# docker run --gpus all -p 8000:8000 triadrank-ats:gpu
#
# Docker Compose example:
# version: '3.8'
# services:
#   triadrank-ats:
#     build:
#       context: .
#       target: production
#     ports:
#       - "8000:8000"
#     environment:
#       - MODEL_DEVICE=cuda
#     deploy:
#       resources:
#         reservations:
#           devices:
#             - driver: nvidia
#               count: 1
#               capabilities: [gpu]
