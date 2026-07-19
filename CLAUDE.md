# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Environment Setup
```bash
# Create and activate a virtual environment (recommended)
python -m venv ats_env
# Windows
.\\ats_env\\Scripts\\activate
# Unix/macOS
source ats_env/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Download the spaCy English model (used by the extractor)
python -m spacy download en_core_web_sm
```

### Interactive Ranking (CLI)
```bash
# Run the interactive ranker. It will prompt for:
#   - resume folder (PDFs are scanned recursively)
#   - job description *.txt
#   - job category (one of 24 configured categories)
#   - local vs API mode
#   - pipeline_top_k and top_k
#   - output format (CSV/JSON)
python run.py
```

### FastAPI Service
```bash
# Start the API server (useful for programmatic access)
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

*Health check*: `GET /health`
*List categories*: `GET /categories`
*Rank candidates*: `POST /rank` (JSON body, see `api/main.py` for request schema)
*Single‑candidate rank*: `POST /rank/single`
*Entity extraction*: `POST /extract`

### Model Training
```bash
# Cross‑Encoder (Tier‑1) training
cd scripts
python train_cross_encoder.py  # uses config defaults
# Category Encoder (Tier‑2) training
python train_category.py
# Extractor / Hard‑Negative Miner (Tier‑3) training
python train_extractor.py
```

All scripts accept explicit `--train_path`, `--val_path`, `--epochs`, `--batch_size`, etc. Running them without arguments falls back to the paths defined in `config.yaml`.

### Evaluation
```bash
# Evaluate ranking on PDF resumes (local pipeline or API mode)
python scripts/eval_rank_pdfs.py                # single‑run
python scripts/eval_rank_pdfs.py --matrix       # matrix across categories
python scripts/eval_rank_pdfs.py --use_api      # evaluate against the running API
```

### Lint / Formatting (optional)
```bash
# Basic linting (uses flake8 if installed)
flake8 .
# Code formatting
black .
```

### Running a Single Test (data loader sanity check)
```bash
cd data
python loaders.py   # prints dataset stats and validates loading
```

## High‑Level Architecture

- **Three‑Tier Scoring Pipeline** (`pipeline/inference.py`)
  1. **Tier 3 – Extraction** (`models/extractor.py`)
     - spaCy‑based NER extracts skills, experience, education, etc.
     - Computes cheap keyword Jaccard overlap and optional skill‑overlap score.
  2. **Tier 2 – Category Validation** (`models/category_encoder.py`)
     - DistilBERT classifier predicts the resume’s job category.
     - A configurable penalty (`config.yaml → penalties → category_mismatch`) de‑weights scores when the predicted category mismatches the target.
  3. **Tier 1 – Deep Scoring** (`models/cross_encoder.py`)
     - BERT‑based cross‑encoder produces a regression score (0‑1) and a three‑class label (Good Fit / Potential Fit / Bad Fit).

- **Configuration** (`config.yaml`)
  - Centralizes dataset paths, model checkpoint locations, inference defaults (`device`, `top_k`, batch sizes), and penalty constants.
  - Category ID ↔ name mappings live under `categories` and are also exposed via the API.

- **API Layer** (`api/main.py`)
  - FastAPI app creates a global `TriadRankPipeline` instance on startup.
  - Endpoints wrap the pipeline methods (`rank_candidates`, `rank_single`, `extract_entities`).
  - Health endpoint reports pipeline readiness and model load status.

- **Data Layer** (`data/loaders.py`)
  - Handles CSV and PDF‑derived datasets, performs cleaning (HTML tag stripping, URL/email removal, whitespace normalization) and provides helper functions for batch tokenisation.

- **Model Packages** (`models/`)
  - `cross_encoder.py` – multi‑head transformer for final scoring.
  - `category_encoder.py` – DistilBERT classifier with checkpoint loading.
  - `extractor.py` – spaCy NER wrapper and hard‑negative mining utilities.

- **Scripts** (`scripts/`)
  - Training scripts for each tier, evaluation script for PDF ranking, and a simple interactive CLI (`run.py`).

- **Checkpoints & Logs** (`checkpoints/`, `logs/`)
  - Model checkpoints are saved here during training; logs capture training metrics.

- **Datasets** (`datasets/`)
  - Structured CSV datasets for each tier and a real‑world PDF resume collection organized by the 24 job categories.

The pipeline flows **Extractor → Category Encoder → Cross‑Encoder**, with each tier optionally configurable via `config.yaml`. The API and CLI share the same pipeline instance, ensuring consistent behavior across interactive and service‑based use.

---

*When adding new functionality, follow the three‑tier pattern: place model‑specific code under `models/`, expose any new CLI flags in the relevant `scripts/` file, and add API endpoints only if external consumption is required.*