# TriadRank ATS - 3-Tier Resume Scoring and Ranking System

A complete Applicant Tracking System (ATS) with a 3-tier architecture for intelligent resume scoring and ranking. This project has been customized to work with the provided resume-job datasets.

## Quickstart (Inference)

### 1) Install

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2) Run the interactive ranker (recommended)

`run.py` is an app-style CLI that walks you through selecting:
- resume folder (PDFs scanned recursively)
- job description `.txt`
- job category (from the 24 configured categories)
- local vs API mode
- `pipeline_top_k` and `top_k`
- output format (CSV/JSON)

```bash
python run.py
```

Notes:
- For the included dataset, PDF resumes live at: `datasets/category resumes/data/data/<CATEGORY>/`
- Job descriptions can be stored in `jds/<CATEGORY>.txt`.

## System Architecture

### Model 1: Cross-Encoder (Deep Ranking)
- **Purpose**: Final scoring and ranking of resume-job pairs
- **Input**: Resume text with scores and labels
- **Output**: Regression score (0-1) and classification (good/potential/bad fit)
- **Architecture**: BERT-based transformer with multi-head output

### Model 2: Category Encoder (Gatekeeper)
- **Purpose**: Classify resumes into job categories
- **Input**: Resume text
- **Output**: Category classification (24 job categories)
- **Architecture**: DistilBERT-based classifier

### Model 3: Extractor & Miner
- **Purpose**: Initial retrieval and hard negative mining
- **Input**: Resume with multiple sections (skills, experience, education)
- **Output**: Extracted entities and structured data
- **Architecture**: spaCy NER with custom training

## Project Structure

```
ATS_Project_Clean/
├── config.yaml              # Main configuration file (UPDATED with dataset paths)
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── data/
│   ├── loaders.py          # Data loading and preprocessing (CUSTOMIZED)
│   ├── raw/                # Raw data directory
│   ├── processed/          # Processed datasets (.pt files)
│   └── cache/              # Cache files
├── models/
│   ├── __init__.py
│   ├── cross_encoder.py    # Cross-Encoder model
│   ├── category_encoder.py # Category Encoder model
│   └── extractor.py        # Entity Extractor model
├── scripts/
│   ├── __init__.py
│   ├── train_cross_encoder.py  # UPDATED for new data loaders
│   ├── train_category.py       # UPDATED for new data loaders
│   └── train_extractor.py      # UPDATED for new data loaders
├── datasets/               # User datasets (INCLUDED)
│   ├── score dataset/
│   │   ├── train.csv       # Training data for Cross-Encoder
│   │   └── validation.csv  # Validation data for Cross-Encoder
│   ├── category resumes/
│   │   └── Resume/
│   │       └── Resume.csv  # Training data for Category Encoder
│   └── bi encoder retrieval dataset/
│       └── resume_data_for_ranking.csv  # Training data for Extractor
├── checkpoints/            # Model checkpoints (created during training)
│   ├── cross_encoder/
│   ├── category_encoder/
│   └── extractor/
├── logs/                   # Training logs
├── jds/                    # Job descriptions used for evaluation/inference (.txt)
├── reports/                # Evaluation and inference outputs (.json/.csv)
└── api/                    # FastAPI application
    └── main.py
```

## Dataset Schemas

### Score Dataset (Cross-Encoder)
**Location**: `datasets/score dataset/`

| Column | Type | Description |
|--------|------|-------------|
| text | string | Resume text content |
| ats_score | float | ATS score (0-1 or 0-100) |
| original_label | string | Label (good_fit, potential_fit, bad_fit) |

### Category Dataset (Category Encoder)
**Location**: `datasets/category resumes/Resume/`

| Column | Type | Description |
|--------|------|-------------|
| ID | string | Resume identifier |
| Resume_str | string | Resume text content |
| Resume_html | string | Resume HTML content |
| Category | string | Job category (24 classes) |

### Retrieval Dataset (Extractor)
**Location**: `datasets/bi encoder retrieval dataset/`

| Column | Type | Description |
|--------|------|-------------|
| ID | string | Resume identifier |
| career_objective | string | Career objective text |
| skills | string | Skills section |
| work_experience | string | Work experience text |
| education | string | Education section |
| positions | string | Job positions |
| companies | string | Company names |
| project_details | string | Project descriptions |

### PDF Resume Data (Real-World Processing)
**Location**: `datasets/category resumes/data/data/`

The project includes real PDF resumes organized by job category for realistic training scenarios.

**Directory Structure**:
```
datasets/category resumes/data/data/
├── ACCOUNTANT/
│   ├── 10089434.pdf
│   ├── 10247517.pdf
│   └── ...
├── ADVOCATE/
│   ├── 10123456.pdf
│   └── ...
├── AGRICULTURE/
│   └── ...
├── ...
└── TEACHER/
    └── ...
```

**Features**:
- 24 job categories with real PDF resumes
- Supports real-world resume processing pipeline
- Text extraction using pypdf and pdfplumber
- Automatic text caching for faster subsequent processing

**Supported Categories**:
ACCOUNTANT, ADVOCATE, AGRICULTURE, APPAREL, ARTS, AUTOMOBILE, AVIATION, BANKING, BPO, BUSINESS-DEVELOPMENT, CHEF, CONSTRUCTION, CONSULTANT, DESIGNER, DIGITAL-MEDIA, ENGINEERING, FINANCE, FITNESS, HEALTHCARE, HR, INFORMATION-TECHNOLOGY, PUBLIC-RELATIONS, SALES, TEACHER

source ats_env/bin/activate  # Linux/Mac
# or
.\ats_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

## Usage

### Run the API (optional)

Start the FastAPI server:

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Useful endpoints:
- `GET /health`
- `GET /categories`
- `POST /rank`

If you run `run.py` in `api` mode, it calls `http://127.0.0.1:8000/rank`.

### Inference (interactive)

```bash
python run.py
```

The output includes `file_path` for each ranked PDF, plus model scores and category match signals.

Important parameters:
- `pipeline_top_k`: how many resumes are processed internally (Tier-3 selection size)
- `top_k`: how many results you want written to the output

### Training Model 1: Cross-Encoder
```bash
cd scripts
python train_cross_encoder.py \
    --train_path ../datasets/score dataset/train.csv \
    --val_path ../datasets/score dataset/validation.csv \
    --epochs 20 \
    --batch_size 16 \
    --learning_rate 2e-5
```

Or using config defaults:
```bash
cd scripts
python train_cross_encoder.py
```

### Training Model 2: Category Encoder
```bash
cd scripts
python train_category.py \
    --csv_path ../datasets/category resumes/Resume/Resume.csv \
    --pdf_dir ../datasets/category resumes/data/data \
    --epochs 10 \
    --batch_size 32 \
    --learning_rate 3e-5
```

**PDF-Only Training** (using real PDF resumes):
```bash
cd scripts
python train_category.py \
    --pdf_dir ../datasets/category resumes/data/data \
    --epochs 10 \
    --batch_size 32
```

**CSV-Only Training** (using extracted text):
```bash
cd scripts
python train_category.py \
    --csv_path ../datasets/category resumes/Resume/Resume.csv \
    --epochs 10 \
    --batch_size 32
```

Or using config defaults:
```bash
cd scripts
python train_category.py
```

### Training Model 3: Extractor
```bash
cd scripts
python train_extractor.py \
    --data_path ../datasets/bi encoder retrieval dataset/resume_data_for_ranking.csv \
    --epochs 50 \
    --batch_size 32
```

Or using config defaults:
```bash
cd scripts
python train_extractor.py
```

## Evaluation

### PDF ranking evaluation (local or API)

The script `scripts/eval_rank_pdfs.py` evaluates ranking over PDF resumes. It supports:
- **single** run (one job category vs one distractor category)
- **matrix** mode across multiple categories with near/far distractors (`--matrix`)
- local pipeline or API (`--use_api`)

Example (API matrix):

```bash
python scripts/eval_rank_pdfs.py --matrix --use_api --api_url "http://127.0.0.1:8000/rank"
```

Outputs are written under `reports/`:
- `rank_eval_matrix_pdfs_<timestamp>.json`
- `rank_eval_matrix_pdfs_<timestamp>.csv`

The CSV includes additional ranking-quality metrics:
- `top10_job_precision` (Top-10 category correctness)
- `pairwise_accuracy` (how often job resumes score above distractors)
- `ndcg` and `ndcg_at_10` (ranking sharpness using model-predicted relevance)

Score-gap reporting note:
- If no distractors appear in the returned Top-N, distractor mean and gap are left blank (not forced to 0.0).

## Configuration

All settings are managed via `config.yaml`. The data paths have been pre-configured:

```yaml
data:
  raw:
    cross_encoder:
      train: "datasets/score dataset/train.csv"
      validation: "datasets/score dataset/validation.csv"
    category_resumes: "datasets/category resumes/Resume/Resume.csv"
    extraction_pairs: "datasets/bi encoder retrieval dataset/resume_data_for_ranking.csv"
```

Key parameters you can modify:
- Model hyperparameters (learning rate, epochs, batch size)
- Evaluation thresholds
- Penalty weights for ranking
- Logging settings

## Data Cleaning and Preprocessing

The data loaders include comprehensive cleaning functions:

### Text Cleaning
- HTML tag removal
- URL and email removal
- Phone number normalization
- Whitespace normalization
- Special character handling

### Resume-Specific Cleaning
- Common abbreviation normalization (yrs → years, exp → experience)
- Skills extraction using pattern matching
- Score normalization (0-100 to 0-1)

### Category Validation
- Category name normalization
- Missing category handling

## Evaluation Metrics

### Cross-Encoder
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Accuracy
- F1 Score (Macro/Weighted)

### Category Encoder
- Accuracy
- F1 Score (Macro/Weighted)
- Confusion Matrix

### Extractor
- Precision
- Recall
- F1 Score
- Entity Accuracy

## Inference Pipeline

The system processes resumes through all three tiers:

1. **Tier 3 (Extractor)**: Retrieve top-K candidates based on skills and experience
2. **Tier 2 (Category Encoder)**: Verify category match, penalize mismatches
3. **Tier 1 (Cross-Encoder)**: Final scoring and ranking

## Dependencies

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.30+
- spaCy 3.6+
- scikit-learn 1.3+
- pandas 2.0+
- numpy 1.24+
- FastAPI 0.100+

## Quick Test

Test data loaders without training:
```bash
cd data
python loaders.py
```

This will verify all datasets load correctly and print statistics.

## Project Files Modified

The following files have been customized for your datasets:

1. **config.yaml** - Updated data paths and dataset schemas
2. **data/loaders.py** - Complete rewrite with data cleaning and new dataset classes
3. **scripts/train_cross_encoder.py** - Updated to use new CrossEncoderDataset
4. **scripts/train_category.py** - Updated to use new CategoryDataset
5. **scripts/train_extractor.py** - Updated to use new ExtractorDataset

## License

MIT License
