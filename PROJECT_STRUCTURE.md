"""
TriadRank ATS - Project Structure and File Summary
Generated: 2024
"""

PROJECT_STRUCTURE = """
triadrank-ats/
├── README.md                          # Main project documentation
├── config.yaml                        # Complete configuration file
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Docker configuration
│
├── config/
│   └── config.yaml                    # (Duplicate reference)
│
├── data/
│   ├── __init__.py                    # Data package initialization
│   ├── loaders.py                     # Data loading utilities
│   ├── raw/                           # Raw training data
│   │   ├── resume_job_pairs.csv       # Resume-job pairs with scores
│   │   ├── category_resumes.csv       # Resumes with category labels
│   │   └── extraction_pairs.csv       # Annotated resumes for NER
│   ├── processed/                     # Processed datasets
│   └── cache/                         # Cache directory
│
├── models/
│   ├── __init__.py                    # Models package initialization
│   ├── cross_encoder.py               # Model 1: Cross-Encoder (Deep Ranking)
│   ├── category_encoder.py            # Model 2: Category Encoder (Gatekeeper)
│   └── extractor.py                   # Model 3: Entity Extractor (Miner)
│
├── pipeline/
│   ├── __init__.py                    # Pipeline package initialization
│   └── inference.py                   # Main inference pipeline
│
├── api/
│   ├── main.py                        # FastAPI application
│   └── schemas.py                     # API request/response schemas
│
├── scripts/
│   ├── train_cross_encoder.py         # Training script for Model 1
│   ├── train_category.py              # Training script for Model 2
│   └── train_extractor.py             # Training script for Model 3
│
├── tests/
│   ├── test_models.py                 # Unit tests for models
│   ├── test_pipeline.py               # Integration tests
│   └── conftest.py                    # Pytest configuration
│
├── checkpoints/                       # Model checkpoints
│   ├── cross_encoder/
│   │   ├── best_model.pt
│   │   ├── final_model.pt
│   │   └── training_summary.json
│   ├── category_encoder/
│   │   ├── best_model.pt
│   │   ├── final_model.pt
│   │   └── category_mapping.json
│   └── extractor/
│       ├── extractor_model/
│       └── miner_config.json
│
├── logs/                              # Training logs
└── docs/                              # Additional documentation

========================================
QUICK START GUIDE
========================================

1. Installation:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. Train Models:
   ```bash
   # Train Category Encoder (fastest)
   python scripts/train_category.py --epochs 10
   
   # Train Cross-Encoder (main scoring model)
   python scripts/train_cross_encoder.py --epochs 20
   
   # Train Extractor (entity extraction)
   python scripts/train_extractor.py --epochs 50
   ```

3. Run API Server:
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```

4. Test Ranking:
   ```bash
   curl -X POST "http://localhost:8000/rank" \
     -H "Content-Type: application/json" \
     -d '{
       "job_description": "Senior Python Developer...",
       "job_category": "Software Engineering",
       "candidates": [{"id": "1", "text": "..."}]
     }'
   ```

========================================
MODEL ARCHITECTURE SUMMARY
========================================

┌─────────────────────────────────────────────────────────────┐
│                    TIER 3: EXTRACTION                       │
│  Model: ResumeExtractor + HardNegativeMiner                 │
│  Input: Raw resumes + Job description                       │
│  Output: Top-K candidates by skill/keyword overlap          │
│  Components:                                                │
│    - spaCy NER for entity extraction                        │
│    - TF-IDF for keyword similarity                          │
│    - Custom regex patterns for skills                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   TIER 2: CATEGORY CHECK                     │
│  Model: CategoryEncoder (DistilBERT-based)                  │
│  Input: Top-K resume texts                                  │
│  Output: Category predictions with confidence               │
│  Logic: Apply penalty (0.5x) if category mismatch           │
│  Classes: 24 job categories                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   TIER 1: DEEP SCORING                      │
│  Model: MultiHeadCrossEncoder (BERT-based)                  │
│  Input: Resume-Job description pairs                         │
│  Output: Regression score (0-1) + Classification (3-class)  │
│  Heads:                                                      │
│    - Regression: MSE loss for fit score                     │
│    - Classification: Cross-Entropy for Good/Potential/Bad   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    FINAL OUTPUT                             │
│  - Combined score = Raw score × Category penalty            │
│  - Ranked list of candidates                                │
│  - Includes confidence scores and metadata                  │
└─────────────────────────────────────────────────────────────┘

========================================
KEY FILES DESCRIPTION
========================================

1. README.md
   - Complete project documentation
   - Architecture diagrams
   - Usage examples
   - API reference

2. config.yaml
   - All configuration in one place
   - Model hyperparameters
   - Category mappings
   - Penalty weights
   - Paths and settings

3. models/cross_encoder.py
   - MultiHeadCrossEncoder class
   - CrossEncoderTrainer class
   - Methods: forward(), predict(), get_loss()

4. models/category_encoder.py
   - CategoryEncoder class
   - CategoryEncoderTrainer class
   - Methods: predict(), get_loss(), check_category_penalty()

5. models/extractor.py
   - ResumeExtractor class
   - HardNegativeMiner class
   - Methods: extract(), mine_hard_negatives(), select_top_k()

6. pipeline/inference.py
   - TriadRankPipeline class
   - BatchProcessor class
   - Main ranking flow: Tier 3 → Tier 2 → Tier 1

7. api/main.py
   - FastAPI application
   - REST endpoints: /rank, /health, /categories
   - Request/response schemas

8. scripts/train_*.py
   - Training scripts for each model
   - Logging and checkpointing
   - Evaluation metrics

========================================
PERFORMANCE METRICS TARGETS
========================================

Model 1 (Cross-Encoder):
  - MSE: < 0.05
  - Accuracy: > 85%
  - F1 (Macro): > 0.80

Model 2 (Category Encoder):
  - Accuracy: > 90%
  - F1 (Macro): > 0.85

Model 3 (Extractor):
  - Precision: > 0.85
  - Recall: > 0.80
  - F1: > 0.82

System (End-to-End):
  - Latency: < 5s for 50 candidates
  - Recall@50: > 95% (covers top candidates)

========================================
LICENSING AND CONTRIBUTIONS
========================================

For licensing information, see LICENSE file.
For contributions, see CONTRIBUTING.md (to be created).
"""

if __name__ == "__main__":
    print(PROJECT_STRUCTURE)
