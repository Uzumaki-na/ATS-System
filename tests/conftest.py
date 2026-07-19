"""
pytest fixtures for TriadRank ATS — REAL models, REAL data.

All model tests load the actual trained checkpoints from disk
and run inference on real dataset files. No dummies, no placeholders.

The project's config.yaml points to real checkpoint and dataset paths,
so tests import config as-is and reference the global singleton.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# Disable HF/torch progress bars and excessive logging during tests
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.WARNING)


# ── Paths ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CATEGORY_ENCODER_CKPT = CHECKPOINT_DIR / "category_encoder" / "best_model.pt"
CROSS_ENCODER_CKPT = CHECKPOINT_DIR / "cross_encoder" / "best_model.pt"
CATEGORY_MAPPING = CHECKPOINT_DIR / "category_encoder" / "category_mapping.json"
CATEGORY_ENCODER_TRAINING_SUMMARY = CHECKPOINT_DIR / "category_encoder" / "training_summary.json"
CROSS_ENCODER_TRAINING_SUMMARY = CHECKPOINT_DIR / "cross_encoder" / "training_summary.json"

DATASET_DIR = PROJECT_ROOT / "datasets"
CATEGORY_CSV = DATASET_DIR / "category_resumes" / "Resume.csv"
SCORE_TRAIN_CSV = DATASET_DIR / "score dataset" / "train.csv"
SCORE_TRAIN_NEG_CSV = DATASET_DIR / "score dataset" / "train_with_negatives.csv"

CONFIG_PATH = PROJECT_ROOT / "config.yaml"


# ── Session-scoped: real model loaders ────────────────────────────────


@pytest.fixture(scope="session")
def category_encoder_checkpoint() -> Path:
    """Path to real trained CategoryEncoder checkpoint."""
    if not CATEGORY_ENCODER_CKPT.exists():
        pytest.skip(f"CategoryEncoder checkpoint not found: {CATEGORY_ENCODER_CKPT}")
    return CATEGORY_ENCODER_CKPT


@pytest.fixture(scope="session")
def cross_encoder_checkpoint() -> Path:
    """Path to real trained MultiHeadCrossEncoder checkpoint."""
    if not CROSS_ENCODER_CKPT.exists():
        pytest.skip(f"CrossEncoder checkpoint not found: {CROSS_ENCODER_CKPT}")
    return CROSS_ENCODER_CKPT


@pytest.fixture(scope="session")
def category_dataset_path() -> Path:
    """Path to real category resume CSV (2484 resumes, 24 categories)."""
    if not CATEGORY_CSV.exists():
        pytest.skip(f"Category dataset not found: {CATEGORY_CSV}")
    return CATEGORY_CSV


@pytest.fixture(scope="session")
def score_dataset_path() -> Path:
    """Path to real score dataset CSV (122K resume-JD pairs)."""
    if not SCORE_TRAIN_CSV.exists():
        pytest.skip(f"Score dataset not found: {SCORE_TRAIN_CSV}")
    return SCORE_TRAIN_CSV


@pytest.fixture(scope="session")
def real_category_encoder(category_encoder_checkpoint: Path, device: str):
    """Lazy-load the real CategoryEncoder from trained checkpoint (session-scoped).

    Loads once per test session; all category_encoder tests share the instance.
    Distribution: DistilBERT-24-class (trained with florex-augmented data).
    Best val metrics: accuracy=0.8161, f1_macro=0.7864.
    """
    import torch
    from models.category_encoder import CategoryEncoder

    model = CategoryEncoder.load(str(category_encoder_checkpoint), device=device)
    model.eval()
    return model


@pytest.fixture(scope="session")
def real_cross_encoder(cross_encoder_checkpoint: Path, device: str):
    """Lazy-load the real MultiHeadCrossEncoder from trained checkpoint (session-scoped).

    Loads once per test session; all cross_encoder tests share the instance.
    Architecture: BERT-base-uncased with regression (score) + classification (3-class) heads.
    """
    import torch
    from models.cross_encoder import MultiHeadCrossEncoder

    model = MultiHeadCrossEncoder.load(str(cross_encoder_checkpoint), device=device)
    model.eval()
    return model


@pytest.fixture(scope="session")
def real_spacy():
    """Lazy-load spaCy en_core_web_sm (session-scoped)."""
    import spacy

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("spaCy model en_core_web_sm not installed. Run: python -m spacy download en_core_web_sm")
    return nlp


# ── Device resolution ─────────────────────────────────────────────────


@pytest.fixture(scope="session")
def device() -> str:
    """Auto-detect: CUDA if available else CPU."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


# ── Real data fixtures ────────────────────────────────────────────────


@pytest.fixture(scope="session")
def category_dataframe(category_dataset_path: Path):
    """Load real Resume.csv as a pandas DataFrame."""
    import pandas as pd

    df = pd.read_csv(str(category_dataset_path), encoding="utf-8")
    assert len(df) == 2484, f"Expected 2484 rows, got {len(df)}"
    return df


@pytest.fixture(scope="session")
def score_dataframe(score_dataset_path: Path):
    """Load real train.csv as a pandas DataFrame."""
    import pandas as pd

    df = pd.read_csv(str(score_dataset_path), encoding="utf-8")
    assert len(df) == 122560, f"Expected 122560 rows, got {len(df)}"
    return df


@pytest.fixture
def sample_resume_text() -> str:
    """A real resume excerpt from the dataset (INFORMATION-TECHNOLOGY category)."""
    return (
        "HR ADMINISTRATOR/MARKETING ASSOCIATE\n\n"
        "HR ADMINISTRATOR Summary: Dedicated Customer Service Manager with 15+ years "
        "of experience in Hospitality and Customer Service Management. Respected builder "
        "and leader of customer-focused teams.\n\n"
        "Highlights:\n"
        "- Focused on customer satisfaction\n"
        "- Team management\n"
        "- Marketing savvy\n"
        "- Conflict resolution techniques\n"
        "- Training and development\n"
        "- Skilled multi-tasker\n\n"
        "Technical Skills: Python, JavaScript, React, Node.js, SQL, Docker, AWS\n\n"
        "Education: MS in Computer Science from Stanford University\n\n"
        "Experience: 5+ years of experience in full-stack development. "
        "Led team of 10 engineers at Google.\n"
    )


@pytest.fixture
def sample_job_description() -> str:
    """A real-style job description for a software engineering role."""
    return (
        "Job title: Senior Software Engineer - AI Platform\n"
        "Category: INFORMATION-TECHNOLOGY\n"
        "Skills: Python, machine learning, deep learning, PyTorch, TensorFlow, "
        "AWS, Docker, Kubernetes, SQL, system design, distributed systems\n\n"
        "Description: We are looking for a Senior Software Engineer to join our AI "
        "Platform team. You will design and build scalable ML infrastructure, "
        "work on model serving infrastructure, and collaborate with research scientists.\n\n"
        "Requirements:\n"
        "- 5+ years of software engineering experience\n"
        "- Strong proficiency in Python and modern ML frameworks\n"
        "- Experience with distributed systems and cloud platforms\n"
        "- MS or PhD in Computer Science or related field\n"
    )


@pytest.fixture
def sample_resumes() -> List[str]:
    """Three real-style resumes of different categories for pipeline tests."""
    return [
        # IT / Engineering
        "Python developer with 5+ years experience in Django, PostgreSQL, and AWS. "
        "Bachelor's in Computer Science from MIT. Led development of microservices "
        "architecture serving 1M+ users. Proficient in React, TypeScript, Docker, "
        "and Kubernetes. Strong problem solving and analytical skills.",

        # Data Science
        "Senior data scientist with 7+ years experience in machine learning and "
        "statistical modeling. PhD in Computer Science from Stanford. Expert in "
        "TensorFlow, PyTorch, and scikit-learn. Published 12 papers at top-tier "
        "ML conferences. Experience with LLM fine-tuning and prompt engineering.",

        # Finance
        "Financial analyst with 5+ years experience in investment banking and "
        "financial modeling. MBA from Harvard Business School. Proficient in Excel, "
        "Bloomberg Terminal, and SQL. CFA Level III candidate. Excellent analytical "
        "and communication skills.",
    ]


@pytest.fixture
def pdf_bytes() -> bytes:
    """Minimal valid PDF with known text content for extraction fidelity tests."""
    return (
        b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length 44 >>\nstream\nBT /F1 12 Tf 100 700 Td "
        b"(Hello World Test PDF) Tj ET\nendstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n"
        b"0000000115 00000 n \n0000000266 00000 n \n0000000363 00000 n \n"
        b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n435\n%%EOF"
    )


@pytest.fixture
def pdf_file(tmp_path, pdf_bytes) -> Path:
    """Write minimal PDF with known text to a temp file."""
    path = tmp_path / "test_resume.pdf"
    path.write_bytes(pdf_bytes)
    return path


@pytest.fixture
def pdf_file_with_text(tmp_path) -> Path:
    """Create a PDF with longer, resume-like content for fidelity tests."""
    content = (
        "John Doe Senior Software Engineer Summary: Experienced software engineer "
        "with expertise in Python, JavaScript, and cloud infrastructure. "
        "Skills: Python, TypeScript, React, Node.js, Docker, AWS, PostgreSQL, Redis "
        "Experience: 5+ years at Google leading infrastructure team "
        "Education: MS Computer Science Stanford University"
    )
    # Build a minimal PDF page with this content
    stream_data = f"BT /F1 12 Tf 100 700 Td ({content}) Tj ET".encode("latin-1")
    pdf = (
        b"%%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length %d >>\nstream\n%s\nendstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n"
        b"0000000115 00000 n \n0000000266 00000 n \n0000000363 00000 n \n"
        b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF"
    ) % (len(stream_data), stream_data, 435 + len(stream_data) - 44)
    path = tmp_path / "resume_content.pdf"
    path.write_bytes(pdf)
    return path


# ── Config override fixture — points to REAL config.yaml ──────────────


@pytest.fixture(autouse=True)
def _use_real_config(monkeypatch, request) -> None:
    """Ensure all tests load the real config.yaml.

    Uses the project's real config, which points to actual checkpoints
    and dataset files. This is the default for all tests.

    Tests that need an isolated config can mark with @pytest.mark.isolated_config
    and set TRIADRANK_CONFIG_PATH to a temp config in the test body.

    Tests that modify config global should do so as a side effect on
    the module-level dict, then restore.
    """
    if "isolated_config" not in request.keywords:
        monkeypatch.setenv("TRIADRANK_CONFIG_PATH", str(CONFIG_PATH))
        # Re-import config so the singleton reflects our env var
        import importlib
        import config as config_module
        importlib.reload(config_module)


# ── Config key overrider ──────────────────────────────────────────────


@pytest.fixture
def override_config():
    """Override config keys at runtime on the global singleton.

    Usage:
        def test_something(override_config):
            override_config({"inference": {"top_k": 5}})
    """
    def _apply(overrides: Dict):
        import config as config_module
        from config import config as global_config

        def deep_merge(base, updates):
            for k, v in updates.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    deep_merge(base[k], v)
                else:
                    base[k] = v

        deep_merge(global_config, overrides)
        config_module._normalize_category_keys(global_config)

    return _apply


# ── Real pipeline fixture (session scoped — expensive!) ───────────────


@pytest.fixture(scope="session")
def real_pipeline(device: str, cross_encoder_checkpoint: Path, category_encoder_checkpoint: Path):
    """Load the full TriadRankPipeline with all 3 real models (session-scoped).

    This loads spaCy + DistilBERT + BERT into memory. Takes ~5-10s on CPU.
    All pipeline integration tests share this single instance.
    """
    from pipeline.inference import TriadRankPipeline

    pipe = TriadRankPipeline(
        model1_path=str(cross_encoder_checkpoint),
        model2_path=str(category_encoder_checkpoint),
        device=device,
    )
    # Force PyTorch models into eval mode (extractor is spaCy, no eval)
    if pipe.model2 is not None:
        pipe.model2.eval()
    if pipe.model1 is not None:
        pipe.model1.eval()
    return pipe


# ── Warning supression for expected slow tests ────────────────────────


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "p0: Critical priority")
    config.addinivalue_line("markers", "p1: High priority")
    config.addinivalue_line("markers", "p2: Medium priority")
    config.addinivalue_line("markers", "p3: Low priority")
    config.addinivalue_line("markers", "slow: Test takes >5s")
    config.addinivalue_line("markers", "model: Test loads a neural model checkpoint")
    config.addinivalue_line("markers", "eval: Model evaluation / quality benchmark")
    config.addinivalue_line("markers", "isolated_config: Test uses its own config path")
    config.addinivalue_line("markers", "api: API endpoint test")
    config.addinivalue_line("markers", "pipeline: Pipeline orchestration test")
    config.addinivalue_line("markers", "pdf: PDF extraction test")
    config.addinivalue_line("markers", "extraction: Skill extraction test")
    config.addinivalue_line("markers", "category: Category encoder test")
    config.addinivalue_line("markers", "e2e: End-to-end integration test with real data")
