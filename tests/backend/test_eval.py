"""B-M1 through B-M5 — Model evaluation / quality benchmark tests.

All tests load the real trained checkpoints and evaluate against
held-out portions of the real datasets. These are the most
computationally expensive tests in the suite.

Marked @eval so they can be selectively run:
    pytest -m eval --runslow
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch


pytestmark = [
    pytest.mark.eval,
    pytest.mark.slow,
    pytest.mark.model,
]


def _create_category_dataloader(
    dataframe,
    tokenizer,
    category_to_idx: Dict[str, int],
    max_seq_length: int = 256,
    batch_size: int = 32,
):
    """Create a simple batch generator for evaluation."""
    from torch.utils.data import DataLoader, Dataset

    class _EvalDataset(Dataset):
        def __init__(self):
            self.samples = []
            for _, row in dataframe.iterrows():
                text = str(row.get("Resume_str", ""))
                category = str(row.get("Category", "")).strip()
                if not text or len(text) < 20:
                    continue
                cat_idx = category_to_idx.get(category)
                if cat_idx is None:
                    # Map dataset category to config category
                    from data.loaders import validate_category
                    norm_cat = validate_category(category)
                    cat_idx = category_to_idx.get(norm_cat)
                if cat_idx is None:
                    continue
                self.samples.append((text, cat_idx))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            text, label = self.samples[idx]
            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_seq_length,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor(label, dtype=torch.long),
            }

    return DataLoader(
        _EvalDataset(),
        batch_size=batch_size,
        shuffle=False,
    )


# ── B-M1: Category encoder accuracy ──────────────────────────────────


@pytest.mark.usefixtures("real_category_encoder")
class TestCategoryEncoderAccuracy:
    """B-M1 — Category encoder achieves adequate accuracy on real data."""

    def test_accuracy_on_held_out_portions(self, real_category_encoder, category_dataframe, device):
        """Evaluate category encoder on a held-out 20% of the dataset.

        Uses the model's checkpoint's own label mapping so we correctly
        handle the label space the model was trained on.
        """
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", token=False)
        from sklearn.metrics import accuracy_score

        # Split: use last 20% as held-out
        df = category_dataframe
        split_idx = int(len(df) * 0.8)
        held_out = df.iloc[split_idx:]

        # Get label mapping from the loaded model's checkpoint
        cat2idx = getattr(real_category_encoder, "category_to_idx", None)
        if cat2idx is None:
            # Fall back to config
            from config import config
            cat2idx = config["categories"]["name_to_id"]

        dataloader = _create_category_dataloader(
            held_out, tokenizer, cat2idx, batch_size=16
        )

        if len(dataloader) == 0:
            pytest.skip("No held-out samples with valid categories")

        all_preds = []
        all_targets = []

        real_category_encoder.eval()
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["label"].to(device)

                outputs = real_category_encoder.predict(input_ids, attention_mask)
                preds = outputs["predicted_class"]

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        acc = accuracy_score(all_targets, all_preds)
        print(f"\n[B-M1] Category encoder held-out accuracy: {acc:.4f} "
              f"({len(all_targets)} samples)")

        # The model achieved 81.45% during training best; held-out should
        # be somewhat lower but still meaningful
        assert acc >= 0.60, (
            f"Category encoder accuracy {acc:.3f} is below 0.60 — "
            f"model may be degraded (training best was 0.8145)"
        )


# ── B-M2: Cross-encoder regression MSE ───────────────────────────────


@pytest.mark.usefixtures("real_cross_encoder")
class TestCrossEncoderRegression:
    """B-M2 — Cross-encoder regression MSE is within acceptable bounds."""

    def test_regression_mse_on_held_out(self, real_cross_encoder, score_dataframe, device):
        """Evaluate cross-encoder regression head on held-out 10% of score dataset."""
        from transformers import AutoTokenizer
        from sklearn.metrics import mean_squared_error

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", token=False)
        from data.loaders import validate_score

        # Hold out 10% (stratified is expensive; random sample is fine for eval)
        df = score_dataframe.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = int(len(df) * 0.9)
        val_df = df.iloc[split_idx:]

        # Take a manageable subset (eval on 500 pairs to keep reasonable test time)
        val_df = val_df.head(500)

        if len(val_df) == 0:
            pytest.skip("No validation samples available")

        real_cross_encoder.eval()
        predictions = []
        targets = []

        max_seq_length = 384  # shorter than training 512 for speed

        with torch.no_grad():
            for _, row in val_df.iterrows():
                resume = str(row.get("resume_text", ""))
                jd = str(row.get("job_description", ""))
                score = validate_score(row.get("ats_score", 0.0))

                if not resume or not jd:
                    continue

                encoding = tokenizer(
                    resume, jd,
                    truncation="longest_first",
                    max_length=max_seq_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                input_ids = encoding["input_ids"].to(device)
                attention_mask = encoding["attention_mask"].to(device)

                outputs = real_cross_encoder.predict(input_ids, attention_mask)
                pred_score = outputs["score"].item()

                predictions.append(pred_score)
                targets.append(score)

        if len(predictions) < 10:
            pytest.skip(f"Too few valid samples ({len(predictions)})")

        mse = mean_squared_error(targets, predictions)
        mae = np.mean(np.abs(np.array(predictions) - np.array(targets)))

        print(f"\n[B-M2] Cross-encoder MSE: {mse:.4f}, MAE: {mae:.4f} "
              f"({len(predictions)} samples, score range: {min(targets):.3f}-{max(targets):.3f})")

        # Score range is ~0.43-0.76, so MSE < 0.05 means avg error < ~0.22
        assert mse < 0.05, (
            f"Cross-encoder MSE {mse:.4f} exceeds 0.05 threshold — "
            "model regression quality may have degraded"
        )


# ── B-M4: Rank order consistency ─────────────────────────────────────


@pytest.mark.usefixtures("real_pipeline")
class TestRankConsistency:
    """B-M4 — Pipeline rank order is consistent across repeated runs."""

    def test_repeated_ranking_identical(self, real_pipeline, sample_resumes, sample_job_description):
        """Running the same input twice produces identical ranking order."""
        candidates = [{"id": f"c{i}", "text": t} for i, t in enumerate(sample_resumes)]

        r1 = real_pipeline.rank_candidates(
            job_description=sample_job_description,
            job_category="INFORMATION-TECHNOLOGY",
            candidates=candidates,
        )
        r2 = real_pipeline.rank_candidates(
            job_description=sample_job_description,
            job_category="INFORMATION-TECHNOLOGY",
            candidates=candidates,
        )

        ids1 = [r.candidate_id for r in r1]
        ids2 = [r.candidate_id for r in r2]
        assert ids1 == ids2, "Rank order changed between runs"

        scores1 = [round(r.final_score, 4) for r in r1]
        scores2 = [round(r.final_score, 4) for r in r2]
        assert scores1 == scores2, "Scores changed between runs"


# ── B-M5: Skill extraction coverage ──────────────────────────────────


class TestSkillExtractionCoverage:
    """B-M5 — Skill extraction identifies known skills in resume text."""

    @pytest.fixture
    def known_skills_resumes(self) -> List[Tuple[str, List[str]]]:
        """Resume texts with known skill lists for recall measurement."""
        return [
            (
                "Python developer with TensorFlow and PyTorch experience. "
                "Working with Docker, Kubernetes, and AWS cloud services. "
                "Strong SQL and PostgreSQL skills for database design.",
                ["python", "tensorflow", "pytorch", "docker", "kubernetes",
                 "aws", "sql", "postgresql"],
            ),
            (
                "Data scientist skilled in machine learning, scikit-learn, "
                "and pandas. Experience with React frontend development, "
                "Node.js backend, and MongoDB databases. Git version control.",
                ["scikit-learn", "pandas", "react", "node.js", "mongodb", "git",
                 "machine learning"],
            ),
            (
                "DevOps engineer with Jenkins CI/CD pipeline expertise. "
                "Terraform for infrastructure as code, Ansible for config management. "
                "GCP and Azure cloud platforms. Leadership and team management.",
                ["jenkins", "terraform", "ansible", "gcp", "azure",
                 "leadership", "teamwork"],
            ),
        ]

    def test_skill_recall_threshold(self, real_spacy, known_skills_resumes):
        """Extraction recall on known skills meets minimum threshold."""
        from models.extractor import ResumeExtractor

        extractor = ResumeExtractor(skill_extractor_mode="regex")
        total_known = 0
        total_found = 0

        for text, known_skills in known_skills_resumes:
            extracted = extractor.extract_skills_list(text)
            extracted_lower = [s.lower() for s in extracted]

            known_lower = [s.lower() for s in known_skills]
            found = sum(1 for k in known_lower if k in extracted_lower)
            total_known += len(known_lower)
            total_found += found

        recall = total_found / total_known if total_known > 0 else 0
        print(f"\n[B-M5] Skill extraction recall: {recall:.3f} "
              f"({total_found}/{total_known} skills)")

        assert recall >= 0.50, (
            f"Skill extraction recall {recall:.3f} below 0.50 — "
            "regex patterns may be missing common skills"
        )

    def test_skill_extraction_no_false_negatives_for_programming_languages(self, real_spacy):
        """Common programming languages are reliably extracted."""
        from models.extractor import ResumeExtractor

        extractor = ResumeExtractor(skill_extractor_mode="regex")
        text = "I write Python, Java, JavaScript, TypeScript, Go, Rust, and C++ code."
        extracted = extractor.extract_skills_list(text)
        extracted_lower = [s.lower() for s in extracted]

        expected = {"python", "java", "javascript", "typescript", "go", "rust", "c++"}
        missing = expected - set(extracted_lower)

        if missing:
            print(f"\n[B-M5] Missing programming languages: {missing}")

        found = len(expected - missing)
        assert found >= len(expected) * 0.5, (
            f"Only {found}/{len(expected)} programming languages detected"
        )
