"""B-U21, B-U22 — MultiHeadCrossEncoder tests with real checkpoint.

All tests use the trained BERT checkpoint at
checkpoints/cross_encoder/best_model.pt (211 layers, 3 classes).

Tests are session-scoped; the model loads once per pytest run.
Marked @slow because loading BERT on CPU takes ~3–5s.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest


pytestmark = [
    pytest.mark.model,
    pytest.mark.slow,
]


# ── B-U21: MultiHeadCrossEncoder.from_config ─────────────────────────


class TestCrossEncoderFromConfig:
    """B-U21 — MultiHeadCrossEncoder.from_config creates model from config.yaml."""

    def test_from_config_defaults(self):
        """from_config with no args reads from real config.yaml."""
        from models.cross_encoder import MultiHeadCrossEncoder

        model = MultiHeadCrossEncoder.from_config()
        assert model.num_classes == 3
        assert model.pretrained_model_name == "bert-base-uncased"
        assert model.hidden_size == 768

    def test_from_config_with_override(self):
        """from_config accepts explicit config dict."""
        from models.cross_encoder import MultiHeadCrossEncoder

        model = MultiHeadCrossEncoder.from_config({
            "pretrained_model": "bert-base-uncased",
            "num_classes": 3,
            "hidden_size": 768,
            "regression_head_units": 256,
            "classification_head_units": 128,
        })
        assert model.num_classes == 3
        assert model.regression_head is not None
        assert model.classification_head is not None

    def test_forward_output_shapes(self):
        """Forward pass returns regression score + classification logits with correct shapes."""
        import torch
        from models.cross_encoder import MultiHeadCrossEncoder

        model = MultiHeadCrossEncoder()
        model.eval()

        batch = {
            "input_ids": torch.randint(0, 1000, (2, 384)),
            "attention_mask": torch.ones(2, 384, dtype=torch.long),
        }
        with torch.no_grad():
            outputs = model(**batch)

        assert "regression_score" in outputs
        assert "classification_logits" in outputs
        assert outputs["regression_score"].shape == (2,)
        assert outputs["classification_logits"].shape == (2, 3)

    def test_classification_logits_not_uniform(self):
        """With real weights, logits are not all identical (sanity for initialization)."""
        import torch
        from models.cross_encoder import MultiHeadCrossEncoder

        model = MultiHeadCrossEncoder()
        model.eval()

        batch = {
            "input_ids": torch.randint(0, 1000, (1, 384)),
            "attention_mask": torch.ones(1, 384, dtype=torch.long),
        }
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs["classification_logits"][0]
        # All 3 logits should not be equal (untrained would be uniform)
        assert not torch.allclose(logits[0], logits[1]), (
            "Logits are suspiciously uniform — model may be randomly initialized"
        )


# ── B-U22: MultiHeadCrossEncoder.load from real checkpoint ───────────


@pytest.mark.usefixtures("real_cross_encoder")
class TestCrossEncoderLoad:
    """B-U22 — MultiHeadCrossEncoder.load restores trained weights from real checkpoint."""

    def test_load_returns_eval_mode(self, real_cross_encoder):
        """Loaded model is in eval mode."""
        assert not real_cross_encoder.training

    def test_load_correct_num_classes(self, real_cross_encoder):
        """Loaded model has 3 output classes."""
        assert real_cross_encoder.num_classes == 3

    def test_load_bert_architecture(self, real_cross_encoder):
        """Loaded model uses bert-base-uncased."""
        assert real_cross_encoder.pretrained_model_name == "bert-base-uncased"

    def test_predict_returns_all_keys(self, real_cross_encoder, device):
        """predict() returns score, predicted_class, probabilities, logits."""
        import torch

        input_ids = torch.randint(0, 1000, (1, 384)).to(device)
        attention_mask = torch.ones(1, 384, dtype=torch.long).to(device)

        with torch.no_grad():
            result = real_cross_encoder.predict(input_ids, attention_mask)

        assert "score" in result
        assert "predicted_class" in result
        assert "probabilities" in result
        assert "logits" in result

    def test_score_in_0_1_range(self, real_cross_encoder, device):
        """Regression score is clamped to [0.0, 1.0]."""
        import torch

        input_ids = torch.randint(0, 1000, (1, 384)).to(device)
        attention_mask = torch.ones(1, 384, dtype=torch.long).to(device)

        with torch.no_grad():
            result = real_cross_encoder.predict(input_ids, attention_mask)

        score = result["score"].item()
        assert 0.0 <= score <= 1.0, f"Score {score} outside [0, 1]"

    def test_predicted_class_is_valid(self, real_cross_encoder, device):
        """predicted_class is 0, 1, or 2."""
        import torch

        input_ids = torch.randint(0, 1000, (1, 384)).to(device)
        attention_mask = torch.ones(1, 384, dtype=torch.long).to(device)

        with torch.no_grad():
            result = real_cross_encoder.predict(input_ids, attention_mask)

        pred = result["predicted_class"].item()
        assert pred in (0, 1, 2), f"Predicted class {pred} not in {0, 1, 2}"

    def test_probabilities_sum_to_one(self, real_cross_encoder, device):
        """Softmax probabilities across 3 classes sum to ~1.0."""
        import torch

        input_ids = torch.randint(0, 1000, (1, 384)).to(device)
        attention_mask = torch.ones(1, 384, dtype=torch.long).to(device)

        with torch.no_grad():
            result = real_cross_encoder.predict(input_ids, attention_mask)

        probs = result["probabilities"]
        assert abs(probs.sum().item() - 1.0) < 0.01

    def test_checkpoint_keys(self, cross_encoder_checkpoint):
        """Checkpoint contains expected keys for MultiHeadCrossEncoder."""
        import torch

        ckpt = torch.load(str(cross_encoder_checkpoint), map_location="cpu")
        assert "pretrained_model_name" in ckpt
        assert ckpt["pretrained_model_name"] == "bert-base-uncased"
        assert "num_classes" in ckpt
        assert ckpt["num_classes"] == 3
        assert "model_state_dict" in ckpt

    def test_predict_different_inputs_different_scores(self, real_cross_encoder, device):
        """Same IT resume scores higher when paired with IT JD than Finance JD."""
        import torch
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", token=False)
        max_len = 384

        it_resume = "Python developer with 5 years of Django, React, and AWS experience"
        it_jd = "Senior Software Engineer — Python, Django, React, AWS, Docker, Kubernetes"
        fin_jd = "Senior Accountant — financial reporting, budgeting, QuickBooks, Excel"

        # Same IT resume paired with two different JDs — the matching JD should score higher
        pairs = [
            (it_resume, it_jd, "Matched (IT resume + IT JD)"),
            (it_resume, fin_jd, "Mismatched (IT resume + Finance JD)"),
        ]

        scores = {}
        for resume, jd, label in pairs:
            encoding = tokenizer(
                resume, jd,
                truncation="longest_first",
                max_length=max_len,
                padding="max_length",
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                result = real_cross_encoder.predict(
                    encoding["input_ids"], encoding["attention_mask"],
                    token_type_ids=encoding.get("token_type_ids"),
                )
            scores[label] = result["score"].item()

        matched = "Matched (IT resume + IT JD)"
        mismatched = "Mismatched (IT resume + Finance JD)"
        assert scores[matched] > scores[mismatched], (
            f"Expected IT resume+IT JD ({scores[matched]:.4f}) to score higher than "
            f"IT resume+Finance JD ({scores[mismatched]:.4f})"
        )

    def test_batch_inference_consistency(self, real_cross_encoder, device):
        """Same input twice yields identical predictions (deterministic eval mode)."""
        import torch
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", token=False)
        resume = "Python developer with 5 years experience"
        jd = "Looking for a Python developer"

        encoding = tokenizer(
            resume, jd,
            truncation="longest_first",
            max_length=384,
            padding="max_length",
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            r1 = real_cross_encoder.predict(
                encoding["input_ids"], encoding["attention_mask"]
            )
            r2 = real_cross_encoder.predict(
                encoding["input_ids"], encoding["attention_mask"]
            )

        assert torch.equal(r1["predicted_class"], r2["predicted_class"])
        assert torch.allclose(r1["score"], r2["score"], atol=1e-6)
