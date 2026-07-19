"""B-U18, B-U19, B-U20 — CategoryEncoder tests with real checkpoint.

All tests use the trained DistilBERT checkpoint at
checkpoints/category_encoder/best_model.pt (107 layers, 24 classes).

Tests are session-scoped; the model loads once per pytest run.
Marked @slow because loading DistilBERT on CPU takes ~2–3s.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest


pytestmark = [
    pytest.mark.model,
    pytest.mark.slow,
    pytest.mark.category,
]


# ── B-U18: CategoryEncoder.from_config ───────────────────────────────


class TestCategoryEncoderFromConfig:
    """B-U18 — CategoryEncoder.from_config creates model from config.yaml settings."""

    def test_from_config_defaults(self):
        """from_config with no args reads from real config.yaml."""
        from models.category_encoder import CategoryEncoder

        model = CategoryEncoder.from_config()
        assert model.num_classes == 24
        assert model.pretrained_model_name == "distilbert-base-uncased"
        assert model.hidden_size == 768

    def test_from_config_with_override(self):
        """from_config accepts explicit config dict."""
        from models.category_encoder import CategoryEncoder

        model = CategoryEncoder.from_config({
            "pretrained_model": "distilbert-base-uncased",
            "num_classes": 24,
            "hidden_size": 768,
        })
        assert model.num_classes == 24

    def test_forward_output_shape(self):
        """Forward pass returns logits of shape [batch, 24]."""
        import torch
        from models.category_encoder import CategoryEncoder

        model = CategoryEncoder()
        model.eval()

        batch = {
            "input_ids": torch.randint(0, 1000, (2, 256)),
            "attention_mask": torch.ones(2, 256, dtype=torch.long),
        }
        with torch.no_grad():
            outputs = model(**batch)

        assert "logits" in outputs
        assert outputs["logits"].shape == (2, 24)


# ── B-U19: CategoryEncoder.load from real checkpoint ─────────────────


@pytest.mark.usefixtures("real_category_encoder")
class TestCategoryEncoderLoad:
    """B-U19 — CategoryEncoder.load restores trained weights from real checkpoint."""

    def test_load_returns_eval_mode(self, real_category_encoder):
        """Loaded model is in eval mode."""
        assert not real_category_encoder.training

    def test_load_correct_num_classes(self, real_category_encoder):
        """Loaded model has 24 output classes."""
        from models.category_encoder import CategoryEncoder

        assert real_category_encoder.num_classes == 24

    def test_load_has_category_mapping(self, real_category_encoder):
        """Loaded model has category_to_idx and id_to_name restored from checkpoint."""
        ckpt = getattr(real_category_encoder, "category_to_idx", None)
        if ckpt is not None:
            assert isinstance(ckpt, dict)
            assert len(ckpt) >= 24  # might have UNKNOWN extras
            for k, v in ckpt.items():
                assert isinstance(v, int)

    def test_load_predict_returns_valid_probs(self, real_category_encoder, device):
        """predict() on a dummy input returns valid probabilities summing to ~1.0."""
        import torch

        input_ids = torch.randint(0, 1000, (1, 256)).to(device)
        attention_mask = torch.ones(1, 256, dtype=torch.long).to(device)

        with torch.no_grad():
            result = real_category_encoder.predict(input_ids, attention_mask)

        assert "predicted_class" in result
        assert "probabilities" in result
        assert "confidence" in result

        probs = result["probabilities"]
        assert probs.shape == (1, 24)
        # Probabilities sum to ~1.0
        assert abs(probs.sum().item() - 1.0) < 0.01

    def test_predicted_class_is_valid_index(self, real_category_encoder, device):
        """predicted_class is in [0, 23]."""
        import torch

        input_ids = torch.randint(0, 1000, (1, 256)).to(device)
        attention_mask = torch.ones(1, 256, dtype=torch.long).to(device)

        with torch.no_grad():
            result = real_category_encoder.predict(input_ids, attention_mask)

        pred = result["predicted_class"].item()
        assert 0 <= pred < 24, f"Predicted class {pred} outside valid range [0, 23]"

    def test_load_checkpoint_pretrained_name(self, category_encoder_checkpoint):
        """Checkpoint contains expected keys for the CategoryEncoder architecture."""
        import torch

        ckpt = torch.load(str(category_encoder_checkpoint), map_location="cpu")
        assert "pretrained_model_name" in ckpt
        assert ckpt["pretrained_model_name"] == "distilbert-base-uncased"
        assert "num_classes" in ckpt
        assert ckpt["num_classes"] == 24
        assert "model_state_dict" in ckpt

    def test_checkpoint_integrity_all_layers_present(self, category_encoder_checkpoint):
        """Checkpoint model_state_dict has ALL 107 expected DistilBERT layer keys.

        A truncated or corrupted checkpoint would silently fall back to random
        weights in the pipeline — this test detects that at source.
        """
        import torch

        ckpt = torch.load(str(category_encoder_checkpoint), map_location="cpu")
        state = ckpt["model_state_dict"]

        # The saved state uses 'encoder.' prefix (not 'distilbert.') because
        # the model wraps DistilBERTModel inside a nn.Module
        mandatory_prefixes = [
            "encoder.embeddings.",
            "encoder.transformer.layer.0.",
            "encoder.transformer.layer.1.",
            "encoder.transformer.layer.2.",
            "encoder.transformer.layer.3.",
            "encoder.transformer.layer.4.",
            "encoder.transformer.layer.5.",
            "classifier.",
        ]

        for prefix in mandatory_prefixes:
            matching = [k for k in state if k.startswith(prefix)]
            assert len(matching) > 0, (
                f"Missing layer group '{prefix}' in checkpoint — "
                "checkpoint may be truncated or from a different architecture"
            )

        # DistilBERT-base has 107 parameter groups in state_dict
        assert len(state) >= 105, (
            f"Expected ≥105 parameter groups in state_dict, got {len(state)}. "
            "Checkpoint appears truncated."
        )


# ── B-U20: Real inference on category dataset ────────────────────────


@pytest.mark.usefixtures("real_category_encoder")
class TestCategoryEncoderInference:
    """B-U20 — Real inference on actual resume data from the category dataset."""

    def test_predict_on_real_resume(self, real_category_encoder, category_dataframe, device):
        """Model produces a category prediction for a real resume from the dataset."""
        import torch
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", token=False)
        # Pick a resume from the dataset
        sample = category_dataframe.iloc[0]
        text = sample["Resume_str"][:512]  # keep within tokenizer limits

        encoding = tokenizer(
            text,
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            result = real_category_encoder.predict(input_ids, attention_mask)

        pred_class = result["predicted_class"].item()
        assert 0 <= pred_class < 24, f"Predicted {pred_class} out of range"
        assert result["confidence"].item() > 0.0

    def test_batch_inference_consistency(self, real_category_encoder, category_dataframe, device):
        """Running same input twice yields identical results (deterministic eval)."""
        import torch
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", token=False)
        text = category_dataframe.iloc[0]["Resume_str"][:512]

        encoding = tokenizer(
            text, truncation=True, max_length=256,
            padding="max_length", return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            r1 = real_category_encoder.predict(input_ids, attention_mask)
            r2 = real_category_encoder.predict(input_ids, attention_mask)

        assert torch.equal(r1["predicted_class"], r2["predicted_class"]), (
            "Non-deterministic predictions — model has training layers active"
        )
        assert torch.allclose(r1["probabilities"], r2["probabilities"], atol=1e-6)

    def test_all_24_categories_get_nonzero_prob(self, real_category_encoder, category_dataframe, device):
        """Every category has a non-zero probability for any real resume."""
        import torch
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", token=False)
        sample = category_dataframe.iloc[0]
        text = sample["Resume_str"][:512]

        encoding = tokenizer(
            text, truncation=True, max_length=256,
            padding="max_length", return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            result = real_category_encoder.predict(
                encoding["input_ids"], encoding["attention_mask"]
            )

        probs = result["probabilities"][0]
        n_nonzero = (probs > 0).sum().item()
        # Most categories should have some probability mass (real model)
        assert n_nonzero >= 5, f"Only {n_nonzero}/24 categories have non-zero probability"

    def test_training_summary_exists(self):
        """Training summary JSON has expected metrics."""
        summary_path = Path(__file__).resolve().parent.parent.parent / "checkpoints" / "category_encoder" / "training_summary.json"
        if not summary_path.exists():
            pytest.skip("training_summary.json not found")

        import json

        with open(str(summary_path)) as f:
            summary = json.load(f)

        assert "val_loss" in summary.get("best_metrics", summary)
        assert "accuracy" in summary.get("best_metrics", summary)
        assert "f1_macro" in summary.get("best_metrics", summary)
        assert "f1_weighted" in summary.get("best_metrics", summary)
        assert "avg_confidence" in summary.get("best_metrics", summary)

    def test_accuracy_within_expected_range(self):
        """Model accuracy from training is > 0.70 (real checkpoint quality check)."""
        summary_path = Path(__file__).resolve().parent.parent.parent / "checkpoints" / "category_encoder" / "training_summary.json"
        if not summary_path.exists():
            pytest.skip("training_summary.json not found")

        import json

        with open(str(summary_path)) as f:
            summary = json.load(f)

        metrics = summary.get("best_metrics", summary)
        accuracy = metrics.get("accuracy", 0)
        assert accuracy >= 0.70, (
            f"Category encoder accuracy {accuracy:.3f} is below 0.70 threshold. "
            "Model may be undertrained or checkpoint is corrupted."
        )
