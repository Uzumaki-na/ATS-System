"""B-U1, B-U2 — Config loading and normalization tests.

All tests use the REAL config.yaml (no synthetic configs).
The conftest.py _use_real_config fixture (autouse) sets TRIADRANK_CONFIG_PATH
to the real config.yaml path before every test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest


# ── B-U1: Config loads from real config.yaml ─────────────────────────


class TestConfigLoading:
    """B-U1 — Verify the real config.yaml loads correctly via config module."""

    def test_config_global_singleton_is_dict(self):
        """The global config singleton is a non-empty dict."""
        from config import config

        assert isinstance(config, dict)
        assert len(config) > 0

    def test_config_has_all_top_level_keys(self):
        """All required top-level sections are present."""
        from config import config

        required = {
            "models", "inference", "penalties", "categories",
            "entity_types", "data", "pdf_processing", "dataset_schemas",
            "hard_negative_mining", "logging", "evaluation", "mlflow",
            "api", "docker",
        }
        assert required.issubset(config.keys()), f"Missing keys: {required - config.keys()}"

    def test_config_categories_24_entries(self):
        """id_to_name and name_to_id each have exactly 24 entries (0-23)."""
        from config import config

        cats = config["categories"]
        assert len(cats["id_to_name"]) == 24
        assert len(cats["name_to_id"]) == 24
        assert set(cats["id_to_name"].keys()) == set(range(24))
        assert all(
            isinstance(v, str) for v in cats["id_to_name"].values()
        )

    def test_config_categories_roundtrip(self):
        """Every id maps to a name that maps back to the same id."""
        from config import config

        id_to_name = config["categories"]["id_to_name"]
        name_to_id = config["categories"]["name_to_id"]
        for cid, cname in id_to_name.items():
            assert name_to_id[cname] == cid, f"Mismatch for id={cid} name={cname}"

    def test_config_inference_section(self):
        """Inference section has all required keys."""
        from config import config

        inf = config["inference"]
        for key in ("device", "batch_size", "top_k", "threshold_score", "timeout_seconds", "tier3"):
            assert key in inf, f"Missing inference key: {key}"

    def test_config_models_section(self):
        """Models section has all 3 model configs with required keys."""
        from config import config

        models = config["models"]
        for model_key in ("cross_encoder", "category_encoder", "extractor"):
            assert model_key in models, f"Missing model config: {model_key}"
            assert "save_path" in models[model_key]
            assert "pretrained_model" in models[model_key] or "spacy_model" in models[model_key]

    def test_config_model_save_paths_exist(self):
        """Check that model save_path entries exist on disk.

        NOTE: This test is marked P2 because checkpoints may not be
        present in every environment (CI without model artifacts).
        """
        from config import config

        models = config["models"]
        for model_key in ("cross_encoder", "category_encoder"):
            save_path = models[model_key].get("save_path", "")
            if save_path:
                # save_path is relative to project root
                full_path = Path(__file__).resolve().parent.parent.parent / save_path
                assert full_path.exists(), (
                    f"Checkpoint not found: {full_path}. "
                    "Train the model or download artifacts."
                )


# ── B-U2: Config loading edge cases ──────────────────────────────────


class TestConfigLoadingEdgeCases:
    """B-U2 — Config behaviour with env var override and missing file."""

    def test_load_config_with_custom_path(self):
        """load_config() with explicit path returns expected keys."""
        from config import load_config

        cfg = load_config(str(CONFIG_PATH := Path(__file__).resolve().parent.parent.parent / "config.yaml"))
        assert isinstance(cfg, dict)
        assert "categories" in cfg

    def test_load_config_via_env_var(self, monkeypatch):
        """Reload config after env-var override points to real config."""
        import importlib
        import config as cfg_module

        real_path = str(Path(__file__).resolve().parent.parent.parent / "config.yaml")
        monkeypatch.setenv("TRIADRANK_CONFIG_PATH", real_path)
        importlib.reload(cfg_module)
        assert "categories" in cfg_module.config
        # Restore for other tests
        monkeypatch.delenv("TRIADRANK_CONFIG_PATH", raising=False)
        importlib.reload(cfg_module)

    def test_load_config_missing_file_raises(self, monkeypatch):
        """Missing config file path raises FileNotFoundError."""
        from config import load_config

        with pytest.raises(FileNotFoundError):
            load_config("/tmp/nonexistent_config_xyz123.yaml")

    def test_load_config_empty_file_raises(self, tmp_path, monkeypatch):
        """Empty YAML file raises no error but returns empty-like defaults handled by caller."""
        from config import load_config

        empty_cfg = tmp_path / "empty.yaml"
        empty_cfg.write_text("")
        cfg = load_config(str(empty_cfg))
        assert isinstance(cfg, dict)
        # An empty YAML file returns None from safe_load, which becomes {}
        assert len(cfg) == 0

    def test_load_config_invalid_yaml_raises(self, tmp_path):
        """Invalid YAML content raises YAMLError."""
        import yaml
        from config import load_config

        bad_cfg = tmp_path / "bad.yaml"
        bad_cfg.write_text(": invalid yaml : [ : : }")
        with pytest.raises(yaml.YAMLError):
            load_config(str(bad_cfg))


# ── B-U2: Category key normalization ─────────────────────────────────


class TestCategoryKeyNormalization:
    """B-U2 — category key normalization (int coercion)."""

    INT_CONFIG: Dict[str, Any] = {
        "categories": {
            "id_to_name": {0: "IT", 1: "HR", 2: "FINANCE"},
            "name_to_id": {"IT": 0, "HR": 1, "FINANCE": 2},
        }
    }

    STR_CONFIG: Dict[str, Any] = {
        "categories": {
            "id_to_name": {"0": "IT", "1": "HR", "2": "FINANCE"},
            "name_to_id": {"IT": "0", "HR": "1", "FINANCE": "2"},
        }
    }

    def test_normalize_int_keys_preserved(self):
        """Int-keyed id_to_name and int-valued name_to_id remain unchanged."""
        from config import _normalize_category_keys

        cfg = {k: v for k, v in self.INT_CONFIG.items()}
        _normalize_category_keys(cfg)
        cats = cfg["categories"]
        assert set(cats["id_to_name"].keys()) == {0, 1, 2}
        assert all(isinstance(k, int) for k in cats["id_to_name"].keys())
        assert all(isinstance(v, int) for v in cats["name_to_id"].values())

    def test_normalize_str_keys_coerced(self):
        """String-keyed id_to_name and str-valued name_to_id get coerced to int."""
        from config import _normalize_category_keys

        cfg = {k: v for k, v in self.STR_CONFIG.items()}
        _normalize_category_keys(cfg)
        cats = cfg["categories"]
        assert set(cats["id_to_name"].keys()) == {0, 1, 2}
        assert all(isinstance(k, int) for k in cats["id_to_name"].keys())
        assert all(isinstance(v, int) for v in cats["name_to_id"].values())

    def test_normalize_missing_categories_no_error(self):
        """_normalize_category_keys handles missing categories key gracefully."""
        from config import _normalize_category_keys

        _normalize_category_keys({})  # should not raise

    def test_normalize_missing_id_to_name_no_error(self):
        """_normalize_category_keys handles missing id_to_name or name_to_id."""
        from config import _normalize_category_keys

        _normalize_category_keys({"categories": {"id_to_name": {0: "IT"}}})
        _normalize_category_keys({"categories": {"name_to_id": {"IT": 0}}})
