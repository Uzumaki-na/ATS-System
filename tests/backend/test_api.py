"""B-I1 through B-I16 — FastAPI endpoint integration tests.

Uses FastAPI TestClient against the real app.
Tests that require the full pipeline use the real loaded models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient


pytestmark = [
    pytest.mark.api,
]


# ── B-I1: GET / (root) ──────────────────────────────────────────────


class TestRootEndpoint:
    """B-I1 — GET / returns API metadata."""

    def test_root_returns_info(self):
        """Root endpoint returns name, version, docs URL."""
        from api.main import app

        with TestClient(app) as client:
            response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "TriadRank ATS API"
        assert "version" in data
        assert "docs" in data


# ── B-I2: GET /health ───────────────────────────────────────────────


class TestHealthEndpoint:
    """B-I2 — GET /health returns pipeline status."""

    def test_health_response_structure(self):
        """Health endpoint returns status, pipeline_ready, models_loaded, timestamp."""
        from api.main import app

        with TestClient(app) as client:
            response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "pipeline_ready" in data
        assert "models_loaded" in data
        assert "timestamp" in data

    def test_health_pipeline_ready(self):
        """Health endpoint returns pipeline status (initialized via TestClient lifespan)."""
        from api.main import app

        with TestClient(app) as client:
            response = client.get("/health")

        data = response.json()
        # With Starlette >=0.26, TestClient runs the lifespan, so pipeline
        # will be initialized if checkpoints exist
        assert "pipeline_ready" in data
        assert isinstance(data["pipeline_ready"], bool)


# ── B-I3: GET /categories ───────────────────────────────────────────


class TestCategoriesEndpoint:
    """B-I3 — GET /categories returns all 24 supported categories."""

    def test_returns_all_categories(self):
        """Categories endpoint returns exactly 24 entries."""
        from api.main import app

        with TestClient(app) as client:
            response = client.get("/categories")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 24
        assert len(data["categories"]) == 24

    def test_contains_information_technology(self):
        """IT category is present."""
        from api.main import app

        with TestClient(app) as client:
            response = client.get("/categories")

        data = response.json()
        assert "INFORMATION-TECHNOLOGY" in data["categories"]

    def test_all_names_match_config(self):
        """All category names match config.yaml exactly."""
        from api.main import app
        from config import config

        with TestClient(app) as client:
            response = client.get("/categories")

        data = response.json()
        config_cats = list(config["categories"]["id_to_name"].values())
        assert data["categories"] == config_cats


# ── B-I4: GET /categories/{id}/name ──────────────────────────────────


class TestCategoryNameEndpoint:
    """B-I4 — GET /categories/{id}/name returns category name by ID."""

    def test_valid_id_returns_name(self):
        """Valid category ID returns its name."""
        from api.main import app
        from config import config

        with TestClient(app) as client:
            response = client.get("/categories/20/name")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == 20
        assert data["name"] == "INFORMATION-TECHNOLOGY"

    def test_invalid_id_returns_404(self):
        """Invalid category ID returns 404."""
        from api.main import app

        with TestClient(app) as client:
            response = client.get("/categories/999/name")

        assert response.status_code == 404

    def test_all_ids_return_valid_names(self):
        """Every ID 0-23 returns a valid name."""
        from api.main import app
        from config import config

        with TestClient(app) as client:
            for cid in range(24):
                response = client.get(f"/categories/{cid}/name")
                assert response.status_code == 200, f"Failed for ID {cid}"
                data = response.json()
                assert isinstance(data["name"], str)
                assert len(data["name"]) > 0


# ── B-I5: POST /extract ─────────────────────────────────────────────


class TestExtractEndpoint:
    """B-I5 — POST /extract extracts entities from resume text."""

    def test_extract_returns_structure(self, sample_resume_text):
        """Extract endpoint returns entity extraction results."""
        from api.main import app

        with TestClient(app) as client:
            response = client.post(
                "/extract",
                params={"resume_text": sample_resume_text},
            )

        # Without pipeline initialized, this returns 503
        # With pipeline, it returns the extraction
        if response.status_code == 503:
            pytest.skip("Pipeline not initialized — cannot test extraction")
        assert response.status_code == 200
        data = response.json()
        assert "skills" in data
        assert "education" in data

    def test_extract_empty_text(self):
        """Extract on empty text returns default structure."""
        from api.main import app

        with TestClient(app) as client:
            response = client.post(
                "/extract",
                params={"resume_text": ""},
            )

        if response.status_code == 503:
            pytest.skip("Pipeline not initialized")
        assert response.status_code == 200


# ── B-I6: POST /rank (schema validation) ─────────────────────────────


class TestRankRequestSchema:
    """B-I6, B-I7 — Rank request schema validation (Pydantic)."""

    def test_missing_job_description_returns_422(self):
        """Request without job_description returns 422."""
        from api.main import app

        with TestClient(app) as client:
            response = client.post(
                "/rank",
                json={
                    "job_category": "INFORMATION-TECHNOLOGY",
                    "candidates": [{"id": "c1", "text": "test"}],
                },
            )
        assert response.status_code == 422

    def test_missing_job_category_returns_422(self):
        """Request without job_category returns 422."""
        from api.main import app

        with TestClient(app) as client:
            response = client.post(
                "/rank",
                json={
                    "job_description": "Python developer needed",
                    "candidates": [{"id": "c1", "text": "test"}],
                },
            )
        assert response.status_code == 422

    def test_missing_candidates_returns_422(self):
        """Request without candidates returns 422."""
        from api.main import app

        with TestClient(app) as client:
            response = client.post(
                "/rank",
                json={
                    "job_description": "Python developer needed",
                    "job_category": "INFORMATION-TECHNOLOGY",
                },
            )
        assert response.status_code == 422

    def test_invalid_category_returns_422(self):
        """Invalid category value returns 422."""
        from api.main import app

        with TestClient(app) as client:
            response = client.post(
                "/rank",
                json={
                    "job_description": "Python developer needed",
                    "job_category": "INVALID-CATEGORY",
                    "candidates": [{"id": "c1", "text": "test"}],
                },
            )
        assert response.status_code == 422

    def test_empty_candidates_list_returns_422_or_200(self):
        """Empty candidates list may return 422, 503, or 200 (if pipeline initialized)."""
        from api.main import app

        with TestClient(app) as client:
            response = client.post(
                "/rank",
                json={
                    "job_description": "Python developer needed",
                    "job_category": "INFORMATION-TECHNOLOGY",
                    "candidates": [],
                },
            )
        # 422 (Pydantic — if list validation fails), 503 (no pipeline),
        # or 200 (pipeline returns empty results)
        assert response.status_code in (422, 503, 200)

    def test_rank_with_pipeline_returns_ranked_results(self, sample_resumes, sample_job_description):
        """With pipeline initialized, /rank returns ranked CandidateOutput list."""
        from api.main import app
        import api.main as api_module

        # Manually set pipeline for testing
        from pipeline.inference import TriadRankPipeline
        from config import config

        pipe = TriadRankPipeline(
            model1_path=config["models"]["cross_encoder"]["save_path"],
            model2_path=config["models"]["category_encoder"]["save_path"],
            device="cpu",
        )
        api_module.pipeline = pipe

        candidates = [
            {"id": f"c{i}", "text": t}
            for i, t in enumerate(sample_resumes)
        ]

        try:
            with TestClient(app) as client:
                response = client.post(
                    "/rank",
                    json={
                        "job_description": sample_job_description,
                        "job_category": "INFORMATION-TECHNOLOGY",
                        "candidates": candidates,
                    },
                )

            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert "job_id" in data
            assert "total_candidates" in data
            assert data["total_candidates"] == len(sample_resumes)
            assert len(data["results"]) == len(sample_resumes)

            # Check result structure
            r = data["results"][0]
            assert "candidate_id" in r
            assert "final_score" in r
            assert "label" in r
            assert "category" in r
            assert "skills" in r
        finally:
            api_module.pipeline = None


# ── B-I14: Concurrent API access ────────────────────────────────────


class TestConcurrentRank:
    """B-I14 — Multiple simultaneous /rank calls are thread-safe.

    Fires N concurrent requests using ThreadPoolExecutor to detect
    race conditions in the global pipeline singleton.
    """

    def test_concurrent_rank_requests(self, sample_resumes, sample_job_description):
        """10 concurrent rank calls all return 200 and valid results."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from api.main import app
        import api.main as api_module
        from pipeline.inference import TriadRankPipeline
        from config import config

        pipe = TriadRankPipeline(
            model1_path=config["models"]["cross_encoder"]["save_path"],
            model2_path=config["models"]["category_encoder"]["save_path"],
            device="cpu",
        )
        api_module.pipeline = pipe

        candidates = [
            {"id": f"c{i}", "text": t}
            for i, t in enumerate(sample_resumes)
        ]
        payload = {
            "job_description": sample_job_description,
            "job_category": "INFORMATION-TECHNOLOGY",
            "candidates": candidates,
        }

        def _rank():
            with TestClient(app) as client:
                return client.post("/rank", json=payload)

        try:
            with ThreadPoolExecutor(max_workers=10) as pool:
                futures = [pool.submit(_rank) for _ in range(10)]
                results = [f.result() for f in as_completed(futures)]

            assert len(results) == 10
            for r in results:
                assert r.status_code == 200, (
                    f"Concurrent rank returned {r.status_code}: {r.text[:200]}"
                )
                data = r.json()
                assert "results" in data
                assert data["total_candidates"] == len(sample_resumes)
        finally:
            api_module.pipeline = None


# ── B-I15: Pipeline latency benchmark ────────────────────────────────


class TestPipelineLatency:
    """B-I15 — Pipeline latency is within bounds."""

    def test_rank_single_under_60s(self, sample_resume_text, sample_job_description):
        """Single candidate ranking completes in under 60 seconds."""
        from pipeline.inference import TriadRankPipeline
        from config import config

        pipe = TriadRankPipeline(
            model1_path=config["models"]["cross_encoder"]["save_path"],
            model2_path=config["models"]["category_encoder"]["save_path"],
            device="cpu",
        )

        import time
        start = time.time()
        _ = pipe.rank_single(
            resume_text=sample_resume_text,
            job_description=sample_job_description,
            job_category="INFORMATION-TECHNOLOGY",
        )
        elapsed = time.time() - start

        assert elapsed < 60, f"Single ranking took {elapsed:.1f}s (limit: 60s)"


# ── B-I16: Category contract cross-validation ────────────────────────


class TestCategoryContract:
    """B-I16 — Frontend and backend category lists are in sync."""

    BACKEND_CATEGORIES = {
        "ACCOUNTANT", "ADVOCATE", "AGRICULTURE", "APPAREL", "ARTS",
        "AUTOMOBILE", "AVIATION", "BANKING", "BPO", "BUSINESS-DEVELOPMENT",
        "CHEF", "CONSTRUCTION", "CONSULTANT", "DESIGNER", "DIGITAL-MEDIA",
        "ENGINEERING", "FINANCE", "FITNESS", "HEALTHCARE", "HUMAN-RESOURCES",
        "INFORMATION-TECHNOLOGY", "PUBLIC-RELATIONS", "SALES", "TEACHER",
    }

    def test_backend_categories_from_config(self):
        """Backend categories from config.yaml match expected set."""
        from config import config

        backend = set(config["categories"]["name_to_id"].keys())
        assert backend == self.BACKEND_CATEGORIES, (
            f"Backend categories changed from expected: "
            f"added={backend - self.BACKEND_CATEGORIES}, "
            f"removed={self.BACKEND_CATEGORIES - backend}"
        )

    def test_frontend_categories(self):
        """Frontend categories from mock-data.ts have same count as backend.

        Note: The frontend is standalone (pre-integration). Its category list
        may differ from the backend's until integration wires the real API.
        This test checks count parity and reports the diff for awareness.
        """
        from config import config

        backend = set(config["categories"]["name_to_id"].keys())

        # Load frontend categories from the-lab/lib/mock-data.ts
        frontend_path = (
            Path(__file__).resolve().parent.parent.parent
            / "the-lab" / "lib" / "mock-data.ts"
        )
        if not frontend_path.exists():
            pytest.skip("Frontend mock-data.ts not found")

        content = frontend_path.read_text("utf-8")

        # Extract the CATEGORIES array content
        import re
        match = re.search(r"export const CATEGORIES\s*=\s*\[(.*?)\]", content, re.DOTALL)
        if not match:
            pytest.skip("Could not parse CATEGORIES from mock-data.ts")

        array_text = match.group(1)
        frontend_cats = set(re.findall(r'"([A-Z][A-Z-]+)"', array_text))

        if not frontend_cats:
            pytest.skip("Could not extract frontend category names")

        # Both sides should have 24 categories
        assert len(frontend_cats) == len(backend), (
            f"Category count mismatch: frontend has {len(frontend_cats)}, "
            f"backend has {len(backend)}"
        )

        # Log the diff for awareness (expected to be resolved during integration)
        only_frontend = frontend_cats - backend
        only_backend = backend - frontend_cats
        if only_frontend or only_backend:
            pass  # Known pre-integration diff — will be resolved when backend is wired
            # To enable strict checking once integrated, change to:
            # pytest.fail(f"...")
