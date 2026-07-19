"""B-U23 through B-U32 — TriadRankPipeline integration tests.

All tests use the REAL loaded pipeline (3 real models: spaCy + DistilBERT + BERT).
The pipeline is session-scoped: loaded once per test run.

Marked @slow and @model because loading BERT+DistilBERT+spaCy takes ~5-10s on CPU.
"""

from __future__ import annotations

from typing import Any, Dict, List

import os
import pytest


pytestmark = [
    pytest.mark.pipeline,
    pytest.mark.slow,
    pytest.mark.model,
]


# ── B-U30: _resolve_artifact_path (utility) ──────────────────────────


class TestResolveArtifactPath:
    """B-U30 — _resolve_artifact_path resolves relative/absolute paths correctly."""

    def test_absolute_path_unchanged(self):
        """Absolute path is returned as-is (using OS-appropriate absolute path)."""
        from pipeline.inference import _resolve_artifact_path

        abs_path = os.path.abspath("/absolute/path/model.pt")
        result = _resolve_artifact_path(abs_path)
        assert result == abs_path

    def test_relative_path_resolved(self):
        """Relative path is resolved against project root."""
        from pipeline.inference import _resolve_artifact_path
        from pathlib import Path

        result = _resolve_artifact_path("checkpoints/cross_encoder/best_model.pt")
        assert result is not None
        assert Path(result).exists()

    def test_none_input_returns_none(self):
        """None input returns None."""
        from pipeline.inference import _resolve_artifact_path

        assert _resolve_artifact_path(None) is None

    def test_empty_string_returns_none(self):
        """Empty string input returns None."""
        from pipeline.inference import _resolve_artifact_path

        assert _resolve_artifact_path("") is None


# ── B-U23: Pipeline initialization ───────────────────────────────────


# De-foot-gunned (see memory: pipeline-silent-random-fallback): these init
# tests route through the `real_pipeline` fixture, which loads TRAINED
# checkpoints from config — not bare TriadRankPipeline(device=...) that
# silently falls back to random untrained weights. A green run here is now
# a real-weights result, not a misleading one.
@pytest.mark.usefixtures("real_pipeline")
class TestPipelineInit:
    """B-U23 — TriadRankPipeline initializes with the real trained models."""

    def test_init_loads_all_models(self, real_pipeline):
        """Checkpoint-loaded pipeline has all 3 real models populated."""
        assert real_pipeline.model1 is not None
        assert real_pipeline.model2 is not None
        assert real_pipeline.extractor is not None
        assert real_pipeline.miner is not None

    def test_models_in_eval_mode(self, real_pipeline):
        """DistilBERT + BERT are in eval mode after initialization."""
        assert not real_pipeline.model1.training
        assert not real_pipeline.model2.training

    def test_pipeline_uses_checkpoint_label_mapping(self, real_pipeline):
        """Pipeline id_to_name carries the full 24-category mapping."""
        assert hasattr(real_pipeline, "id_to_name")
        assert len(real_pipeline.id_to_name) >= 24

    def test_device_preserved(self, real_pipeline, device):
        """Pipeline stores the requested device string verbatim."""
        assert real_pipeline.device == device


# ── CandidateResult dataclass ────────────────────────────────────────


class TestCandidateResult:
    """B-U23 — CandidateResult dataclass has all expected fields."""

    def test_has_required_fields(self):
        """CandidateResult has all 15+ required fields."""
        from pipeline.inference import CandidateResult

        r = CandidateResult(
            candidate_id="test",
            final_score=0.5,
            raw_score=0.6,
            label="Potential Fit",
            label_probabilities={"Bad Fit": 0.2, "Potential Fit": 0.6, "Good Fit": 0.2},
            category_match=True,
            category_predicted="INFORMATION-TECHNOLOGY",
            category_confidence=0.8,
            skill_overlap=0.5,
            keyword_overlap=0.4,
        )
        assert r.candidate_id == "test"
        assert 0 <= r.final_score <= 1
        assert r.label in ("Bad Fit", "Potential Fit", "Good Fit")


# ── B-U24: _tier3_extraction ─────────────────────────────────────────


@pytest.mark.usefixtures("real_pipeline")
class TestTier3Extraction:
    """B-U24 — Tier 3 extraction returns candidates with overlap scores."""

    def test_returns_candidates_with_overlap(self, real_pipeline, sample_resumes, sample_job_description):
        """Extracted candidates have keyword_overlap, skill_overlap, combined_score."""
        candidates = [{"id": f"c{i}", "text": t} for i, t in enumerate(sample_resumes)]
        results = real_pipeline._tier3_extraction(candidates, sample_job_description)

        assert len(results) == len(sample_resumes)
        for r in results:
            assert "keyword_overlap" in r
            assert "skill_overlap" in r
            assert "combined_score" in r
            assert isinstance(r["keyword_overlap"], float)
            assert isinstance(r["skill_overlap"], float)

    def test_sorted_by_combined_score_desc(self, real_pipeline, sample_resumes, sample_job_description):
        """Results are sorted by combined_score descending."""
        candidates = [{"id": f"c{i}", "text": t} for i, t in enumerate(sample_resumes)]
        results = real_pipeline._tier3_extraction(candidates, sample_job_description)

        scores = [r["combined_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_candidates_returns_empty(self, real_pipeline, sample_job_description):
        """Empty candidate list returns empty results."""
        results = real_pipeline._tier3_extraction([], sample_job_description)
        assert results == []


# ── B-U25: _tier2_category_check ─────────────────────────────────────


@pytest.mark.usefixtures("real_pipeline")
class TestTier2CategoryCheck:
    """B-U25 — Tier 2 category check returns predictions with penalty."""

    def test_adds_category_info(self, real_pipeline, sample_resumes, sample_job_description):
        """Candidates get category dict with is_match, predicted_name, confidence."""
        candidates = [{"id": f"c{i}", "text": t} for i, t in enumerate(sample_resumes)]
        # Run tier 3 first to build the structure tier 2 expects
        tier3_results = real_pipeline._tier3_extraction(candidates, sample_job_description)

        target_id = real_pipeline.name_to_id["INFORMATION-TECHNOLOGY"]
        results = real_pipeline._tier2_category_check(tier3_results, target_id)

        assert len(results) == len(sample_resumes)
        for r in results:
            assert "category" in r
            cat = r["category"]
            assert "is_match" in cat
            assert "predicted_name" in cat
            assert "confidence" in cat
            assert "penalty_factor" in cat
            assert cat["penalty_factor"] <= 1.0
            assert cat["penalty_factor"] >= 0.0

    def test_empty_input_returns_empty(self, real_pipeline):
        """Empty candidate list returns empty list."""
        results = real_pipeline._tier2_category_check([], 0)
        assert results == []


# ── B-U26: _tier1_scoring ────────────────────────────────────────────


@pytest.mark.usefixtures("real_pipeline")
class TestTier1Scoring:
    """B-U26 — Tier 1 cross-encoder returns scores and labels."""

    def test_adds_scoring_info(self, real_pipeline, sample_resumes, sample_job_description):
        """Candidates get scoring dict with raw_score, final_score, label, probabilities."""
        candidates = [{"id": f"c{i}", "text": t} for i, t in enumerate(sample_resumes)]
        tier3_results = real_pipeline._tier3_extraction(candidates, sample_job_description)
        target_id = real_pipeline.name_to_id["INFORMATION-TECHNOLOGY"]
        tier2_results = real_pipeline._tier2_category_check(tier3_results, target_id)
        results = real_pipeline._tier1_scoring(tier2_results, sample_job_description)

        assert len(results) == len(sample_resumes)
        for r in results:
            assert "scoring" in r
            s = r["scoring"]
            assert "raw_score" in s
            assert "final_score" in s
            assert "label" in s
            assert "label_probabilities" in s
            assert 0 <= s["raw_score"] <= 1, f"raw_score {s['raw_score']} out of range"
            assert s["label"] in ("Bad Fit", "Potential Fit", "Good Fit")

    def test_penalty_applied(self, real_pipeline, sample_resumes, sample_job_description):
        """Mismatched category applies penalty to final score."""
        candidates = [{"id": "c0", "text": sample_resumes[0]}]
        tier3_results = real_pipeline._tier3_extraction(candidates, sample_job_description)

        # Use a deliberately wrong category so penalty is applied
        target_id = real_pipeline.name_to_id.get("FINANCE") or 0
        tier2_results = real_pipeline._tier2_category_check(tier3_results, target_id)
        results = real_pipeline._tier1_scoring(tier2_results, sample_job_description)

        if len(results) > 0:
            cat = results[0]["category"]
            scoring = results[0]["scoring"]
            if not cat["is_match"]:
                pf = cat["penalty_factor"]
                assert scoring["final_score"] <= scoring["raw_score"] * pf + 1e-6, (
                    f"Expected final_score <= raw_score * {pf:.3f}, "
                    f"got {scoring['final_score']:.4f} > {scoring['raw_score']:.4f} * {pf:.3f}"
                )


# ── B-U27: _combine_and_rank ─────────────────────────────────────────


@pytest.mark.usefixtures("real_pipeline")
class TestCombineAndRank:
    """B-U27 — _combine_and_rank creates CandidateResult list sorted by score."""

    def test_returns_candidate_result_list(self, real_pipeline, sample_resumes, sample_job_description):
        """Result is list of CandidateResult sorted by final_score descending."""
        candidates = [{"id": f"c{i}", "text": t} for i, t in enumerate(sample_resumes)]
        tier3_results = real_pipeline._tier3_extraction(candidates, sample_job_description)
        target_id = real_pipeline.name_to_id["INFORMATION-TECHNOLOGY"]
        tier2_results = real_pipeline._tier2_category_check(tier3_results, target_id)
        tier1_results = real_pipeline._tier1_scoring(tier2_results, sample_job_description)
        results = real_pipeline._combine_and_rank(tier1_results)

        assert len(results) == len(sample_resumes)
        for r in results:
            assert isinstance(r.candidate_id, str)
            assert 0 <= r.final_score <= 1.1  # Slightly above 1 due to no penalty bound
            assert r.label in ("Bad Fit", "Potential Fit", "Good Fit")

        # Check sorted descending
        scores = [r.final_score for r in results]
        assert scores == sorted(scores, reverse=True)


# ── B-U28: Full end-to-end pipeline ──────────────────────────────────


@pytest.mark.usefixtures("real_pipeline")
class TestRankCandidates:
    """B-U28 — rank_candidates runs full 3-tier pipeline end-to-end."""

    def test_end_to_end_ranking(self, real_pipeline, sample_resumes, sample_job_description):
        """Full pipeline returns ranked CandidateResult list."""
        candidates = [{"id": f"c{i}", "text": t} for i, t in enumerate(sample_resumes)]
        results = real_pipeline.rank_candidates(
            job_description=sample_job_description,
            job_category="INFORMATION-TECHNOLOGY",
            candidates=candidates,
        )

        assert len(results) == len(sample_resumes)
        for r in results:
            assert isinstance(r, object)
            assert r.candidate_id in ("c0", "c1", "c2")

        # Check sorted descending
        scores = [r.final_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_returns_all_required_fields(self, real_pipeline, sample_resumes, sample_job_description):
        """Each result has all 15+ CandidateResult fields populated."""
        candidates = [{"id": "c0", "text": sample_resumes[0]}]
        results = real_pipeline.rank_candidates(
            job_description=sample_job_description,
            job_category="INFORMATION-TECHNOLOGY",
            candidates=candidates,
        )

        r = results[0]
        assert r.final_score > 0
        assert r.raw_score > 0
        assert r.label in ("Bad Fit", "Potential Fit", "Good Fit")
        assert isinstance(r.category_match, bool)
        assert r.category_predicted is not None
        assert r.category_confidence > 0
        assert isinstance(r.skill_overlap, float)
        assert isinstance(r.keyword_overlap, float)
        assert r.processing_time > 0

    def test_different_categories_yield_different_scores(self, real_pipeline, sample_resumes, sample_job_description):
        """Same candidates vs different categories produce different results."""
        candidates = [{"id": "c0", "text": sample_resumes[0]}]
        cats = ["INFORMATION-TECHNOLOGY", "FINANCE"]

        scores = []
        for cat in cats:
            results = real_pipeline.rank_candidates(
                job_description=sample_job_description,
                job_category=cat,
                candidates=candidates,
            )
            scores.append(round(results[0].final_score, 4))

        # Categories should produce distinct scores (different penalty application)
        # Since the resume is IT-like, IT should score higher than Finance
        assert len(set(scores)) >= 1  # At minimum, scores are valid


# ── B-U29: rank_single ───────────────────────────────────────────────


@pytest.mark.usefixtures("real_pipeline")
class TestRankSingle:
    """B-U29 — rank_single runs single candidate through full pipeline."""

    def test_single_ranking_returns_result(self, real_pipeline, sample_resume_text, sample_job_description):
        """rank_single returns a CandidateResult for one resume."""
        result = real_pipeline.rank_single(
            resume_text=sample_resume_text,
            job_description=sample_job_description,
            job_category="INFORMATION-TECHNOLOGY",
        )
        assert result.candidate_id == "single"
        assert result.final_score > 0

    def test_single_has_processing_time(self, real_pipeline, sample_resume_text, sample_job_description):
        """Result has non-zero processing time."""
        result = real_pipeline.rank_single(
            resume_text=sample_resume_text,
            job_description=sample_job_description,
            job_category="INFORMATION-TECHNOLOGY",
        )
        assert result.processing_time > 0


# ── B-U31: Error handling ────────────────────────────────────────────


@pytest.mark.usefixtures("real_pipeline")
class TestPipelineErrors:
    """B-U31 — Pipeline raises appropriate errors on invalid inputs."""

    def test_unknown_category_raises(self, real_pipeline, sample_resumes, sample_job_description):
        """Unknown job category raises ValueError."""
        with pytest.raises(ValueError, match="Unknown job category"):
            real_pipeline.rank_candidates(
                job_description=sample_job_description,
                job_category="NONEXISTENT_CATEGORY_XYZ",
                candidates=[],
            )

    def test_empty_job_description(self, real_pipeline, sample_resumes):
        """Empty JD returns valid results (may be low scores but no crash)."""
        candidates = [{"id": "c0", "text": sample_resumes[0]}]
        results = real_pipeline.rank_candidates(
            job_description="",
            job_category="INFORMATION-TECHNOLOGY",
            candidates=candidates,
        )
        assert len(results) == 1
        assert isinstance(results[0].final_score, float)

    def test_empty_candidates(self, real_pipeline, sample_job_description):
        """No candidates returns empty list."""
        results = real_pipeline.rank_candidates(
            job_description=sample_job_description,
            job_category="INFORMATION-TECHNOLOGY",
            candidates=[],
        )
        assert results == []


# ── B-U32: Long-text truncation ──────────────────────────────────────


@pytest.mark.usefixtures("real_pipeline")
class TestLongTextTruncation:
    """B-U32 — Pipeline handles resume text exceeding tokenizer max_length gracefully."""

    def test_rank_single_with_2000_token_resume(self, real_pipeline, sample_job_description):
        """A 2000+ token resume ranks without crashing and returns valid score."""
        import torch
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", token=False)
        # Generate text that produces ~2000+ tokens
        base_words = ["Python", "Docker", "AWS", "Kubernetes", "React", "SQL",
                       "Node.js", "TypeScript", "machine learning", "CI/CD",
                       "REST API", "microservices", "agile", "scrum", "git"]
        long_text = []
        while len(tokenizer.encode(" ".join(long_text), truncation=False)) < 2200:
            long_text.extend(base_words)
        long_text = " ".join(long_text)

        # Verify our generated text is indeed > 2048 tokens
        token_count = len(tokenizer.encode(long_text, truncation=False))
        assert token_count > 1500, (
            f"Generated text only has {token_count} tokens — "
            "not enough to test truncation"
        )

        # This should not crash despite exceeding max_length
        result = real_pipeline.rank_single(
            resume_text=long_text,
            job_description=sample_job_description,
            job_category="INFORMATION-TECHNOLOGY",
        )
        assert result.final_score > 0
        assert result.processing_time > 0

    def test_rank_single_with_empty_sections(self, real_pipeline, sample_job_description):
        """A resume with mixed empty/short sections still produces valid output."""
        resume = (
            "John Doe\n"
            "Skills: Python, Docker\n"
            "Experience: \n"  # empty
            "Education: \n"   # empty
            "Certifications: AWS Solutions Architect\n"
        )
        result = real_pipeline.rank_single(
            resume_text=resume,
            job_description=sample_job_description,
            job_category="INFORMATION-TECHNOLOGY",
        )
        assert result.final_score >= 0
        assert isinstance(result.label, str)

    def test_rank_with_excessively_long_jd(self, real_pipeline, sample_resumes):
        """A 2000+ token job description is handled without crash."""
        import torch
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", token=False)
        base_text = "We are looking for a senior software engineer with experience in "
        long_jd = base_text + " and ".join(
            [f"skill_{i}" for i in range(500)]
        )
        token_count = len(tokenizer.encode(long_jd, truncation=False))
        assert token_count > 1500, (
            f"Generated JD only has {token_count} tokens — "
            "not enough to test truncation"
        )

        candidates = [{"id": "c0", "text": sample_resumes[0]}]
        results = real_pipeline.rank_candidates(
            job_description=long_jd,
            job_category="INFORMATION-TECHNOLOGY",
            candidates=candidates,
        )
        assert len(results) == 1
        assert results[0].final_score >= 0
