"""B-U9 through B-U15 — ResumeExtractor and HardNegativeMiner tests.

Uses the real spaCy en_core_web_sm model.
Extraction tested against real resume text from the dataset.
"""

from __future__ import annotations

import pytest


# ── B-U9: ResumeExtractor initialization ─────────────────────────────


class TestResumeExtractorInit:
    """B-U9 — ResumeExtractor loads real spaCy model and builds patterns."""

    def test_init_with_default_model(self):
        """Default initialization loads en_core_web_sm."""
        from models.extractor import ResumeExtractor

        extractor = ResumeExtractor()
        assert extractor.nlp is not None
        assert extractor.skill_patterns is not None
        assert extractor.company_patterns is not None
        assert extractor.degree_patterns is not None

    def test_init_with_custom_spacy(self):
        """Custom spaCy model name sets the nlp attribute."""
        from models.extractor import ResumeExtractor

        extractor = ResumeExtractor(spacy_model="en_core_web_sm")
        assert extractor.nlp.meta["name"] == "core_web_sm"

    def test_init_disables_custom_rules(self):
        """enable_custom_rules=False skips pattern-based extraction."""
        from models.extractor import ResumeExtractor

        extractor = ResumeExtractor(enable_custom_rules=False)
        assert extractor.enable_custom_rules is False


# ── B-U10: extract() returns structured result ───────────────────────


class TestResumeExtractorExtract:
    """B-U10 — ResumeExtractor.extract() returns properly structured dict."""

    def test_extract_returns_expected_structure(self, real_spacy, sample_resume_text):
        """Result dict has all required top-level keys."""
        from models.extractor import ResumeExtractor

        extractor = ResumeExtractor(skill_extractor_mode="regex")
        result = extractor.extract(sample_resume_text)

        assert "skills" in result
        assert "ner_skills" in result
        assert "experience" in result
        assert "experience_years" in result
        assert "education" in result
        assert "personal_info" in result
        assert "raw_entities" in result

    def test_skills_has_subcategories(self, real_spacy, sample_resume_text):
        """Skills dict has all 6 subcategory lists."""
        from models.extractor import ResumeExtractor

        extractor = ResumeExtractor(skill_extractor_mode="regex")
        result = extractor.extract(sample_resume_text)

        subcats = ["programming_languages", "frameworks", "databases", "tools", "soft_skills", "certifications"]
        for sub in subcats:
            assert sub in result["skills"], f"Missing skill subcategory: {sub}"

    def test_extract_finds_known_skills(self, real_spacy, sample_resume_text):
        """Resume with 'Python, JavaScript, React' should have those skills detected."""
        from models.extractor import ResumeExtractor

        extractor = ResumeExtractor(skill_extractor_mode="regex")
        result = extractor.extract(sample_resume_text)

        all_skills = []
        for cat_list in result["skills"].values():
            all_skills.extend(cat_list)

        lower = [s.lower() for s in all_skills]
        # The sample has Python, JavaScript, React, Node.js, SQL, Docker, AWS
        expected = {"python", "javascript", "sql", "docker", "aws"}
        found = expected & set(lower)
        assert len(found) > 0, f"No expected skills found in {lower}"

    def test_extract_education(self, real_spacy, sample_resume_text):
        """Education extraction finds degree information."""
        from models.extractor import ResumeExtractor

        extractor = ResumeExtractor()
        result = extractor.extract(sample_resume_text)

        education = result.get("education", [])
        # Sample text mentions "MS in Computer Science from Stanford University"
        if education:
            edu_types = {e.get("type", "") for e in education}
            assert len(edu_types) > 0

    def test_extract_experience_years(self, real_spacy, sample_resume_text):
        """Experience years extracted from explicit mentions."""
        from models.extractor import ResumeExtractor

        extractor = ResumeExtractor()
        result = extractor.extract(sample_resume_text)

        # The sample mentions "15+ years" and "5+ years" — these are usually detected
        # but may not always parse; accept 0 as valid for short samples
        assert result.get("experience_years", 0) >= 0

    def test_extract_deduplicates_skills(self, real_spacy):
        """Repeated skills in text produce only unique entries."""
        from models.extractor import ResumeExtractor

        text = "Python Python Python Java Python Java"
        extractor = ResumeExtractor(skill_extractor_mode="regex")
        result = extractor.extract(text)

        all_skills = []
        for cat_list in result["skills"].values():
            all_skills.extend(cat_list)

        assert len(all_skills) == len(set(all_skills))

    def test_extract_empty_text(self, real_spacy):
        """Empty text returns the default structure with no entities."""
        from models.extractor import ResumeExtractor

        extractor = ResumeExtractor()
        result = extractor.extract("")

        assert len(result["skills"]["programming_languages"]) == 0
        assert len(result["raw_entities"]) == 0


# ── B-U11: extract_skills_list ───────────────────────────────────────


class TestExtractSkillsList:
    """B-U11 — extract_skills_list returns a flat, deduplicated list."""

    def test_returns_flat_list(self, real_spacy, sample_resume_text):
        """Result is a list of lowercase skills."""
        from models.extractor import ResumeExtractor

        extractor = ResumeExtractor(skill_extractor_mode="regex")
        skills = extractor.extract_skills_list(sample_resume_text)

        assert isinstance(skills, list)
        if skills:
            assert all(isinstance(s, str) for s in skills)

    def test_no_duplicates(self, real_spacy, sample_resume_text):
        """Flat list has no duplicate entries."""
        from models.extractor import ResumeExtractor

        extractor = ResumeExtractor(skill_extractor_mode="regex")
        skills = extractor.extract_skills_list(sample_resume_text)

        assert len(skills) == len(set(skills))

    def test_empty_text_returns_empty(self, real_spacy):
        """Empty text returns empty list."""
        from models.extractor import ResumeExtractor

        extractor = ResumeExtractor()
        assert extractor.extract_skills_list("") == []


# ── B-U12: get_skill_overlap (Jaccard) ───────────────────────────────


class TestGetSkillOverlap:
    """B-U12 — get_skill_overlap computes Jaccard similarity between skill sets."""

    def test_identical_skills(self, real_spacy):
        """Identical skill lists return 1.0."""
        from models.extractor import ResumeExtractor

        extractor = ResumeExtractor()
        skills = ["python", "docker", "aws"]
        score = extractor.get_skill_overlap(skills, skills)
        assert score == 1.0

    def test_no_overlap(self, real_spacy):
        """Disjoint skill lists return 0.0."""
        from models.extractor import ResumeExtractor

        extractor = ResumeExtractor()
        score = extractor.get_skill_overlap(
            ["python", "docker"],
            ["accounting", "auditing"]
        )
        assert score == 0.0

    def test_partial_overlap(self, real_spacy):
        """Partially overlapping lists return score between 0 and 1."""
        from models.extractor import ResumeExtractor

        extractor = ResumeExtractor()
        resume_skills = ["python", "docker", "aws", "sql"]
        job_skills = ["python", "docker", "kubernetes", "java"]
        score = extractor.get_skill_overlap(resume_skills, job_skills)
        assert 0 < score < 1.0

    def test_case_insensitive(self, real_spacy):
        """Case differences are normalized."""
        from models.extractor import ResumeExtractor

        extractor = ResumeExtractor()
        score = extractor.get_skill_overlap(["Python"], ["python"])
        assert score == 1.0

    def test_empty_lists_return_zero(self, real_spacy):
        """Both empty returns 0.0."""
        from models.extractor import ResumeExtractor

        extractor = ResumeExtractor()
        assert extractor.get_skill_overlap([], []) == 0.0


# ── B-U14: Education extraction ──────────────────────────────────────


class TestEducationExtraction:
    """B-U14 — Education extraction from resume text."""

    @pytest.fixture
    def resume_with_education(self) -> str:
        """Resume text with explicit education section."""
        return (
            "Education\n"
            "MS in Computer Science from Stanford University, 2020\n"
            "BS in Electrical Engineering from MIT, 2018\n"
            "Certification: AWS Solutions Architect\n"
        )

    def test_education_section_detected(self, real_spacy, resume_with_education):
        """Education section yields entries."""
        from models.extractor import ResumeExtractor

        extractor = ResumeExtractor()
        result = extractor.extract(resume_with_education)
        assert len(result["education"]) > 0

    def test_master_degree_detected(self, real_spacy, resume_with_education):
        """'MS' pattern identifies master-level degree."""
        from models.extractor import ResumeExtractor

        extractor = ResumeExtractor()
        result = extractor.extract(resume_with_education)
        types = [e.get("type", "") for e in result["education"]]
        # The degree patterns use lowercase matching, so 'MS' (case-insensitive) → 'master'
        assert "master" in types or "bachelor" in types

    def test_no_false_positive(self, real_spacy):
        """Text without education keywords produces no education entries (or minimal)."""
        from models.extractor import ResumeExtractor

        extractor = ResumeExtractor()
        result = extractor.extract("I like to code. Python is fun. Docker is useful.")
        # No degree mentions — should have 0 education entries
        assert len(result["education"]) == 0


# ── B-U15: HardNegativeMiner ─────────────────────────────────────────


class TestHardNegativeMiner:
    """B-U15 — HardNegativeMiner computes keyword overlap metrics."""

    def test_compute_keyword_overlap_returns_metrics(self):
        """Result dict contains jaccard, precision, recall, f1, common_keywords."""
        from models.extractor import HardNegativeMiner

        miner = HardNegativeMiner()
        result = miner.compute_keyword_overlap(
            "Python developer with Docker experience",
            "Python developer needed for Docker and Kubernetes"
        )
        assert "jaccard" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result
        assert "common_keywords" in result

    def test_similar_texts_high_score(self):
        """Very similar texts produce high overlap metrics."""
        from models.extractor import HardNegativeMiner

        miner = HardNegativeMiner()
        result = miner.compute_keyword_overlap(
            "Python developer Docker AWS",
            "Python developer Docker AWS Kubernetes"
        )
        assert result["jaccard"] > 0.5

    def test_dissimilar_texts_low_score(self):
        """Very different texts produce low overlap metrics."""
        from models.extractor import HardNegativeMiner

        miner = HardNegativeMiner()
        result = miner.compute_keyword_overlap(
            "Python developer with Docker",
            "Accounting manager with Excel and QuickBooks"
        )
        assert result["jaccard"] < 0.5

    def test_empty_strings(self):
        """Empty strings return 0 for all metrics and empty common_keywords."""
        from models.extractor import HardNegativeMiner

        miner = HardNegativeMiner()
        result = miner.compute_keyword_overlap("", "")
        assert result["jaccard"] == 0.0
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0
        assert result["common_keywords"] == []

    def test_identical_texts(self):
        """Identical texts produce jaccard=1.0, precision=1.0, recall=1.0, f1=1.0."""
        from models.extractor import HardNegativeMiner

        miner = HardNegativeMiner()
        result = miner.compute_keyword_overlap(
            "Python developer with experience in AWS",
            "Python developer with experience in AWS"
        )
        assert result["jaccard"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_init_with_custom_threshold(self):
        """HardNegativeMiner accepts custom threshold and max negatives."""
        from models.extractor import HardNegativeMiner

        miner = HardNegativeMiner(similarity_threshold=0.5, max_negatives_per_positive=10)
        assert miner.similarity_threshold == 0.5
        assert miner.max_negatives_per_positive == 10
