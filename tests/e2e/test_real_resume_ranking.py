"""B-E1 through B-E4 — End-to-end pipeline tests with real resume text.

Loads real human-written resumes from the florex/resume_corpus dataset
(29,783 multi-label resumes) and runs them through the full TriadRank
pipeline. These tests verify that the system produces meaningful scores
and rankings on authentic (not synthetic) resume text.

Dataset location: datasets/external/resume_corpus/resumes_corpus/
Resume files: *.txt (text), *.lab (occupation labels)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import pytest


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.model,
]


# ── Dataset path ─────────────────────────────────────────────────────

FLOREX_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "datasets" / "external" / "resume_corpus" / "resumes_corpus"
)

# Map florex occupation labels to our 24 ATS categories
OCCUPATION_TO_CATEGORY = {
    "Software_Developer": "INFORMATION-TECHNOLOGY",
    "Front_End_Developer": "INFORMATION-TECHNOLOGY",
    "Web_Developer": "INFORMATION-TECHNOLOGY",
    "Java_Developer": "INFORMATION-TECHNOLOGY",
    "Python_Developer": "INFORMATION-TECHNOLOGY",
    "Database_Administrator": "INFORMATION-TECHNOLOGY",
    "Network_Administrator": "INFORMATION-TECHNOLOGY",
    "Systems_Administrator": "INFORMATION-TECHNOLOGY",
    "Security_Analyst": "INFORMATION-TECHNOLOGY",
    "Project_manager": "BUSINESS-DEVELOPMENT",
    "Sales": "SALES",
    "Teacher": "TEACHER",
    "Accountant": "ACCOUNTANT",
    "Financial_Analyst": "FINANCE",
    "Nurse": "HEALTHCARE",
    "Doctor": "HEALTHCARE",
    "Chef": "CHEF",
    "Construction_Worker": "CONSTRUCTION",
    "Lawyer": "ADVOCATE",
    "HR_Manager": "HUMAN-RESOURCES",
    "Marketing_Manager": "DIGITAL-MEDIA",
    "Artist": "ARTS",
    "Designer": "DESIGNER",
    "Consultant": "CONSULTANT",
    "Banker": "BANKING",
    "Engineer": "ENGINEERING",
    "Aviation": "AVIATION",
    "Fitness_Trainer": "FITNESS",
    "Public_Relations": "PUBLIC-RELATIONS",
    "Automobile": "AUTOMOBILE",
    "Agriculture": "AGRICULTURE",
    "Apparel": "APPAREL",
}


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode common entities from resume text."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;|\xa0", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _load_real_resumes(
    max_files: int = 20,
    min_words: int = 50,
) -> List[Tuple[str, str, str]]:
    """Load real resume texts from the florex corpus.

    Returns list of (file_id, category, cleaned_text) tuples.
    Only resumes with ≥ `min_words` words and a mapped category are returned.
    """
    if not FLOREX_DIR.exists():
        pytest.skip(f"Florex resume corpus not found at {FLOREX_DIR}")

    txt_files = sorted(FLOREX_DIR.glob("*.txt"))
    if not txt_files:
        pytest.skip("No resume text files found in florex corpus")

    results = []
    for tf in txt_files:
        if len(results) >= max_files:
            break
        stem = tf.stem
        lab_file = tf.with_suffix(".lab")

        # Read label
        if not lab_file.exists():
            continue
        label = lab_file.read_text("utf-8").strip()

        # Map to our category
        cat = OCCUPATION_TO_CATEGORY.get(label)
        if cat is None:
            continue

        # Read and clean text
        raw = tf.read_text("utf-8", errors="replace")
        clean = _strip_html(raw)

        if len(clean.split()) < min_words:
            continue

        results.append((stem, cat, clean))

    if not results:
        pytest.skip(
            "No florex resumes mapped to ATS categories after filtering. "
            f"Checked {len(txt_files)} files."
        )

    return results


def _jd_for_category(category: str) -> str:
    """Generate a realistic job description for a given category."""
    JDS = {
        "INFORMATION-TECHNOLOGY": (
            "Senior Software Engineer — build scalable distributed systems. "
            "Python, Java, or Go. Cloud-native (AWS/GCP), containers (Docker/Kubernetes), "
            "CI/CD. Strong CS fundamentals, system design, and team collaboration."
        ),
        "SALES": (
            "Sales Account Executive — manage enterprise accounts, drive revenue growth. "
            "CRM (Salesforce), pipeline management, negotiation, and closing skills. "
            "B2B sales experience with quarterly target achievement."
        ),
        "FINANCE": (
            "Financial Analyst — financial modeling, forecasting, and reporting. "
            "Excel, Bloomberg, SQL. CFA or MBA preferred. Strong analytical skills "
            "and attention to detail."
        ),
        "HEALTHCARE": (
            "Registered Nurse — patient care, medication administration, care planning. "
            "BLS/ACLS certified. Strong clinical judgment and communication skills. "
            "Electronic health records experience."
        ),
        "BUSINESS-DEVELOPMENT": (
            "Project Manager — lead cross-functional initiatives, manage timelines, "
            "budgets, and stakeholder communication. PMP certification preferred. "
            "Agile/Scrum experience. Strong leadership and risk management."
        ),
        "ENGINEERING": (
            "Mechanical Engineer — product design, CAD modeling (SolidWorks), "
            "FEA analysis, prototyping. BS in Mechanical Engineering. "
            "Experience with manufacturing processes and quality control."
        ),
        "HUMAN-RESOURCES": (
            "HR Manager — talent acquisition, employee relations, performance management, "
            "compliance. SHRM-CP or PHR certification. HRIS experience. "
            "Strong interpersonal and conflict resolution skills."
        ),
    }
    return JDS.get(category, f"Professional role in {category.replace('-', ' ').title()}.")


# ── B-E1: Real resume pipeline ranking ───────────────────────────────


@pytest.mark.usefixtures("real_pipeline")
class TestRealResumeRanking:
    """B-E1 — Rank real human-written resumes through the full pipeline."""

    def test_rank_real_resumes_all_return_valid(self, real_pipeline):
        """All loaded real resumes produce valid CandidateResult objects.

        NOTE: The first call loads SkillNer's 31k-skill DB, adding ~3 min.
        Subsequent calls are much faster. Use max_files=3 for a reasonable runtime.
        """
        resumes = _load_real_resumes(max_files=3, min_words=50)
        if not resumes:
            pytest.skip("No real resumes available for testing")

        for fid, cat, text in resumes:
            jd = _jd_for_category(cat)
            result = real_pipeline.rank_single(
                resume_text=text,
                job_description=jd,
                job_category=cat,
            )
            assert result.final_score >= 0, (
                f"File {fid} ({cat}): final_score {result.final_score} < 0"
            )
            assert result.label in ("Bad Fit", "Potential Fit", "Good Fit"), (
                f"File {fid}: unexpected label '{result.label}'"
            )
            assert result.processing_time > 0, (
                f"File {fid}: zero processing time"
            )

    def test_same_resume_higher_in_own_category(self, real_pipeline):
        """A Software_Developer resume scores higher under IT than under Finance."""
        resumes = _load_real_resumes(max_files=5, min_words=50)
        if not resumes:
            pytest.skip("No real resumes available for testing")

        # Pick first IT resume
        it_resume = None
        for fid, cat, text in resumes:
            if cat == "INFORMATION-TECHNOLOGY":
                it_resume = (fid, text)
                break

        if it_resume is None:
            pytest.skip("No INFORMATION-TECHNOLOGY resume in sample")

        fid, text = it_resume
        it_jd = _jd_for_category("INFORMATION-TECHNOLOGY")
        finance_jd = _jd_for_category("FINANCE")

        it_result = real_pipeline.rank_single(
            resume_text=text,
            job_description=it_jd,
            job_category="INFORMATION-TECHNOLOGY",
        )
        finance_result = real_pipeline.rank_single(
            resume_text=text,
            job_description=finance_jd,
            job_category="FINANCE",
        )
        # IT resume should score higher when category matches
        assert it_result.final_score >= finance_result.final_score, (
            f"IT resume scored {it_result.final_score:.4f} in IT but "
            f"{finance_result.final_score:.4f} in Finance — expected IT to be higher"
        )

    def test_html_stripping_produces_clean_text(self):
        """HTML tags are properly removed from resume text."""
        resumes = _load_real_resumes(max_files=5, min_words=10)
        if not resumes:
            pytest.skip("No real resumes available for testing")

        for fid, cat, text in resumes:
            assert "<span" not in text, f"File {fid}: still contains <span> tag"
            assert "<hl>" not in text, f"File {fid}: still contains <hl> tag"
            assert "class=" not in text, f"File {fid}: still contains class="
            assert len(text) > 0, f"File {fid}: empty after stripping"


# ── B-E2: Cross-category ranking invariance ──────────────────────────


@pytest.mark.usefixtures("real_pipeline")
class TestCrossCategoryRanking:
    """B-E2 — Same resume ranked across different categories behaves predictably."""

    def test_resume_ranked_in_all_categories_no_crash(self, real_pipeline):
        """A single real resume ranked against all 24 categories doesn't crash.

        NOTE: SkillNer loads on the first call (~3 min), so this test runs
        one warm-up ranking first, then iterates through all categories.
        """
        resumes = _load_real_resumes(max_files=2, min_words=50)
        if not resumes:
            pytest.skip("No real resumes available for testing")

        fid, _, text = resumes[0]
        from config import config

        # Warm-up: first ranking triggers SkillNer loading (~3 min)
        _ = real_pipeline.rank_single(
            resume_text=text,
            job_description=_jd_for_category("INFORMATION-TECHNOLOGY"),
            job_category="INFORMATION-TECHNOLOGY",
        )

        categories = list(config["categories"]["name_to_id"].keys())
        jd = _jd_for_category("INFORMATION-TECHNOLOGY")

        for cat in categories:
            result = real_pipeline.rank_single(
                resume_text=text,
                job_description=jd,
                job_category=cat,
            )
            assert result.final_score >= 0, (
                f"Category {cat}: score {result.final_score} < 0"
            )
            assert result.label in ("Bad Fit", "Potential Fit", "Good Fit")


# ── B-E3: PDF text extraction fidelity ───────────────────────────────


class TestPDFExtraction:
    """B-E3 — Text extraction from real PDF files is accurate and complete."""

    PDF_DIR = (
        Path(__file__).resolve().parent.parent.parent
        / "datasets" / "resume_pdfs"
    )

    def test_pdf_directory_exists(self):
        """The PDF dataset directory is accessible."""
        assert self.PDF_DIR.exists(), f"PDF directory not found: {self.PDF_DIR}"

    def test_extract_text_from_real_pdf(self):
        """Extract text from a real synthetic PDF resume and verify content."""
        from data.pdf_processing import PDFTextExtractor

        extractor = PDFTextExtractor()

        cat_dirs = list(self.PDF_DIR.iterdir())
        if not cat_dirs:
            pytest.skip("No category subdirectories in resume_pdfs")

        # Pick a random PDF from the first category
        cat_dir = cat_dirs[0]
        pdfs = list(cat_dir.glob("*.pdf"))
        if not pdfs:
            pytest.skip(f"No PDFs found in {cat_dir}")

        pdf_path = str(pdfs[0])
        text = extractor.extract(pdf_path)
        assert text is not None, f"PDFTextExtractor returned None for {pdf_path}"
        assert len(text.strip()) > 0, f"Empty text extracted from {pdf_path}"

    def test_pdf_text_has_content_words(self):
        """Extracted PDF text contains meaningful content (not just whitespace)."""
        from data.pdf_processing import PDFTextExtractor

        extractor = PDFTextExtractor()

        cat_dirs = list(self.PDF_DIR.iterdir())
        if not cat_dirs:
            pytest.skip("No category subdirectories in resume_pdfs")

        word_counts = []
        for cat_dir in cat_dirs[:3]:  # Sample 3 categories
            pdfs = list(cat_dir.glob("*.pdf"))[:2]  # 2 per category
            for pdf in pdfs:
                text = extractor.extract(str(pdf))
                if text:
                    words = text.strip().split()
                    word_counts.append(len(words))

        if word_counts:
            avg = sum(word_counts) / len(word_counts)
            assert avg > 10, f"Average word count per PDF: {avg:.1f} — suspiciously low"
        else:
            pytest.skip("No PDFs produced extractable text")

    def test_pdf_text_contains_skills(self):
        """Extracted PDF resumes contain skill-like keywords."""
        from data.pdf_processing import PDFTextExtractor

        extractor = PDFTextExtractor()

        cat_dir = self.PDF_DIR / "INFORMATION-TECHNOLOGY"
        if not cat_dir.exists():
            pytest.skip("INFORMATION-TECHNOLOGY PDF directory not found")

        pdfs = list(cat_dir.glob("*.pdf"))
        if not pdfs:
            pytest.skip("No PDFs in INFORMATION-TECHNOLOGY directory")

        tech_keywords = {"python", "java", "sql", "docker", "aws", "react", "javascript"}
        found_keywords = set()
        for pdf in pdfs[:5]:
            text = extractor.extract(str(pdf))
            if text:
                found_keywords.update(
                    kw for kw in tech_keywords if kw in text.lower()
                )

        # At least some tech keywords should be found
        assert len(found_keywords) > 0, (
            f"No tech keywords found in IT PDFs. Checked: {pdfs[:5]}"
        )


# ── B-E4: Full-stack consistency — same text, same score ─────────────


@pytest.mark.usefixtures("real_pipeline")
class TestRankingConsistency:
    """B-E4 — Pipeline produces deterministic scores for identical inputs."""

    def test_three_runs_identical_scores_real_resume(self, real_pipeline):
        """Running the same real resume 3 times yields identical scores."""
        resumes = _load_real_resumes(max_files=3, min_words=50)
        if not resumes:
            pytest.skip("No real resumes available for testing")

        _, cat, text = resumes[0]
        jd = _jd_for_category(cat)

        scores = []
        for _ in range(3):
            result = real_pipeline.rank_single(
                resume_text=text,
                job_description=jd,
                job_category=cat,
            )
            scores.append(result.final_score)

        assert len(set(round(s, 6) for s in scores)) == 1, (
            f"Scores not identical across 3 runs: {scores}"
        )

    def test_candidate_order_invariant(self, real_pipeline):
        """Ranking the same candidates in different orders gives same results by ID."""
        resumes = _load_real_resumes(max_files=6, min_words=50)
        if not resumes:
            pytest.skip("No real resumes available for testing")

        cat = resumes[0][1]
        jd = _jd_for_category(cat)
        candidates = [
            {"id": f"r{i:04d}", "text": text}
            for i, (_, _, text) in enumerate(resumes)
        ]

        # Shuffle and rank twice
        import random

        rng = random.Random(42)
        order_a = list(candidates)
        rng.shuffle(order_a)
        results_a = real_pipeline.rank_candidates(
            job_description=jd, job_category=cat, candidates=order_a,
        )

        order_b = list(reversed(candidates))
        results_b = real_pipeline.rank_candidates(
            job_description=jd, job_category=cat, candidates=order_b,
        )

        # Score for each candidate ID should be the same regardless of input order
        scores_a = {r.candidate_id: r.final_score for r in results_a}
        scores_b = {r.candidate_id: r.final_score for r in results_b}

        mismatches = []
        for cid in scores_a:
            if abs(scores_a[cid] - scores_b.get(cid, -1)) > 1e-6:
                mismatches.append(
                    f"{cid}: {scores_a[cid]:.6f} vs {scores_b[cid]:.6f}"
                )
        assert len(mismatches) == 0, (
            f"Score differences due to input order:\n" + "\n".join(mismatches)
        )
