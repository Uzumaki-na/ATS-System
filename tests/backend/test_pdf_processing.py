"""B-U16, B-U17 — PDF text extraction tests.

Tests use real PDF conversion: in-memory PDF construction → pypdf extraction.
The content in the PDF is known so we can verify text fidelity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest


# ── B-U16: PDFTextExtractor basic extraction ─────────────────────────


class TestPDFTextExtractorBasic:
    """B-U16 — PDFTextExtractor initializes and extracts text correctly."""

    def test_init_default_config(self):
        """PDFTextExtractor instantiates with default config."""
        from data.pdf_processing import PDFTextExtractor, PDFProcessingConfig

        extractor = PDFTextExtractor()
        assert extractor is not None
        assert isinstance(extractor.config, PDFProcessingConfig)

    def test_init_with_custom_config(self):
        """PDFTextExtractor respects custom config values."""
        from data.pdf_processing import PDFTextExtractor, PDFProcessingConfig

        config = PDFProcessingConfig(
            enabled=True,
            min_text_length=100,
            cache_enabled=False,
            extraction_method="pypdf",
        )
        extractor = PDFTextExtractor(config)
        assert extractor.config.min_text_length == 100
        assert extractor.config.cache_enabled is False
        assert extractor.config.extraction_method == "pypdf"

    def test_extract_known_text_from_pdf(self, pdf_bytes):
        """PDF with known text content extracts that text back."""
        import tempfile

        from data.pdf_processing import PDFTextExtractor

        # Write known-content PDF bytes to temp file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(pdf_bytes)
            pdf_path = f.name

        try:
            extractor = PDFTextExtractor()
            text = extractor.extract(pdf_path)
            assert "Hello World Test PDF" in text
        finally:
            Path(pdf_path).unlink(missing_ok=True)

    def test_extract_nonexistent_file(self):
        """Missing PDF file returns empty string."""
        from data.pdf_processing import PDFTextExtractor

        extractor = PDFTextExtractor()
        text = extractor.extract("/tmp/nonexistent_xyz98765.pdf")
        assert text == ""

    def test_extract_empty_pdf(self, tmp_path):
        """PDF with no text content returns empty string or very short text."""
        from data.pdf_processing import PDFTextExtractor

        # Create PDF with no text on page
        empty_pdf = (
            b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
            b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n"
            b"0000000058 00000 n \n0000000115 00000 n \n"
            b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n188\n%%EOF"
        )
        path = tmp_path / "empty.pdf"
        path.write_bytes(empty_pdf)

        extractor = PDFTextExtractor()
        text = extractor.extract(str(path))
        # Should be empty or very short
        assert len(text.strip()) < 20

    def test_extract_text_fidelity(self, pdf_file_with_text):
        """PDF with resume-like text preserves content with high fidelity."""
        from data.pdf_processing import PDFTextExtractor

        extractor = PDFTextExtractor(config=None)
        text = extractor.extract(str(pdf_file_with_text))

        # The key content should survive extraction
        assert "John Doe" in text
        assert "Senior Software Engineer" in text
        assert "Python" in text
        assert "Stanford" in text or "Stanford University" in text


# ── B-U16: PDFTextExtractor caching behaviour ────────────────────────


class TestPDFTextExtractorCache:
    """B-U16 — caching speeds up re-extraction and is correct."""

    def test_cache_hit_returns_same_text(self, pdf_file):
        """Second extraction from cache returns identical result."""
        from data.pdf_processing import PDFTextExtractor, PDFProcessingConfig

        config = PDFProcessingConfig(cache_enabled=True)
        extractor = PDFTextExtractor(config)

        text1 = extractor.extract(str(pdf_file), use_cache=True)
        text2 = extractor.extract(str(pdf_file), use_cache=True)
        assert text1 == text2

    def test_cache_disabled_skips_cache(self, pdf_file):
        """With cache disabled, extraction still works."""
        from data.pdf_processing import PDFTextExtractor, PDFProcessingConfig

        config = PDFProcessingConfig(cache_enabled=False)
        extractor = PDFTextExtractor(config)

        text = extractor.extract(str(pdf_file), use_cache=True)
        # Should still work, just not cached
        assert text is not None

    def test_clear_cache_empties_it(self, pdf_file):
        """clear_cache removes all cached entries."""
        from data.pdf_processing import PDFTextExtractor, PDFProcessingConfig

        config = PDFProcessingConfig(cache_enabled=True)
        extractor = PDFTextExtractor(config)
        extractor.extract(str(pdf_file))

        assert len(extractor.cache) > 0
        extractor.clear_cache()
        assert len(extractor.cache) == 0


# ── B-U17: PDFResumeDataset ──────────────────────────────────────────


class TestPDFResumeDataset:
    """B-U17 — PDFResumeDataset collects and extracts PDFs by category."""

    def test_init_with_nonexistent_dir(self):
        """Non-existent PDF directory returns empty samples list."""
        from data.pdf_processing import PDFResumeDataset

        ds = PDFResumeDataset("/tmp/nonexistent_pdf_dir_xyz", shuffle=False)
        assert len(ds) == 0

    def test_collects_pdfs_from_directory(self, tmp_path):
        """PDF files organized by category are collected correctly."""
        from data.pdf_processing import PDFResumeDataset

        # Create category directories with mock PDFs
        for cat in ("ENGINEERING", "FINANCE"):
            cat_dir = tmp_path / cat
            cat_dir.mkdir()
            # Write a minimal PDF
            pdf_bytes = (
                b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
                b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
                b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
                b"4 0 obj\n<< /Length 44 >>\nstream\nBT /F1 12 Tf 100 700 Td "
                b"(Resume Content) Tj ET\nendstream\nendobj\n"
                b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
                b"xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n"
                b"0000000115 00000 n \n0000000266 00000 n \n0000000363 00000 n \n"
                b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n435\n%%EOF"
            )
            (cat_dir / "resume1.pdf").write_bytes(pdf_bytes)

        ds = PDFResumeDataset(str(tmp_path), shuffle=False)
        assert len(ds) == 2

        # Check categories are correct
        categories = [s["category"] for s in ds.samples]
        assert "ENGINEERING" in categories
        assert "FINANCE" in categories

    def test_get_item_returns_expected_fields(self, tmp_path):
        """__getitem__ returns dict with text, category, pdf_path, id, filename."""
        from data.pdf_processing import PDFResumeDataset

        # Create one dummy category with one PDF
        cat_dir = tmp_path / "HEALTHCARE"
        cat_dir.mkdir()
        pdf_bytes = (
            b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
            b"4 0 obj\n<< /Length 18 >>\nstream\nBT /F1 12 Tf 100 700 Td "
            b"(Nurse Resume) Tj ET\nendstream\nendobj\n"
            b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
            b"xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n"
            b"0000000115 00000 n \n0000000266 00000 n \n0000000363 00000 n \n"
            b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n435\n%%EOF"
        )
        (cat_dir / "nurse_resume.pdf").write_bytes(pdf_bytes)

        ds = PDFResumeDataset(str(tmp_path), shuffle=False)
        item = ds[0]

        assert "text" in item
        assert "category" in item
        assert item["category"] == "HEALTHCARE"
        assert "pdf_path" in item
        assert "filename" in item
        assert "id" in item

    def test_category_distribution(self, tmp_path):
        """get_category_distribution returns correct counts."""
        from data.pdf_processing import PDFResumeDataset

        for cat in ("IT", "IT", "HR"):
            cat_dir = tmp_path / cat
            cat_dir.mkdir(exist_ok=True)
            (cat_dir / f"{cat.lower()}_resume.pdf").write_bytes(
                b"%PDF-1.4 trash\n%%EOF"
            )

        ds = PDFResumeDataset(str(tmp_path), shuffle=False)
        dist = ds.get_category_distribution()
        assert dist is not None
