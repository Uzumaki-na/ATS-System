"""
Build florex external dataset in CategoryDataset CSV format.

Florex: ~29,783 real resumes at datasets/external/resume_corpus/resumes_corpus/
- Each has {id}.lab (occupation label) and {id}.txt (resume text)
- 10 occupation labels → 2 ATS categories (INFORMATION-TECHNOLOGY, BUSINESS-DEVELOPMENT)
- 57% are multi-label (multiple occupations per resume)

Usage:
    python scripts/build_florex_data.py
    python scripts/build_florex_data.py --max_per_category 1000
    python scripts/build_florex_data.py --max_per_category 0  # include all

Output: datasets/category_resumes/florex_resumes.csv
"""

import os
import sys
import csv
import random
import logging
from pathlib import Path
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Florex occupation → our 24 ATS categories
# ---------------------------------------------------------------------------
FLOREX_TO_CATEGORY = {
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
}


def clean_resume_text(text: str) -> str:
    """Clean florex resume text — strip HTML, normalize whitespace."""
    import re
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    if len(text) < 50:
        return ""
    return text


def build_florex_csv(
    florex_dir: str = "datasets/external/resume_corpus/resumes_corpus",
    output_path: str = "datasets/category_resumes/florex_resumes.csv",
    min_text_length: int = 50,
    max_per_category: int = 500,
    seed: int = 42,
):
    """
    Build a CategoryDataset-compatible CSV from florex data.

    Args:
        florex_dir: Path to florex .lab / .txt files
        output_path: Output CSV path
        min_text_length: Minimum resume text length to include
        max_per_category: Max florex samples per category (0 = unlimited)
        seed: Random seed for reproducible sampling
    """
    random.seed(seed)
    florex_dir = Path(florex_dir)
    if not florex_dir.exists():
        logger.error(f"Florex directory not found: {florex_dir}")
        return

    lab_files = sorted(florex_dir.glob("*.lab"))
    logger.info(f"Found {len(lab_files)} florex lab files")

    # First pass: collect valid samples per category without capping
    category_pool = {}  # ats_category -> list of row dicts
    skipped_no_mapping = 0
    skipped_no_text = 0
    skipped_short = 0
    multi_label_count = 0

    for lab_path in lab_files:
        stem = lab_path.stem

        labels = [
            l.strip()
            for l in lab_path.read_text("utf-8", errors="replace").split("\n")
            if l.strip()
        ]

        ats_categories = set()
        for label in labels:
            mapped = FLOREX_TO_CATEGORY.get(label)
            if mapped:
                ats_categories.add(mapped)

        if not ats_categories:
            skipped_no_mapping += 1
            continue

        txt_path = florex_dir / f"{stem}.txt"
        if not txt_path.exists():
            skipped_no_text += 1
            continue

        raw_text = txt_path.read_text("utf-8", errors="replace")
        clean_text = clean_resume_text(raw_text)

        if len(clean_text) < min_text_length:
            skipped_short += 1
            continue

        if len(ats_categories) > 1:
            multi_label_count += 1

        for cat in sorted(ats_categories):
            row = {
                "ID": f"florex_{stem}",
                "Resume_str": clean_text,
                "Resume_html": "",
                "Category": cat,
            }
            category_pool.setdefault(cat, []).append(row)

    # Second pass: cap per category, shuffle to get a random sample
    rows = []
    stats = Counter()
    for cat, pool in sorted(category_pool.items()):
        random.shuffle(pool)
        if max_per_category > 0 and len(pool) > max_per_category:
            pool = pool[:max_per_category]
            logger.info(f"  Capped {cat}: {len(category_pool[cat])} → {len(pool)}")
        rows.extend(pool)
        stats[cat] += len(pool)

    # Shuffle final rows so training batches see interleaved categories
    random.shuffle(rows)

    # Write CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Resume_str", "Resume_html", "Category"])
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Florex CSV Build Complete")
    logger.info(f"{'='*60}")
    logger.info(f"Total florex resumes: {len(lab_files)}")
    logger.info(f"Rows written: {len(rows)}")
    logger.info(f"Max per category: {'unlimited' if max_per_category == 0 else max_per_category}")
    logger.info(f"Multi-label (mapped to >1 ATS cat): {multi_label_count}")
    logger.info(f"Skipped — no mapping: {skipped_no_mapping}")
    logger.info(f"Skipped — no text file: {skipped_no_text}")
    logger.info(f"Skipped — text too short: {skipped_short}")
    logger.info(f"\nCategory distribution:")
    for cat, count in stats.most_common():
        logger.info(f"  {cat}: {count}")
    logger.info(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build florex dataset CSV")
    parser.add_argument("--max_per_category", type=int, default=500,
                        help="Max samples per category (0 = unlimited)")
    args = parser.parse_args()
    build_florex_csv(max_per_category=args.max_per_category)
