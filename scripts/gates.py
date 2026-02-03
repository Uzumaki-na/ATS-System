#!/usr/bin/env python3
"""
Production readiness quality gates for cross-domain ATS ranking.
Run this on the CSV output of eval_rank_pdfs.py --matrix.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def load_rows(csv_path: Path) -> List[Dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def per_row_gates(row: Dict) -> Tuple[bool, List[str]]:
    failures = []
    # Near/far-specific gates
    tag = row.get("tag", "")
    match_rate = float(row.get("match_rate", 0.0))
    job_in_topn = int(row.get("n_job_in_topn", 0))
    distractor_in_topn = int(row.get("n_distractor_in_topn", 0))
    gap = float(row.get("mean_final_score_gap", 0.0))

    if tag == "near":
        if match_rate < 0.60:
            failures.append(f"near match_rate {match_rate:.3f} < 0.60")
        if job_in_topn < 12:
            failures.append(f"near n_job_in_topn {job_in_topn} < 12")
        if distractor_in_topn > 8:
            failures.append(f"near n_distractor_in_topn {distractor_in_topn} > 8")
        if gap < 0.20:
            failures.append(f"near mean_final_score_gap {gap:.3f} < 0.20")
    elif tag == "far":
        if match_rate < 0.80:
            failures.append(f"far match_rate {match_rate:.3f} < 0.80")
        if job_in_topn < 16:
            failures.append(f"far n_job_in_topn {job_in_topn} < 16")
        if distractor_in_topn > 4:
            failures.append(f"far n_distractor_in_topn {distractor_in_topn} > 4")
        if gap < 0.18:
            failures.append(f"far mean_final_score_gap {gap:.3f} < 0.18")
    else:
        failures.append(f"unknown tag: {tag}")

    return len(failures) == 0, failures


def aggregate_gates(rows: List[Dict]) -> Tuple[bool, List[str]]:
    # Overall gates
    avg_match = sum(float(r.get("match_rate", 0.0)) for r in rows) / float(len(rows))
    near_rows = [r for r in rows if r.get("tag") == "near"]
    far_rows = [r for r in rows if r.get("tag") == "far"]
    avg_near_match = sum(float(r.get("match_rate", 0.0)) for r in near_rows) / float(len(near_rows))
    avg_far_match = sum(float(r.get("match_rate", 0.0)) for r in far_rows) / float(len(far_rows))

    failures = []
    if avg_match < 0.70:
        failures.append(f"avg_match_rate {avg_match:.3f} < 0.70")
    if avg_near_match < 0.60:
        failures.append(f"avg_near_match_rate {avg_near_match:.3f} < 0.60")
    if avg_far_match < 0.75:
        failures.append(f"avg_far_match_rate {avg_far_match:.3f} < 0.75")
    return len(failures) == 0, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path, help="CSV from eval_rank_pdfs.py --matrix")
    args = parser.parse_args()

    if not args.csv_path.exists() or not args.csv_path.is_file():
        print(f"Error: {args.csv_path} does not exist or is not a file.")
        return 1

    rows = load_rows(args.csv_path)
    if not rows:
        print("No rows found in CSV.")
        return 1

    ok = True
    for row in rows:
        row_ok, row_fails = per_row_gates(row)
        if not row_ok:
            ok = False
            print(f"[FAIL] {row.get('job_category')} vs {row.get('distractor_category')} ({row.get('tag')})")
            for f in row_fails:
                print(f"   - {f}")
        else:
            print(f"[PASS] {row.get('job_category')} vs {row.get('distractor_category')} ({row.get('tag')})")

    agg_ok, agg_fails = aggregate_gates(rows)
    if not agg_ok:
        ok = False
        print("\n[FAIL] Aggregate gates")
        for f in agg_fails:
            print(f"   - {f}")
    else:
        print("\n[PASS] Aggregate gates")

    print("\nSummary:")
    print(f"  Total tests: {len(rows)}")
    print(f"  Passed: {sum(1 for r in rows if per_row_gates(r)[0])}")
    print(f"  Failed: {sum(1 for r in rows if not per_row_gates(r)[0])}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
