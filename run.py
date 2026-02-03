import csv
import json
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import config
from data.loaders import clean_resume_text, validate_category
from data.pdf_processing import PDFProcessingConfig, PDFTextExtractor


@dataclass
class RunConfig:
    resume_dir: str
    jd_file: str
    job_category: str
    mode: str  # "local" | "api"
    api_url: str
    device: str
    pipeline_top_k: int
    top_k: int
    min_chars: int
    output_format: str  # "csv" | "json"
    output_path: str


def _normalize_user_path(raw: str) -> str:
    s = (raw or "").strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in {"\"", "'"}:
        s = s[1:-1].strip()
    return s


def _resolve_existing_path(raw: str) -> Path:
    s = _normalize_user_path(raw)
    p = Path(s)
    if p.exists():
        return p
    if not p.is_absolute():
        alt = PROJECT_ROOT / p
        if alt.exists():
            return alt
    return p


def _prompt(text: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        v = _normalize_user_path(input(f"{text}{suffix}: "))
        if v:
            return v
        if default is not None:
            return default


def _prompt_int(text: str, default: int, min_value: int = 1) -> int:
    while True:
        raw = _prompt(text, str(default)).strip()
        try:
            v = int(raw)
        except ValueError:
            print("Please enter a valid integer.")
            continue
        if v < min_value:
            print(f"Please enter a value >= {min_value}.")
            continue
        return v


def _prompt_choice(text: str, choices: List[str], default: Optional[str] = None) -> str:
    choice_set = {c.lower(): c for c in choices}
    while True:
        raw = _prompt(text, default).strip()
        key = raw.lower()
        if key in choice_set:
            return choice_set[key]
        print(f"Invalid choice. Allowed: {', '.join(choices)}")


def _list_categories() -> List[str]:
    id_to_name = config.get("categories", {}).get("id_to_name", {})
    ordered: List[Tuple[int, str]] = []
    for k, v in id_to_name.items():
        try:
            ordered.append((int(k), str(v)))
        except Exception:
            continue
    ordered.sort(key=lambda x: x[0])
    return [name for _, name in ordered]


def _prompt_category(default: Optional[str] = None) -> str:
    categories = _list_categories()
    print("\nAvailable categories:")
    for idx, cat in enumerate(categories, 1):
        print(f"  {idx:2d}. {cat}")

    while True:
        raw = _prompt("Enter job category (name or number)", default).strip()
        if not raw:
            continue

        if raw.isdigit():
            i = int(raw)
            if 1 <= i <= len(categories):
                return categories[i - 1]
            print("Invalid number.")
            continue

        cat = validate_category(raw)
        if cat in {validate_category(c) for c in categories}:
            return cat

        print("Unknown category. Please choose from the list.")


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _iter_pdf_paths(resume_dir: str) -> List[Path]:
    root = _resolve_existing_path(resume_dir)
    if not root.exists() or not root.is_dir():
        hint = ""
        # Common dataset layout hint for this repo
        try:
            category_token = validate_category(Path(_normalize_user_path(resume_dir)).name)
        except Exception:
            category_token = ""

        dataset_base = PROJECT_ROOT / "datasets" / "category resumes" / "data" / "data"
        if dataset_base.exists() and dataset_base.is_dir():
            if category_token:
                candidate = dataset_base / category_token
                if candidate.exists() and candidate.is_dir():
                    hint = f"\nDid you mean: {candidate}?"
                else:
                    hint = f"\nDataset PDFs are under: {dataset_base}\\<CATEGORY>"
            else:
                hint = f"\nDataset PDFs are under: {dataset_base}\\<CATEGORY>"

        raise FileNotFoundError(
            f"Resume directory not found or not a directory: {root}{hint}"
        )

    pdfs = sorted(p for p in root.rglob("*.pdf") if p.is_file())
    return pdfs


def _make_candidate_id(resume_root: Path, pdf_path: Path) -> str:
    try:
        rel = pdf_path.relative_to(resume_root)
        rel_no_suffix = rel.with_suffix("")
        return rel_no_suffix.as_posix()
    except Exception:
        return pdf_path.stem


def _extract_candidates(
    resume_dir: str,
    min_chars: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, Any]]:
    pdf_cache_file = config.get("data", {}).get("processed", {}).get(
        "pdf_text_cache", "data/processed/pdf_text_cache.json"
    )
    pdf_cfg = PDFProcessingConfig(cache_enabled=True, cache_file=pdf_cache_file)
    extractor = PDFTextExtractor(pdf_cfg)

    resume_root = Path(resume_dir)
    pdf_paths = _iter_pdf_paths(resume_dir)

    candidates: List[Dict[str, Any]] = []
    id_to_path: Dict[str, str] = {}

    n_failed = 0
    n_too_short = 0

    for pdf_path in pdf_paths:
        try:
            raw_text = extractor.extract(str(pdf_path))
        except Exception:
            n_failed += 1
            continue

        cleaned = clean_resume_text(raw_text)
        if not cleaned or len(cleaned) < min_chars:
            n_too_short += 1
            continue

        cid = _make_candidate_id(resume_root, pdf_path)
        candidates.append({"id": cid, "text": cleaned})
        id_to_path[cid] = str(pdf_path)

    stats = {
        "pdfs_found": len(pdf_paths),
        "candidates_extracted": len(candidates),
        "pdfs_failed": n_failed,
        "pdfs_too_short": n_too_short,
    }
    return candidates, id_to_path, stats


def _post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=600) as resp:
        data = resp.read().decode("utf-8")
        return json.loads(data)


def _rank_via_api(
    api_url: str,
    job_description: str,
    job_category: str,
    candidates: List[Dict[str, Any]],
    pipeline_top_k: int,
) -> List[Dict[str, Any]]:
    payload = {
        "job_description": job_description,
        "job_category": validate_category(job_category),
        "candidates": candidates,
        "top_k": int(pipeline_top_k),
    }
    resp = _post_json(api_url, payload)
    return resp.get("results") or []


def _rank_locally(
    job_description: str,
    job_category: str,
    candidates: List[Dict[str, Any]],
    device: str,
    pipeline_top_k: int,
) -> List[Dict[str, Any]]:
    from pipeline.inference import TriadRankPipeline

    model1_path = config.get("models", {}).get("cross_encoder", {}).get("save_path")
    model2_path = config.get("models", {}).get("category_encoder", {}).get("save_path")
    model3_spacy_model = config.get("models", {}).get("extractor", {}).get("spacy_model", "en_core_web_sm")

    pipeline = TriadRankPipeline(
        model1_path=model1_path,
        model2_path=model2_path,
        model3_spacy_model=model3_spacy_model,
        device=device,
        top_k=max(1, int(pipeline_top_k)),
        category_penalty=config.get("penalties", {}).get("category_mismatch", 0.5),
    )

    results = pipeline.rank_candidates(
        job_description=job_description,
        job_category=job_category,
        candidates=candidates,
        top_k=int(pipeline_top_k),
    )

    out: List[Dict[str, Any]] = []
    for rank, r in enumerate(results, 1):
        out.append(
            {
                "rank": rank,
                "candidate_id": r.candidate_id,
                "final_score": round(float(r.final_score), 6),
                "raw_score": round(float(r.raw_score), 6),
                "label": r.label,
                "label_probabilities": r.label_probabilities,
                "category": {
                    "predicted": r.category_predicted,
                    "match": r.category_match,
                    "confidence": round(float(r.category_confidence), 6),
                },
                "skill_overlap": round(float(r.skill_overlap), 6),
                "keyword_overlap": round(float(r.keyword_overlap), 6),
                "processing_time": round(float(r.processing_time), 6),
                "extracted_skills": (r.metadata or {}).get("extracted_skills", []),
            }
        )

    return out


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return

    fieldnames: List[str] = [
        "rank",
        "candidate_id",
        "file_path",
        "final_score",
        "raw_score",
        "label",
        "p_good_fit",
        "p_potential_fit",
        "p_bad_fit",
        "category_predicted",
        "category_match",
        "category_confidence",
        "skill_overlap",
        "keyword_overlap",
        "processing_time",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            lp = r.get("label_probabilities") or {}
            cat = r.get("category") or {}
            writer.writerow(
                {
                    "rank": r.get("rank"),
                    "candidate_id": r.get("candidate_id"),
                    "file_path": r.get("file_path"),
                    "final_score": r.get("final_score"),
                    "raw_score": r.get("raw_score"),
                    "label": r.get("label"),
                    "p_good_fit": lp.get("good_fit", lp.get("Good Fit", "")),
                    "p_potential_fit": lp.get("potential_fit", lp.get("Potential Fit", "")),
                    "p_bad_fit": lp.get("bad_fit", lp.get("Bad Fit", "")),
                    "category_predicted": cat.get("predicted"),
                    "category_match": cat.get("match"),
                    "category_confidence": cat.get("confidence"),
                    "skill_overlap": r.get("skill_overlap"),
                    "keyword_overlap": r.get("keyword_overlap"),
                    "processing_time": r.get("processing_time"),
                }
            )


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _collect_run_config(prev: Optional[RunConfig]) -> RunConfig:
    print("\nWelcome to TriadRank Resume Ranker")
    print("--------------------------------")

    resume_dir = _prompt("Enter resume folder directory (PDFs will be scanned recursively)", prev.resume_dir if prev else None)
    jd_file = _prompt("Enter job description .txt file path", prev.jd_file if prev else None)

    default_cat = prev.job_category if prev else None
    job_category = _prompt_category(default_cat)

    mode = _prompt_choice(
        "Choose mode (local or api)",
        choices=["local", "api"],
        default=(prev.mode if prev else "local"),
    )

    api_url_default = prev.api_url if prev else "http://127.0.0.1:8000/rank"
    api_url = api_url_default
    if mode == "api":
        api_url = _prompt("Enter API URL", api_url_default)

    device_default = prev.device if prev else config.get("inference", {}).get("device", "auto")
    device = device_default
    if mode == "local":
        device = _prompt("Device (cuda/cpu/auto)", device_default)

    min_chars_default = prev.min_chars if prev else 400
    min_chars = _prompt_int("Minimum extracted chars required per resume", min_chars_default, min_value=1)

    pipeline_top_k_default = prev.pipeline_top_k if prev else int(config.get("inference", {}).get("top_k", 50))
    pipeline_top_k = _prompt_int("Pipeline top_k (how many resumes to process internally)", pipeline_top_k_default, min_value=1)

    top_k_default = prev.top_k if prev else 50
    top_k = _prompt_int("Display top_k (how many resumes to output)", top_k_default, min_value=1)

    if top_k > pipeline_top_k:
        pipeline_top_k = top_k

    output_format = _prompt_choice(
        "Output format (csv or json)",
        choices=["csv", "json"],
        default=(prev.output_format if prev else "csv"),
    )

    ts = int(time.time())
    default_out = prev.output_path if prev else f"reports/run_rank_{ts}.{output_format}"
    output_path = _prompt("Output path", default_out)

    return RunConfig(
        resume_dir=resume_dir,
        jd_file=jd_file,
        job_category=job_category,
        mode=mode,
        api_url=api_url,
        device=device,
        pipeline_top_k=int(pipeline_top_k),
        top_k=int(top_k),
        min_chars=int(min_chars),
        output_format=output_format,
        output_path=output_path,
    )


def _run_once(cfg: RunConfig) -> int:
    jd_path = _resolve_existing_path(cfg.jd_file)
    if not jd_path.exists() or not jd_path.is_file():
        print(f"Job description file not found: {cfg.jd_file}")
        return 1

    job_description = _read_text_file(str(jd_path)).strip()
    if not job_description:
        print("Job description file is empty.")
        return 1

    print("\nScanning resumes...")
    candidates, id_to_path, stats = _extract_candidates(cfg.resume_dir, cfg.min_chars)
    print(f"PDFs found: {stats['pdfs_found']}")
    print(f"Candidates extracted: {stats['candidates_extracted']}")
    print(f"PDFs too short: {stats['pdfs_too_short']}")
    print(f"PDFs failed: {stats['pdfs_failed']}")

    if not candidates:
        print("No valid resumes extracted. Try lowering minimum chars or check PDFs.")
        return 1

    if cfg.pipeline_top_k > len(candidates):
        print(
            f"Warning: pipeline_top_k={cfg.pipeline_top_k} is larger than extracted resumes={len(candidates)}. "
            "Clamping to total resumes."
        )
        cfg.pipeline_top_k = len(candidates)

    if cfg.top_k > len(candidates):
        print(
            f"Warning: top_k={cfg.top_k} is larger than extracted resumes={len(candidates)}. "
            "Clamping to total resumes."
        )
        cfg.top_k = len(candidates)

    if cfg.top_k > cfg.pipeline_top_k:
        cfg.pipeline_top_k = cfg.top_k

    print("\nRanking...")
    t0 = time.time()
    try:
        if cfg.mode == "api":
            results = _rank_via_api(
                api_url=cfg.api_url,
                job_description=job_description,
                job_category=cfg.job_category,
                candidates=candidates,
                pipeline_top_k=cfg.pipeline_top_k,
            )
        else:
            results = _rank_locally(
                job_description=job_description,
                job_category=cfg.job_category,
                candidates=candidates,
                device=cfg.device,
                pipeline_top_k=cfg.pipeline_top_k,
            )
    except Exception as e:
        print(f"Ranking failed: {e}")
        return 1

    elapsed = time.time() - t0

    if not isinstance(results, list):
        print("Unexpected result format from ranker.")
        return 1

    results = results[: cfg.top_k]

    for r in results:
        cid = r.get("candidate_id")
        if cid in id_to_path:
            r["file_path"] = id_to_path[cid]

    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "input": {
            "resume_dir": cfg.resume_dir,
            "jd_file": cfg.jd_file,
            "job_category": validate_category(cfg.job_category),
            "mode": cfg.mode,
            "api_url": cfg.api_url if cfg.mode == "api" else "",
            "device": cfg.device if cfg.mode == "local" else "",
            "pipeline_top_k": cfg.pipeline_top_k,
            "top_k": cfg.top_k,
            "min_chars": cfg.min_chars,
        },
        "stats": stats,
        "runtime_seconds": round(float(elapsed), 4),
        "results": results,
    }

    if cfg.output_format == "csv":
        _write_csv(str(output_path), results)
    else:
        _write_json(str(output_path), payload)

    print(f"\nDone. Wrote: {output_path}")
    print(f"Returned top_k: {cfg.top_k}")
    print(f"Runtime: {elapsed:.2f}s")

    return 0


def main() -> int:
    prev: Optional[RunConfig] = None

    while True:
        cfg = _collect_run_config(prev)
        rc = _run_once(cfg)

        prev = cfg

        print("\nWhat next?")
        next_action = _prompt_choice(
            "Enter choice",
            choices=["rerun", "rerun_keep", "exit"],
            default="exit",
        )

        if next_action == "exit":
            return rc

        if next_action == "rerun_keep":
            rc = _run_once(prev)
            continue

        # rerun: loop, but keep prev as defaults


if __name__ == "__main__":
    raise SystemExit(main())
