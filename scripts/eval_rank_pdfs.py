import argparse
import csv
import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

try:
    from sklearn.metrics import ndcg_score
except Exception:
    ndcg_score = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import config
from data.loaders import clean_resume_text, validate_category
from data.pdf_processing import PDFProcessingConfig, PDFTextExtractor


def _default_accountant_jd() -> str:
    return (
        "We are hiring a Senior Accountant to own the monthly close and deliver accurate financial statements. "
        "Responsibilities include general ledger accounting, journal entries, accruals and prepaids, balance sheet "
        "reconciliations, AP/AR review, fixed assets and depreciation, variance analysis, and supporting external audit. "
        "Requirements: 3+ years accounting experience, strong Excel (pivot tables, VLOOKUP/XLOOKUP), knowledge of "
        "GAAP/IFRS, experience with ERP systems (SAP/Oracle/NetSuite) and/or QuickBooks, and strong attention to detail."
    )


def _default_jd_for_category(category: str) -> str:
    cat = validate_category(category)
    if cat == "ACCOUNTANT":
        return _default_accountant_jd()
    if cat == "FINANCE":
        return (
            "We are hiring a Finance (FP&A) Analyst to support budgeting, forecasting, and monthly management reporting for a fast-growing business. "
            "Responsibilities include building and maintaining financial models (revenue, headcount, margin), performing variance analysis vs. budget/forecast, "
            "supporting month-end close with accrual review, preparing executive-ready dashboards, and partnering with cross-functional teams (Sales, Marketing, Operations) "
            "to translate business drivers into financial outcomes. Requirements: 2+ years FP&A/analytics experience, advanced Excel (pivot tables, Power Query), "
            "experience with BI tools (Power BI/Tableau) preferred, strong communication, and ability to work with large datasets."
        )
    if cat == "HUMAN-RESOURCES":
        return (
            "We are hiring an HR Generalist to support recruiting, onboarding, employee relations, and day-to-day HR operations. "
            "Responsibilities include coordinating interview loops, drafting offer letters, managing onboarding and HR documentation, handling employee queries, "
            "supporting performance management cycles, and assisting with payroll/benefits administration. You will also help maintain HRIS records, ensure compliance "
            "with local labor regulations, and partner with managers to resolve employee relations issues. Requirements: 2+ years HR experience, strong communication, "
            "experience with HRIS/ATS tools, and high discretion with confidential information."
        )
    if cat == "SALES":
        return (
            "We are hiring an Account Executive (B2B SaaS) to manage a full-cycle sales process from prospecting to close. "
            "Responsibilities include outbound prospecting, discovery calls, product demos, proposal/contract negotiation, forecasting, and maintaining accurate CRM hygiene. "
            "You will work closely with Marketing and Customer Success to drive pipeline, reduce churn risk at handoff, and hit monthly/quarterly quota. "
            "Requirements: 2+ years B2B sales experience, strong written/verbal communication, comfort with cold outreach, experience with CRMs (Salesforce/HubSpot), "
            "and a track record of meeting targets."
        )
    if cat == "BUSINESS-DEVELOPMENT":
        return (
            "We are hiring a Business Development Manager to drive strategic partnerships and new client acquisition. "
            "Responsibilities include identifying and qualifying partnership opportunities, mapping stakeholders, developing value propositions, preparing proposals and pitch decks, "
            "negotiating commercial terms, and coordinating with internal delivery/product teams to ensure successful execution. "
            "Requirements: 3+ years in BD/partnerships or consultative sales, strong negotiation and presentation skills, ability to run structured outreach, and comfort with market research and GTM planning."
        )
    if cat == "INFORMATION-TECHNOLOGY":
        return (
            "We are hiring an IT Support Engineer to provide L1/L2 support across endpoints, identity, and SaaS applications. "
            "Responsibilities include troubleshooting Windows/macOS issues, managing user access (Active Directory/Okta/Azure AD), supporting email/collaboration tools (Microsoft 365/Google Workspace), "
            "basic networking (DNS/DHCP/VPN), ticket triage and incident response, imaging/provisioning laptops, and maintaining runbooks/documentation. "
            "Requirements: 2+ years IT support experience, strong troubleshooting mindset, familiarity with ITSM tools (Jira/ServiceNow), and scripting basics (PowerShell/Bash) preferred."
        )
    if cat == "ENGINEERING":
        return (
            "We are hiring a Software Engineer to build and maintain production services that power our core product. "
            "Responsibilities include designing APIs and backend components, writing clean and testable code, reviewing PRs, debugging production issues, and improving system reliability and performance. "
            "You will collaborate with Product and QA, participate in sprint planning, and contribute to technical design docs. "
            "Requirements: strong CS fundamentals, experience with Python/Java/Node (any one), REST APIs, SQL, git, and familiarity with cloud (AWS/Azure/GCP) and CI/CD is a plus."
        )
    if cat == "HEALTHCARE":
        return (
            "We are hiring a Clinical Coordinator / Medical Assistant to support patient flow and clinical operations in an outpatient setting. "
            "Responsibilities include patient intake (vitals, history), appointment scheduling support, maintaining accurate EMR/EHR documentation, assisting clinicians during procedures, "
            "ensuring infection control protocols, and coordinating lab/imaging referrals. Requirements: prior clinic experience preferred, strong attention to detail, clear communication, and comfort with EMR systems."
        )
    if cat == "TEACHER":
        return (
            "We are hiring a Teacher to plan and deliver instruction aligned with curriculum standards and student learning goals. "
            "Responsibilities include lesson planning, differentiated instruction, classroom management, designing assessments, tracking student progress, and communicating with parents/guardians. "
            "You will collaborate with other educators, participate in professional development, and support student well-being. Requirements: teaching experience (K-12 or equivalent), strong organization, and effective communication."
        )
    if cat == "DESIGNER":
        return (
            "We are hiring a Graphic / Product Designer to create high-quality visuals for marketing and product experiences. "
            "Responsibilities include designing social/media assets, landing pages, presentations, and product UI components, ensuring brand consistency, and collaborating with Marketing/Product on creative briefs. "
            "Requirements: strong portfolio, proficiency with Figma and Adobe Creative Suite (Photoshop/Illustrator), understanding of typography/layout, and ability to iterate quickly based on feedback."
        )
    return _default_accountant_jd()


def _calculate_ndcg(results_json: List[Dict], k: Optional[int] = None) -> float:
    if ndcg_score is None or not results_json:
        return 0.0

    predicted_scores = [float(r.get("final_score", 0.0) or 0.0) for r in results_json]
    if not predicted_scores:
        return 0.0

    def _get_prob(d: Dict, a: str, b: str) -> float:
        v = d.get(a)
        if v is None:
            v = d.get(b)
        try:
            return float(v or 0.0)
        except Exception:
            return 0.0

    true_relevance: List[float] = []
    for r in results_json:
        lp = r.get("label_probabilities") or {}
        p_good = _get_prob(lp, "good_fit", "Good Fit")
        p_potential = _get_prob(lp, "potential_fit", "Potential Fit")
        if p_good > 0.0 or p_potential > 0.0:
            true_relevance.append(3.0 * p_good + 2.0 * p_potential)
            continue

        label = (r.get("label") or "").strip()
        if label == "Good Fit":
            true_relevance.append(3.0)
        elif label == "Potential Fit":
            true_relevance.append(2.0)
        else:
            true_relevance.append(0.0)

    if not true_relevance or max(true_relevance) <= 0.0:
        return 0.0

    return float(ndcg_score([true_relevance], [predicted_scores], k=k))


MATRIX_JOB_CATEGORIES = [
    "ACCOUNTANT",
    "FINANCE",
    "HUMAN-RESOURCES",
    "SALES",
    "BUSINESS-DEVELOPMENT",
    "INFORMATION-TECHNOLOGY",
    "ENGINEERING",
    "HEALTHCARE",
    "TEACHER",
    "DESIGNER",
]


MATRIX_DISTRACTORS = {
    "ACCOUNTANT": {"near": "FINANCE", "far": "INFORMATION-TECHNOLOGY"},
    "FINANCE": {"near": "BANKING", "far": "INFORMATION-TECHNOLOGY"},
    "HUMAN-RESOURCES": {"near": "CONSULTANT", "far": "CONSTRUCTION"},
    "SALES": {"near": "BUSINESS-DEVELOPMENT", "far": "ACCOUNTANT"},
    "BUSINESS-DEVELOPMENT": {"near": "SALES", "far": "HEALTHCARE"},
    "INFORMATION-TECHNOLOGY": {"near": "ENGINEERING", "far": "ACCOUNTANT"},
    "ENGINEERING": {"near": "INFORMATION-TECHNOLOGY", "far": "HUMAN-RESOURCES"},
    "HEALTHCARE": {"near": "FITNESS", "far": "INFORMATION-TECHNOLOGY"},
    "TEACHER": {"near": "HUMAN-RESOURCES", "far": "INFORMATION-TECHNOLOGY"},
    "DESIGNER": {"near": "DIGITAL-MEDIA", "far": "ACCOUNTANT"},
}


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _read_jd_from_dir(jd_dir: str, job_category: str) -> Optional[str]:
    if not jd_dir:
        return None
    jd_path = Path(jd_dir) / f"{validate_category(job_category)}.txt"
    if not jd_path.exists() or not jd_path.is_file():
        return None
    return _read_text_file(str(jd_path)).strip()


def _list_pdf_paths(category_pdf_dir: str, category: str) -> List[str]:
    root = Path(category_pdf_dir)
    cat_dir = root / category
    if not cat_dir.exists() or not cat_dir.is_dir():
        target = validate_category(category)
        for child in root.iterdir():
            if child.is_dir() and validate_category(child.name) == target:
                cat_dir = child
                break
    if not cat_dir.exists() or not cat_dir.is_dir():
        return []

    return sorted(
        str(p) for p in cat_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"
    )


def _sample(paths: List[str], n: int, rng: random.Random) -> List[str]:
    if n <= 0:
        return []
    if n >= len(paths):
        out = list(paths)
        rng.shuffle(out)
        return out

    return rng.sample(paths, n)


def _post_json(url: str, payload: Dict) -> Dict:
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


def _build_candidates(
    extractor: PDFTextExtractor,
    pdf_paths: List[str],
    true_category: str,
    min_chars: int,
    target_n: Optional[int] = None,
) -> List[Dict]:
    candidates = []
    for pdf_path in pdf_paths:
        if target_n is not None and len(candidates) >= target_n:
            break
        raw_text = extractor.extract(pdf_path)
        cleaned = clean_resume_text(raw_text)
        if not cleaned or len(cleaned) < min_chars:
            continue

        file_stem = Path(pdf_path).stem
        candidates.append(
            {
                "id": f"{validate_category(true_category)}::{file_stem}",
                "text": cleaned,
                "_meta": {
                    "true_category": validate_category(true_category),
                    "pdf_path": str(pdf_path),
                },
            }
        )

    return candidates


def _results_to_jsonable(results) -> List[Dict]:
    out = []
    for r in results:
        out.append(
            {
                "candidate_id": r.candidate_id,
                "final_score": r.final_score,
                "raw_score": r.raw_score,
                "label": r.label,
                "label_probabilities": r.label_probabilities,
                "category": {
                    "match": r.category_match,
                    "predicted": r.category_predicted,
                    "confidence": r.category_confidence,
                },
                "skill_overlap": r.skill_overlap,
                "keyword_overlap": r.keyword_overlap,
                "processing_time": r.processing_time,
                "metadata": r.metadata or {},
            }
        )
    return out


def _summarize(results_json: List[Dict], id_to_true: Dict[str, str]) -> Dict:
    if not results_json:
        return {
            "topn": 0,
            "true_category_counts": {},
            "predicted_category_counts": {},
            "category_match_rate": 0.0,
        }

    true_counts: Dict[str, int] = {}
    pred_counts: Dict[str, int] = {}
    match_count = 0

    for item in results_json:
        cid = item.get("candidate_id", "")
        true_cat = id_to_true.get(cid, "UNKNOWN")
        pred_cat = (item.get("category") or {}).get("predicted", "UNKNOWN")

        true_counts[true_cat] = true_counts.get(true_cat, 0) + 1
        pred_counts[pred_cat] = pred_counts.get(pred_cat, 0) + 1

        if (item.get("category") or {}).get("match"):
            match_count += 1

    return {
        "topn": len(results_json),
        "true_category_counts": dict(sorted(true_counts.items(), key=lambda x: (-x[1], x[0]))),
        "predicted_category_counts": dict(sorted(pred_counts.items(), key=lambda x: (-x[1], x[0]))),
        "category_match_rate": float(match_count) / float(len(results_json)),
    }


def _score_breakdown(results_json: List[Dict], id_to_true: Dict[str, str], job_category: str, distractor_category: str) -> Dict:
    job_scores = []
    dist_scores = []
    n_missing_score = 0
    for item in results_json:
        cid = item.get("candidate_id", "")
        true_cat = id_to_true.get(cid, "UNKNOWN")

        score_raw = item.get("final_score", None)
        if score_raw is None:
            n_missing_score += 1
            continue
        try:
            score = float(score_raw)
        except Exception:
            n_missing_score += 1
            continue

        if true_cat == job_category:
            job_scores.append(score)
        elif true_cat == distractor_category:
            dist_scores.append(score)

    def _mean(xs: List[float]) -> Optional[float]:
        return float(sum(xs)) / float(len(xs)) if xs else None

    mean_job = _mean(job_scores)
    mean_dist = _mean(dist_scores)

    mean_gap = None
    if mean_job is not None and mean_dist is not None:
        mean_gap = mean_job - mean_dist

    return {
        "mean_final_score_job": round(mean_job, 6) if mean_job is not None else None,
        "mean_final_score_distractor": round(mean_dist, 6) if mean_dist is not None else None,
        "mean_final_score_gap": round(mean_gap, 6) if mean_gap is not None else None,
        "n_job_in_topn": len(job_scores),
        "n_distractor_in_topn": len(dist_scores),
        "n_missing_score": n_missing_score,
    }


def _topn_breakdown(
    results_json: List[Dict],
    id_to_true: Dict[str, str],
    job_category: str,
    distractor_category: str,
    n: int = 10,
) -> Dict:
    top = results_json[: max(0, int(n))]
    job_scores = []
    dist_scores = []
    n_missing_score = 0
    for item in top:
        cid = item.get("candidate_id", "")
        true_cat = id_to_true.get(cid, "UNKNOWN")

        score_raw = item.get("final_score", None)
        if score_raw is None:
            n_missing_score += 1
            continue
        try:
            score = float(score_raw)
        except Exception:
            n_missing_score += 1
            continue

        if true_cat == job_category:
            job_scores.append(score)
        elif true_cat == distractor_category:
            dist_scores.append(score)

    def _mean(xs: List[float]) -> Optional[float]:
        return float(sum(xs)) / float(len(xs)) if xs else None

    mean_job = _mean(job_scores)
    mean_dist = _mean(dist_scores)
    mean_gap = None
    if mean_job is not None and mean_dist is not None:
        mean_gap = mean_job - mean_dist
    denom = float(len(top)) if top else 1.0
    return {
        "topn": len(top),
        "topn_job_precision": round(float(len(job_scores)) / denom, 6),
        "topn_n_job": len(job_scores),
        "topn_n_distractor": len(dist_scores),
        "topn_mean_score_job": round(mean_job, 6) if mean_job is not None else None,
        "topn_mean_score_distractor": round(mean_dist, 6) if mean_dist is not None else None,
        "topn_mean_score_gap": round(mean_gap, 6) if mean_gap is not None else None,
        "topn_n_missing_score": n_missing_score,
    }


def _pairwise_ranking_accuracy(
    results_json: List[Dict],
    id_to_true: Dict[str, str],
    job_category: str,
    distractor_category: str,
) -> Dict:
    job_scores = []
    dist_scores = []
    for item in results_json:
        cid = item.get("candidate_id", "")
        true_cat = id_to_true.get(cid, "UNKNOWN")
        score = float(item.get("final_score", 0.0) or 0.0)
        if true_cat == job_category:
            job_scores.append(score)
        elif true_cat == distractor_category:
            dist_scores.append(score)

    total_pairs = len(job_scores) * len(dist_scores)
    if total_pairs == 0:
        return {"pairwise_accuracy": 0.0, "pairwise_pairs": 0}

    wins = 0.0
    for js in job_scores:
        for ds in dist_scores:
            if js > ds:
                wins += 1.0
            elif js == ds:
                wins += 0.5

    return {
        "pairwise_accuracy": round(wins / float(total_pairs), 6),
        "pairwise_pairs": int(total_pairs),
    }


def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def parse_args() -> argparse.Namespace:
    data_config = config.get("data", {}).get("raw", {})

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf_dir",
        type=str,
        default=data_config.get("category_pdf_dir", "datasets/category resumes/data/data"),
    )
    parser.add_argument("--job_category", type=str, default="ACCOUNTANT")
    parser.add_argument("--distractor_category", type=str, default="INFORMATION-TECHNOLOGY")
    parser.add_argument("--n_job", type=int, default=50)
    parser.add_argument("--n_distractor", type=int, default=50)
    parser.add_argument("--min_chars", type=int, default=400)

    parser.add_argument("--device", type=str, default=config.get("inference", {}).get("device", "auto"))
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--pipeline_top_k", type=int, default=100)

    parser.add_argument(
        "--use_api",
        action="store_true",
        help="If set, do not load models locally. Instead call the running API /rank endpoint.",
    )
    parser.add_argument(
        "--api_url",
        type=str,
        default="http://127.0.0.1:8000/rank",
        help="API /rank URL to call when --use_api is set.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jd_file", type=str, default="")
    parser.add_argument("--jd", type=str, default="")
    parser.add_argument(
        "--jd_dir",
        type=str,
        default="",
        help="Optional directory containing per-category JD text files (e.g., ACCOUNTANT.txt).",
    )

    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Run a cross-domain matrix over 10 categories with predefined near/far distractors.",
    )

    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    pdf_cache_file = config.get("data", {}).get("processed", {}).get(
        "pdf_text_cache", "data/processed/pdf_text_cache.json"
    )
    pdf_cfg = PDFProcessingConfig(cache_enabled=True, cache_file=pdf_cache_file)
    pdf_extractor = PDFTextExtractor(pdf_cfg)
    pdf_dir = args.pdf_dir

    pipeline = None
    if not args.use_api:
        from pipeline.inference import TriadRankPipeline

        model1_path = config.get("models", {}).get("cross_encoder", {}).get("save_path")
        model2_path = config.get("models", {}).get("category_encoder", {}).get("save_path")
        model3_spacy_model = config.get("models", {}).get("extractor", {}).get(
            "spacy_model", "en_core_web_sm"
        )
        pipeline = TriadRankPipeline(
            model1_path=model1_path,
            model2_path=model2_path,
            model3_spacy_model=model3_spacy_model,
            device=args.device,
            top_k=max(1, int(args.pipeline_top_k)),
            category_penalty=config.get("penalties", {}).get("category_mismatch", 0.5),
        )

    def _run_once(job_category_in: str, distractor_category_in: str, seed: int, tag: str) -> Dict:
        job_category = validate_category(job_category_in)
        distractor_category = validate_category(distractor_category_in)

        jd_from_dir = _read_jd_from_dir(args.jd_dir, job_category)
        if args.jd_file:
            job_description = _read_text_file(args.jd_file).strip()
        elif args.jd:
            job_description = args.jd.strip()
        elif jd_from_dir:
            job_description = jd_from_dir
        else:
            job_description = _default_jd_for_category(job_category)

        rng = random.Random(seed)

        job_paths = _list_pdf_paths(pdf_dir, job_category)
        dist_paths = _list_pdf_paths(pdf_dir, distractor_category)
        if not job_paths:
            raise RuntimeError(f"No PDF resumes found for job_category={job_category} under {pdf_dir}")
        if not dist_paths:
            raise RuntimeError(f"No PDF resumes found for distractor_category={distractor_category} under {pdf_dir}")

        rng.shuffle(job_paths)
        rng.shuffle(dist_paths)

        t0 = time.time()

        job_candidates = _build_candidates(
            pdf_extractor, job_paths, job_category, args.min_chars, target_n=args.n_job
        )
        dist_candidates = _build_candidates(
            pdf_extractor,
            dist_paths,
            distractor_category,
            args.min_chars,
            target_n=args.n_distractor,
        )
        candidates = job_candidates + dist_candidates
        if not candidates:
            raise RuntimeError(
                "No candidates were produced after PDF extraction/cleaning. "
                "Try lowering --min_chars or confirm PDFs contain extractable text."
            )

        rng.shuffle(candidates)

        id_to_true = {}
        for c in candidates:
            meta = c.get("_meta") or {}
            id_to_true[c.get("id")] = meta.get("true_category", "UNKNOWN")

        if args.use_api:
            api_candidates = [{"id": c["id"], "text": c["text"]} for c in candidates]
            request_top_k = max(int(args.top_k), int(args.pipeline_top_k))
            api_payload = {
                "job_description": job_description,
                "job_category": job_category,
                "candidates": api_candidates,
                "top_k": request_top_k,
            }
            api_response = _post_json(args.api_url, api_payload)
            api_results = (api_response.get("results") or [])[: int(args.top_k)]

            results_json = []
            for item in api_results:
                cid = item.get("candidate_id")
                results_json.append({**item, "true_category": id_to_true.get(cid, "UNKNOWN")})
        else:
            if pipeline is None:
                raise RuntimeError("Local mode requires pipeline initialization")
            results = pipeline.rank_candidates(job_description, job_category, candidates)
            results_json = _results_to_jsonable(results[: args.top_k])
            for item in results_json:
                cid = item.get("candidate_id")
                item["true_category"] = id_to_true.get(cid, "UNKNOWN")

        summary = _summarize(results_json, id_to_true)
        score_summary = _score_breakdown(results_json, id_to_true, job_category, distractor_category)
        top10_summary = _topn_breakdown(results_json, id_to_true, job_category, distractor_category, n=10)
        pairwise = _pairwise_ranking_accuracy(results_json, id_to_true, job_category, distractor_category)
        ndcg_all = _calculate_ndcg(results_json)
        ndcg_10 = _calculate_ndcg(results_json, k=10)
        runtime_s = time.time() - t0

        return {
            "job_category": job_category,
            "distractor_category": distractor_category,
            "tag": tag,
            "job_description": job_description,
            "input": {
                "pdf_dir": pdf_dir,
                "n_job_requested": args.n_job,
                "n_distractor_requested": args.n_distractor,
                "n_job_used": len(job_candidates),
                "n_distractor_used": len(dist_candidates),
                "n_job_pdfs_available": len(job_paths),
                "n_distractor_pdfs_available": len(dist_paths),
                "min_chars": args.min_chars,
                "pipeline_top_k": int(args.pipeline_top_k),
                "top_k_returned": min(args.top_k, len(results_json)),
                "seed": seed,
                "use_api": bool(args.use_api),
                "api_url": args.api_url if args.use_api else "",
            },
            "candidates_total": len(candidates),
            "runtime_seconds": round(runtime_s, 4),
            "summary": summary,
            "score_summary": score_summary,
            "top10_summary": top10_summary,
            "pairwise": pairwise,
            "ndcg": {
                "all": round(float(ndcg_all), 6),
                "at_10": round(float(ndcg_10), 6),
            },
            "results": results_json,
        }

    if args.matrix:
        tests = []
        for c in MATRIX_JOB_CATEGORIES:
            jc = validate_category(c)
            d = MATRIX_DISTRACTORS.get(jc) or MATRIX_DISTRACTORS.get(c)
            if not d:
                continue
            tests.append((jc, d.get("near", ""), "near"))
            tests.append((jc, d.get("far", ""), "far"))

        reports = []
        rows = []
        for idx, (jc, dc, tag) in enumerate(tests):
            if not dc:
                continue
            r = _run_once(jc, dc, args.seed + idx, tag)
            reports.append(r)

            s = r.get("summary") or {}
            ss = r.get("score_summary") or {}
            t10 = r.get("top10_summary") or {}
            pw = r.get("pairwise") or {}
            nd = r.get("ndcg") or {}

            mean_final_score_job = ss.get("mean_final_score_job")
            mean_final_score_distractor = ss.get("mean_final_score_distractor")
            mean_final_score_gap = ss.get("mean_final_score_gap")
            top10_mean_score_job = t10.get("topn_mean_score_job")
            top10_mean_score_distractor = t10.get("topn_mean_score_distractor")
            top10_mean_score_gap = t10.get("topn_mean_score_gap")
            rows.append(
                {
                    "job_category": r.get("job_category"),
                    "distractor_category": r.get("distractor_category"),
                    "tag": r.get("tag"),
                    "seed": (r.get("input") or {}).get("seed"),
                    "candidates_total": r.get("candidates_total"),
                    "n_job_used": (r.get("input") or {}).get("n_job_used"),
                    "n_distractor_used": (r.get("input") or {}).get("n_distractor_used"),
                    "topn": s.get("topn"),
                    "match_rate": round(float(s.get("category_match_rate") or 0.0), 6),
                    "n_job_in_topn": ss.get("n_job_in_topn"),
                    "n_distractor_in_topn": ss.get("n_distractor_in_topn"),
                    "mean_final_score_job": mean_final_score_job if mean_final_score_job is not None else "",
                    "mean_final_score_distractor": mean_final_score_distractor if mean_final_score_distractor is not None else "",
                    "mean_final_score_gap": mean_final_score_gap if mean_final_score_gap is not None else "",
                    "top10_job_precision": t10.get("topn_job_precision"),
                    "top10_n_job": t10.get("topn_n_job"),
                    "top10_n_distractor": t10.get("topn_n_distractor"),
                    "top10_mean_score_job": top10_mean_score_job if top10_mean_score_job is not None else "",
                    "top10_mean_score_distractor": top10_mean_score_distractor if top10_mean_score_distractor is not None else "",
                    "top10_mean_score_gap": top10_mean_score_gap if top10_mean_score_gap is not None else "",
                    "pairwise_accuracy": pw.get("pairwise_accuracy"),
                    "pairwise_pairs": pw.get("pairwise_pairs"),
                    "ndcg": nd.get("all"),
                    "ndcg_at_10": nd.get("at_10"),
                    "runtime_seconds": r.get("runtime_seconds"),
                }
            )

        avg_match = 0.0
        if rows:
            avg_match = float(sum(float(r.get("match_rate") or 0.0) for r in rows)) / float(len(rows))

        matrix_report = {
            "matrix_job_categories": MATRIX_JOB_CATEGORIES,
            "tests_ran": len(reports),
            "avg_match_rate": round(avg_match, 6),
            "rows": rows,
            "reports": reports,
        }

        if args.output:
            out_path = Path(args.output)
        else:
            out_dir = Path("reports")
            out_path = out_dir / f"rank_eval_matrix_pdfs_{int(time.time())}.json"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(matrix_report, f, indent=2, ensure_ascii=False)

        csv_path = out_path.with_suffix(".csv")
        _write_csv(csv_path, rows)

        print(f"Wrote matrix report: {out_path}")
        print(f"Wrote matrix CSV: {csv_path}")
        print(f"Tests ran: {len(reports)}")
        print(f"Average match-rate: {avg_match:.3f}")
        return 0

    job_category = validate_category(args.job_category)
    distractor_category = validate_category(args.distractor_category)
    report = _run_once(job_category, distractor_category, args.seed, "single")

    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = Path("reports")
        out_path = out_dir / f"rank_eval_pdfs_{job_category}_vs_{distractor_category}_{int(time.time())}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    summary = report.get("summary") or {}
    print(f"Wrote report: {out_path}")
    print(f"Candidates used: {report.get('candidates_total')}")
    print(f"Top-{summary.get('topn')} match-rate: {summary.get('category_match_rate'):.3f}")
    print("Top true-category counts:")
    for k, v in (summary.get("true_category_counts") or {}).items():
        print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
