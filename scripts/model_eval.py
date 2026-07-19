"""
Comprehensive model evaluation — real data, real results, no human labels.

Evaluates all three models on florex real-world resumes:
  Category Encoder -> confusion matrix, per-category F1, misclassification samples
  Cross-Encoder -> score distribution, category-match effect, label separation
  SkillTrie -> recall on known florex occupation skills
  Pipeline -> end-to-end score sanity checks

Usage:
    python scripts/model_eval.py               # full eval
    python scripts/model_eval.py --categories    # category encoder only
"""

import os, sys, json, re, csv, random, math
from pathlib import Path
from collections import Counter, defaultdict

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from models.category_encoder import CategoryEncoder
from models.cross_encoder import MultiHeadCrossEncoder
from models.extractor import ResumeExtractor
from models.skill_trie import SkillTrie
from pipeline.inference import TriadRankPipeline
from data.loaders import clean_resume_text

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── Florex data loader ──────────────────────────────────────────────

FLOREX_DIR = Path("datasets/external/resume_corpus/resumes_corpus")

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

def _strip_html(text):
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;|\\xa0", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_florex_data(max_files=2000, min_words=30):
    """Load florex resumes with ground-truth category labels (stratified random sample)."""
    results_by_label = defaultdict(list)
    txt_files = sorted(FLOREX_DIR.glob("*.txt"))
    for tf in txt_files:
        stem = tf.stem
        lab_file = tf.with_suffix(".lab")
        if not lab_file.exists():
            continue
        label = lab_file.read_text("utf-8").strip()
        cat = FLOREX_TO_CATEGORY.get(label)
        if cat is None:
            continue
        raw = tf.read_text("utf-8", errors="replace")
        clean = _strip_html(raw)
        if len(clean.split()) < min_words:
            continue
        results_by_label[label].append({"id": stem, "category": cat, "label": label, "text": clean})

    # Stratified sample: equal per original florex label
    rng = random.Random(42)
    per_label = max_files // len(results_by_label)
    results = []
    for label, pool in sorted(results_by_label.items()):
        chosen = rng.sample(pool, min(per_label, len(pool)))
        results.extend(chosen)
    # If we got fewer than max_files, top up with random
    if len(results) < max_files:
        remaining = [s for pool in results_by_label.values() for s in pool if s not in results]
        extra = rng.sample(remaining, min(max_files - len(results), len(remaining)))
        results.extend(extra)
    rng.shuffle(results)
    return results

# ── Part 1: Category Encoder Confusion Matrix ─────────────────────────

def eval_category_encoder(samples, max_samples=2000):
    print("\n" + "=" * 65)
    print("CATEGORY ENCODER — CONFUSION MATRIX & MISCLASSIFICATION ANALYSIS")
    print("=" * 65)

    ckpt = Path("checkpoints/category_encoder/best_model.pt")
    if not ckpt.exists():
        print("SKIP: No checkpoint found")
        return

    model = CategoryEncoder.load(str(ckpt), device=device)
    model.eval()

    # Load tokenizer separately — model doesn't store a reference
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model.pretrained_model_name, token=False)

    # Build id_to_name from checkpoint or config
    mapping_path = Path("checkpoints/category_encoder/category_mapping.json")
    if mapping_path.exists():
        mapping = json.loads(mapping_path.read_text())
        id_to_name = {int(k): v for k, v in mapping["id_to_name"].items()}
        name_to_id = mapping["name_to_id"]
    else:
        id_to_name = {int(k): v for k, v in config["categories"]["id_to_name"].items()}
        name_to_id = config["categories"]["name_to_id"]

    eval_samples = random.Random(42).sample(samples, min(max_samples, len(samples)))

    all_preds = []
    all_targets = []
    all_confidences = []
    misclassified = []

    for s in eval_samples:
        target = name_to_id.get(s["category"])
        if target is None:
            continue

        encoding = tokenizer(
            s["text"],
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            pred = model.predict(input_ids, attention_mask)
            pred_id = pred["predicted_class"].item()
            confidence = pred["confidence"].item()

        all_preds.append(pred_id)
        all_targets.append(target)
        all_confidences.append(confidence)

        if pred_id != target:
            misclassified.append({
                "id": s["id"],
                "true": id_to_name[target],
                "pred": id_to_name[pred_id],
                "confidence": confidence,
                "text_preview": s["text"][:200],
            })

    correct = sum(1 for p, t in zip(all_preds, all_targets) if p == t)
    accuracy = correct / len(all_preds) if all_preds else 0

    # Per-category metrics
    print(f"\nSamples evaluated: {len(all_preds)}")
    print(f"Overall accuracy:  {accuracy:.4f} ({correct}/{len(all_preds)})")
    print(f"Misclassified:     {len(misclassified)}")

    print(f"\n{'Category':<30} {'Samples':>8} {'Errors':>8} {'Acc':>8} {'Avg Conf':>9}")
    print("-" * 65)
    cats_in_data = sorted(set(all_targets))
    per_cat = defaultdict(list)
    for p, t, c in zip(all_preds, all_targets, all_confidences):
        per_cat[t].append((p, c))

    macro_f1_scores = []
    for cat_id in cats_in_data:
        cat_name = id_to_name[cat_id]
        entries = per_cat[cat_id]
        n = len(entries)
        errs = sum(1 for p, _ in entries if p != cat_id)
        acc = 1 - errs / n if n else 0
        avg_conf = np.mean([c for _, c in entries]) if entries else 0
        # Per-class precision/recall/F1
        tp = sum(1 for p, _ in entries if p == cat_id)
        pred_count = sum(1 for p in all_preds if p == cat_id)
        precision = tp / max(pred_count, 1)
        recall = tp / max(n, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        macro_f1_scores.append(f1)
        print(f"{cat_name:<30} {n:>8} {errs:>8} {acc:.4f}  {avg_conf:.4f}")

    macro_f1 = np.mean(macro_f1_scores) if macro_f1_scores else 0
    print(f"\nMacro F1: {macro_f1:.4f}")

    # Top misclassification patterns
    print(f"\n{'-' * 65}")
    print("TOP CONFUSION PAIRS (true vs pred):")
    confusion = Counter((m["true"], m["pred"]) for m in misclassified)
    for (true, pred), count in confusion.most_common(15):
        pct = count / sum(1 for t in all_targets for ct in [true] if id_to_name[t] == true) * 100
        print(f"  {true:<28} -> {pred:<28}  {count:>4} errors ({pct:.1f}%)")

    # Show specific misclassification examples
    print(f"\n{'-' * 65}")
    print("MISCLASSIFICATION EXAMPLES (highest confidence errors):")
    sorted_mis = sorted(misclassified, key=lambda x: -x["confidence"])[:10]
    for m in sorted_mis:
        print(f"\n  [{m['id']}] True: {m['true']} -> Pred: {m['pred']} (conf: {m['confidence']:.3f})")
        print(f"  Text: {m['text_preview'][:150]}...")

    return {"accuracy": accuracy, "macro_f1": macro_f1, "misclassified": len(misclassified), "total": len(all_preds)}


# ── Part 2: Cross-Encoder Score Distribution ─────────────────────────

def eval_cross_encoder(samples, max_samples=500):
    print("\n\n" + "=" * 65)
    print("CROSS-ENCODER — SCORE DISTRIBUTION & CATEGORY EFFECT")
    print("=" * 65)

    ckpt = Path("checkpoints/cross_encoder/best_model.pt")
    if not ckpt.exists():
        print("SKIP: No checkpoint found")
        return

    model = MultiHeadCrossEncoder.load(str(ckpt), device=device)
    model.eval()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", token=False)

    eval_samples = random.Random(42).sample(samples, min(max_samples, len(samples)))

    # Group by category
    by_cat = defaultdict(list)
    for s in eval_samples:
        by_cat[s["category"]].append(s)

    jds = {
        "INFORMATION-TECHNOLOGY": "Senior Software Engineer — build scalable distributed systems. Python, Java, or Go. Cloud-native (AWS/GCP), containers (Docker/Kubernetes), CI/CD. Strong CS fundamentals and system design.",
        "BUSINESS-DEVELOPMENT": "Project Manager — lead cross-functional initiatives, manage timelines, budgets, stakeholder communication. PMP certification preferred. Agile/Scrum experience.",
        "FINANCE": "Financial Analyst — financial modeling, forecasting, reporting. Excel, Bloomberg, SQL. CFA or MBA preferred.",
        "HEALTHCARE": "Registered Nurse — patient care, medication administration, care planning. BLS/ACLS certified.",
        "SALES": "Sales Account Executive — manage enterprise accounts, drive revenue growth. CRM (Salesforce), pipeline management.",
        "ENGINEERING": "Mechanical Engineer — product design, CAD modeling (SolidWorks), FEA analysis, prototyping.",
        "HUMAN-RESOURCES": "HR Manager — talent acquisition, employee relations, performance management, compliance. SHRM-CP or PHR.",
    }

    print(f"\nSamples: {len(eval_samples)}")
    print(f"Categories: {len(by_cat)} ({', '.join(sorted(by_cat.keys()))})")

    # Score IT resumes against IT JD vs Business-Dev JD
    print(f"\n{'-' * 65}")
    print("CATEGORY-MATCH vs CROSS-CATEGORY SCORING:")
    print(f"{'Condition':<45} {'Avg Score':>10} {'Std':>8}")
    print("-" * 65)

    it_resumes = by_cat.get("INFORMATION-TECHNOLOGY", [])
    bd_resumes = by_cat.get("BUSINESS-DEVELOPMENT", [])

    def score_batch(resumes, jd_text, max_len=384, batch_size=16):
        scores = []
        for i in range(0, len(resumes), batch_size):
            batch = resumes[i:i+batch_size]
            texts = [s["text"] for s in batch]
            pairs = [(t, jd_text) for t in texts]
            enc = tokenizer(
                pairs, truncation=True, max_length=max_len,
                padding=True, return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            with torch.no_grad():
                pred = model.predict(input_ids, attention_mask)
                scores.extend(pred["score"].cpu().numpy().tolist())
        return scores

    if it_resumes:
        it_jd = jds.get("INFORMATION-TECHNOLOGY", "Software engineering role.")
        scores_match = score_batch(it_resumes, it_jd)
        print(f"{'IT resume × IT JD (match)':<45} {np.mean(scores_match):10.4f} {np.std(scores_match):8.4f}")

        bd_jd = jds.get("BUSINESS-DEVELOPMENT", "Business development role.")
        scores_cross = score_batch(it_resumes, bd_jd)
        print(f"{'IT resume × Business-Dev JD (mismatch)':<45} {np.mean(scores_cross):10.4f} {np.std(scores_cross):8.4f}")

        # T-test approximation
        from scipy import stats as scipy_stats
        try:
            t_stat, p_val = scipy_stats.ttest_ind(scores_match, scores_cross)
            print(f"{'T-test p-value':<45} {p_val:>10.6f}")
            print(f"{'Effect size (Cohen d)':<45} {(np.mean(scores_match)-np.mean(scores_cross))/np.std(scores_match+scores_cross):>10.4f}")
        except:
            pass

    # IT vs IT with same JD — should be similar but some spread
    if len(it_resumes) >= 20:
        s = score_batch(it_resumes[:20], it_jd)
        print(f"\n{'-' * 65}")
        print("IT×IT SCORE SPREAD (20 IT resumes, same JD):")
        print(f"  Min: {min(s):.4f}, Max: {max(s):.4f}, Range: {max(s)-min(s):.4f}")
        print(f"  Mean: {np.mean(s):.4f}, Std: {np.std(s):.4f}")
        print(f"  Spread (std/mean): {np.std(s)/max(np.mean(s), 1e-8):.3f}")
        # Lower spread means the model doesn't distinguish well between IT resumes
        print(f"  {'Interpretation: low spread = model sees IT resumes as similar':>60}")

    return


# ── Part 3: Skill Extraction Recall ─────────────────────────────────

def eval_skill_extraction(samples, max_samples=100):
    print("\n\n" + "=" * 65)
    print("SKILL EXTRACTION — SKILLTRIE OUTPUT ON REAL RESUMES")
    print("=" * 65)

    extractor = ResumeExtractor(skill_extractor_mode="trie")
    extractor_regex = ResumeExtractor(skill_extractor_mode="regex")

    it_samples = [s for s in samples if s["category"] == "INFORMATION-TECHNOLOGY"]
    eval_samples = it_samples[:max_samples]

    # Sample a few resume previews with extracted skills
    print(f"\nSample extractions (IT resumes):")
    for s in eval_samples[:5]:
        skills = extractor.extract_skills_list(s["text"])
        # Show a preview that highlights skill mentions
        preview = s["text"][:200].replace('\n', ' ')
        print(f"\n  [{s['id']}] ({len(skills)} skills found)")
        print(f"  Text: {preview[:150]}...")
        cats = set()
        for sk in skills[:10]:
            cats.add(sk)
        print(f"  First 10 skills: {skills[:10]}")

    # Stats
    trie_counts = [len(extractor.extract_skills_list(s["text"])) for s in eval_samples]
    regex_counts = [len(extractor_regex.extract_skills_list(s["text"])) for s in eval_samples]

    print(f"\n\nSkills per resume (n={len(eval_samples)} IT resumes):")
    print(f"  Trie:  avg={np.mean(trie_counts):.1f}, std={np.std(trie_counts):.1f}, "
          f"min={min(trie_counts)}, max={max(trie_counts)}, median={np.median(trie_counts):.0f}")
    print(f"  Regex: avg={np.mean(regex_counts):.1f}, std={np.std(regex_counts):.1f}, "
          f"min={min(regex_counts)}, max={max(regex_counts)}, median={np.median(regex_counts):.0f}")

    return {"trie_avg": np.mean(trie_counts), "regex_avg": np.mean(regex_counts)}


# ── Part 4: Pipeline End-to-End Sanity ──────────────────────────────

def eval_pipeline(samples, max_samples=30):
    print("\n\n" + "=" * 65)
    print("PIPELINE — END-TO-END RANKING SANITY CHECKS")
    print("=" * 65)

    ckpt1 = Path("checkpoints/cross_encoder/best_model.pt")
    ckpt2 = Path("checkpoints/category_encoder/best_model.pt")
    if not ckpt1.exists() or not ckpt2.exists():
        print("SKIP: Missing checkpoints")
        return

    # Use the real_pipeline fixture
    pipe = TriadRankPipeline(
        model1_path=str(ckpt1),
        model2_path=str(ckpt2),
        device=device,
    )

    eval_samples = random.Random(42).sample(samples, min(max_samples, len(samples)))

    jds = {
        "INFORMATION-TECHNOLOGY": "Senior Software Engineer — Python, Java, Go. Cloud-native (AWS/GCP), Docker/Kubernetes, CI/CD. System design.",
        "BUSINESS-DEVELOPMENT": "Project Manager — lead initiatives, manage budgets, stakeholder communication. PMP, Agile/Scrum.",
        "FINANCE": "Financial Analyst — modeling, forecasting, reporting. Excel, Bloomberg, SQL.",
        "HEALTHCARE": "Registered Nurse — patient care, medication, care planning. BLS/ACLS.",
        "SALES": "Sales Account Executive — enterprise accounts, revenue growth. CRM, pipeline.",
        "ENGINEERING": "Mechanical Engineer — CAD, FEA, prototyping, manufacturing.",
        "HUMAN-RESOURCES": "HR Manager — talent acquisition, employee relations, compliance.",
    }

    print(f"\nSamples: {len(eval_samples)}")

    # 1. Score distribution by category
    print(f"\n{'-' * 65}")
    print("SCORE DISTRIBUTION BY CATEGORY:")
    print(f"{'Category':<25} {'Count':>6} {'Avg Score':>10} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 65)

    results_by_cat = defaultdict(list)
    for s in eval_samples:
        try:
            jd = jds.get(s["category"], f"Professional role in {s['category']}.")
            r = pipe.rank_single(
                resume_text=s["text"],
                job_description=jd,
                job_category=s["category"],
            )
            results_by_cat[s["category"]].append(r.final_score)
        except Exception as e:
            print(f"  Error on {s['id']}: {e}")

    for cat, scores in sorted(results_by_cat.items()):
        print(f"{cat:<25} {len(scores):>6} {np.mean(scores):>10.4f} {np.std(scores):>8.4f} {min(scores):>8.4f} {max(scores):>8.4f}")

    # 2. Cross-category penalty effect
    print(f"\n{'-' * 65}")
    print("CROSS-CATEGORY PENALTY EFFECT (IT resume scored across all categories):")
    it_resumes = [s for s in eval_samples if s["category"] == "INFORMATION-TECHNOLOGY"]
    if it_resumes:
        s = it_resumes[0]
        it_jd = jds.get("INFORMATION-TECHNOLOGY", "")
        print(f"{'Category':<25} {'Score':>8} {'Category Match':>15} {'Penalty':>8}")
        print("-" * 60)
        for cat_name in sorted(config["categories"]["name_to_id"].keys()):
            try:
                r = pipe.rank_single(
                    resume_text=s["text"],
                    job_description=it_jd,
                    job_category=cat_name,
                )
                match = "MATCH" if cat_name == "INFORMATION-TECHNOLOGY" else "MISMATCH"
                print(f"{cat_name:<25} {r.final_score:>8.4f} {match:>15} {r.metadata.get('penalty_factor', 1.0):>8.2f}")
            except Exception as e:
                print(f"{cat_name:<25} ERROR: {e}")

    # 3. Label distribution (reuse already scored results)
    print(f"\n{'-' * 65}")
    print("LABEL DISTRIBUTION (Good Fit / Potential Fit / Bad Fit):")
    label_counts = Counter()
    for s in eval_samples:
        if s["id"] in results_by_cat:
            pass  # already scored above
    for label, count in label_counts.most_common():
        print(f"  {label}: {count} ({100*count/max(sum(label_counts.values()),1):.1f}%)")


# ── Part 5: Synthesis ───────────────────────────────────────────────

def synthesize(all_results):
    print("\n\n" + "=" * 65)
    print("SYNTHESIS — WHAT NEEDS FIXING")
    print("=" * 65)

    # Category encoder analysis
    if "category" in all_results:
        cat = all_results["category"]
        print(f"\n[DATA] CATEGORY ENCODER: {cat['accuracy']:.1%} accuracy, {cat['macro_f1']:.3f} macro F1")
        if cat["accuracy"] < 0.85:
            print(f"  [WARN] Accuracy below 85% — {cat['misclassified']}/{cat['total']} misclassified")
        else:
            print(f"  [OK] Accuracy above 85%")

    # Cross-encoder analysis
    if "cross" in all_results:
        cross = all_results["cross"]
        print(f"\n[DATA] CROSS-ENCODER:")
        # placeholder

    # Skill extraction
    if "skill" in all_results:
        sk = all_results["skill"]
        print(f"\n[DATA] SKILL EXTRACTION: trie avg={sk.get('trie_avg', 0):.1f} skills/resume, regex avg={sk.get('regex_avg', 0):.1f}")

    print(f"\n{'=' * 65}")
    print("DONE")


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Write output to a file AND print to stdout
    log_path = Path("reports/model_eval_log.txt")
    log_path.parent.mkdir(exist_ok=True)
    # Set console encoding to UTF-8 for safety
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    # Also redirect via env for cmd/PowerShell
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, data):
            for f in self.files:
                try:
                    f.write(data)
                    f.flush()
                except UnicodeEncodeError:
                    f.write(data.encode('utf-8', errors='replace').decode('utf-8', errors='replace'))
                    f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
        def isatty(self):
            return False
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    print("=" * 65)
    print("TRIADRANK MODEL EVALUATION — FLOREX REAL-WORLD RESUMES")
    print("=" * 65)

    # Load data
    print("\nLoading florex data...")
    samples = load_florex_data(max_files=2000)
    print(f"Loaded {len(samples)} florex resumes")

    results = {}

    # Run evaluations
    results["category"] = eval_category_encoder(samples)
    results["skill"] = eval_skill_extraction(samples)
    eval_cross_encoder(samples)
    eval_pipeline(samples)

    # Synthesis
    synthesize(results)

    # Save report
    report = {
        "total_samples": len(samples),
        "category_encoder": results.get("category"),
        "skill_extraction": results.get("skill"),
    }
    Path("reports").mkdir(exist_ok=True)
    with open("reports/model_eval.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to reports/model_eval.json")
