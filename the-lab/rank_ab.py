"""END-TO-END ranking quality A/B: weighted vs legacy scoring.

For N IT resumes that the category encoder CORRECTLY classifies as IT (= the
category gate actually works, penalty=1.0 on match), run the full pipeline
against 6 JDs and check: does the IT JD rank #1? (per-resume argmax)
Also report mean rank position of the IT JD. Run BOTH scoring modes on the same
resumes + scaling, so it's a fair A/B, not a sample-size artifact.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import logging
for n in ('pipeline.inference', 'urllib3', 'httpx', 'huggingface_hub', 'transformers', 'models'):
    logging.getLogger(n).setLevel(logging.ERROR)
import glob, random, torch
from pipeline.inference import TriadRankPipeline

device = 'cuda' if torch.cuda.is_available() else 'cpu'

IT = 'INFORMATION-TECHNOLOGY'
JDS = {
    'IT':   ('INFORMATION-TECHNOLOGY', 'Senior Software Engineer. Strong Python/Java/Go, cloud (AWS/GCP), Docker/Kubernetes, CI/CD, distributed systems, system design. 5+ years building scalable backend services. CS degree.'),
    'BD':   ('BUSINESS-DEVELOPMENT',   'Senior Project Manager. Lead cross-functional teams, manage budgets and timelines, stakeholder communication. PMP/Agile/Scrum. 8+ years delivering enterprise programs.'),
    'FIN':  ('FINANCE',                'Financial Analyst. Financial modeling, forecasting, reporting. Excel, Bloomberg, SQL. CFA or MBA preferred. Investment banking exposure.'),
    'HEAL': ('HEALTHCARE',             'Registered Nurse. Patient care, medication administration, care planning, charting. BLS/ACLS certified. Acute-care experience.'),
    'SALES':('SALES',                  'Enterprise Account Executive. Pipeline management in Salesforce CRM, revenue growth, stakeholder negotiation, forecasting.'),
    'ENG':  ('ENGINEERING',            'Mechanical Engineer. CAD (SolidWorks), FEA, GD&T, prototyping, DFM, manufacturing processes. BSME.'),
}
JD_ORDER = list(JDS.keys())  # IT is index 0

# Find IT florex resumes the category encoder classifies as IT (= gate works)
files = sorted(glob.glob('datasets/external/resume_corpus/resumes_corpus/*.txt'))
def read_lab(p):
    lab = p[:-4] + '.lab'
    return open(lab, errors='replace').read().strip() if os.path.exists(lab) else '?'

print(f"Scanning {len(files)} florex files for IT resumes...")
from config import config
pre = TriadRankPipeline(
    model1_path=config['models']['cross_encoder']['save_path'],
    model2_path=config['models']['category_encoder']['save_path'],
    device=device,
)
def pred_cat(text):
    from data.loaders import clean_resume_text
    t = clean_resume_text(text)
    enc = pre._cat_tokenizer([t], truncation=True, max_length=256, padding=True, return_tensors='pt')
    with torch.no_grad():
        p = pre.model2.predict(enc['input_ids'].to(device), enc['attention_mask'].to(device))
    return pre.id_to_name[int(p['predicted_class'][0].item())], float(p['confidence'][0].item())

def clean(raw):
    import re
    t = re.sub(r"<[^>]+>", " ", raw); t = re.sub(r"&nbsp;|\\xa0", " ", t)
    return re.sub(r"\s+", " ", t).strip()

candidate_pool = []
rng = random.Random(42)
scanned = 0; passed_label = 0; pred_it = 0
for fp in rng.sample(files, min(400, len(files))):
    scanned += 1
    lab = read_lab(fp)
    if not any(s in lab for s in ('Developer', 'IT', 'Administrator', 'Engineer')):
        continue
    passed_label += 1
    raw = open(fp, errors='replace').read()
    pname, conf = pred_cat(raw)
    if pname == IT:
        pred_it += 1
        candidate_pool.append((fp, raw, pname, conf))
        if passed_label <= 3:
            print(f"  [debug] {os.path.basename(fp)} lab='{lab}' pred={pname}({conf:.2f}) len={len(raw)}")
    elif passed_label <= 5:
        print(f"  [debug-nIT] {os.path.basename(fp)} lab='{lab}' pred={pname}({conf:.2f}) len={len(raw)}")
    if len(candidate_pool) >= 40:
        break
print(f"[diagnostic] scanned={scanned} passed_label_filter={passed_label} predicted_IT={pred_it} kept={len(candidate_pool)}")
print(f"Found {len(candidate_pool)} IT-correctly-classified resumes. Running A/B...\n")

def run_ab(pipe, label):
    it_rank1 = 0; ranks = []; scores_by_jd = {k: [] for k in JD_ORDER}
    for fp, raw, _, _ in candidate_pool:
        rt = clean(raw)[:4000]
        per_jd = {}
        for jd_key, (cat_name, jd_text) in JDS.items():
            res = pipe.rank_single(resume_text=rt, job_description=jd_text, job_category=cat_name)
            per_jd[jd_key] = res.final_score
            scores_by_jd[jd_key].append(res.final_score)
        ranked = sorted(per_jd, key=lambda k: -per_jd[k])
        rank_of_it = ranked.index('IT') + 1
        ranks.append(rank_of_it)
        if rank_of_it == 1: it_rank1 += 1
    n = len(candidate_pool)
    print(f"=== {label} ===")
    print(f"  IT JD ranked #1 : {it_rank1}/{n}  ({it_rank1/n:.1%})")
    print(f"  mean rank of IT JD (1=best, 6=worst): {sum(ranks)/n:.2f}")
    print(f"  rank distribution: " + ' '.join(f"r{i}:{ranks.count(i)}" for i in range(1, 7)))
    print(f"  mean final_score per JD:")
    for k in JD_ORDER:
        vals = scores_by_jd[k]; print(f"     {k:<5} {sum(vals)/len(vals):.4f}  (min {min(vals):.3f} max {max(vals):.3f})")
    print()
    return it_rank1, ranks

# Run WEIGHTED (current config)
i1, r1 = run_ab(pre, "WEIGHTED (content+.50 category+.30 cross+.20)")

# Switch to LEGACY in-place, same pipeline object (same state) -> fair A/B
pre.scoring_mode = 'legacy'
i2, r2 = run_ab(pre, "LEGACY (raw_score * penalty)")

print("=== VERDICT ===")
winner = "WEIGHTED" if i1 > i2 else ("LEGACY" if i2 > i1 else "TIE")
print(f"  IT-rank-1:  weighted={i1}/{len(candidate_pool)}  legacy={i2}/{len(candidate_pool)}  -> {winner}")
print(f"  mean IT-rank: weighted={sum(r1)/len(r1):.2f}  legacy={sum(r2)/len(r2):.2f}")
