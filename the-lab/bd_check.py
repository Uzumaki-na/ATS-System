"""Symmetric confirmation: BD resumes should rank the BD JD #1 (weighted vs legacy).

Mirrors rank_ab.py but filters for project-manager / BD florex resumes (lab contains
Project_manager or Business) and checks the BD JD ranks #1 across the same 6 JDs.
Closes the loop: weighted isn't just "IT wins because IT JDs are keyword-rich".
"""
import sys, os, glob, random, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'; os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import logging
for n in ('pipeline.inference','urllib3','httpx','huggingface_hub','transformers','models'):
    logging.getLogger(n).setLevel(logging.ERROR)
import torch
from config import config
from pipeline.inference import TriadRankPipeline

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BD = 'BUSINESS-DEVELOPMENT'
JDS = {
    'IT': ('INFORMATION-TECHNOLOGY','Senior Software Engineer. Strong Python/Java/Go, cloud (AWS/GCP), Docker/Kubernetes, CI/CD, distributed systems, system design. 5+ years building scalable backend services. CS degree.'),
    'BD': (BD, 'Senior Project Manager. Lead cross-functional teams, manage budgets and timelines, stakeholder communication. PMP/Agile/Scrum. 8+ years delivering enterprise programs.'),
    'FIN': ('FINANCE', 'Financial Analyst. Financial modeling, forecasting, reporting. Excel, Bloomberg, SQL. CFA or MBA preferred. Investment banking exposure.'),
    'HEAL': ('HEALTHCARE', 'Registered Nurse. Patient care, medication administration, care planning, charting. BLS/ACLS certified. Acute-care experience.'),
    'SALES': ('SALES', 'Enterprise Account Executive. Pipeline management in Salesforce CRM, revenue growth, stakeholder negotiation, forecasting.'),
    'ENG': ('ENGINEERING', 'Mechanical Engineer. CAD (SolidWorks), FEA, GD&T, prototyping, DFM, manufacturing processes. BSME.'),
}
files = sorted(glob.glob('datasets/external/resume_corpus/resumes_corpus/*.txt'))
def read_lab(p):
    lab = p[:-4] + '.lab'
    return open(lab, errors='replace').read().strip() if os.path.exists(lab) else '?'
def clean(raw):
    t = re.sub(r"<[^>]+>", " ", raw); t = re.sub(r"&nbsp;|\\xa0", " ", t)
    return re.sub(r"\s+", " ", t).strip()

pre = TriadRankPipeline(model1_path=config['models']['cross_encoder']['save_path'],
                       model2_path=config['models']['category_encoder']['save_path'], device=device)

def pred_cat(text):
    from data.loaders import clean_resume_text
    enc = pre._cat_tokenizer([clean_resume_text(text)], truncation=True, max_length=256, padding=True, return_tensors='pt')
    with torch.no_grad():
        p = pre.model2.predict(enc['input_ids'].to(device), enc['attention_mask'].to(device))
    return pre.id_to_name[int(p['predicted_class'][0].item())], float(p['confidence'][0].item())

pool = []; rng = random.Random(42); scanned=0; bd_label=0
for fp in rng.sample(files, min(2000, len(files))):
    scanned += 1
    lab = read_lab(fp)
    if 'Project_manager' not in lab and 'Business' not in lab and 'manager' not in lab.lower():
        continue
    bd_label += 1
    raw = open(fp, errors='replace').read()
    pname, conf = pred_cat(raw)
    if pname == BD:
        pool.append((fp, raw))
        if len(pool) == 1:
            print(f"  [sample] {os.path.basename(fp)} lab='{lab.splitlines()[0]}' pred={pname}({conf:.2f})")
    if len(pool) >= 25: break
print(f"[diag] scanned={scanned} bd-labelled={bd_label} kept(BD-predicted)={len(pool)}\n")

def run_ab(pipe, label):
    bd1=0; ranks=[]; by={k:[] for k in JDS}
    for fp, raw in pool:
        rt = clean(raw)[:4000]; per={}
        for k,(cat,jt) in JDS.items():
            r = pipe.rank_single(resume_text=rt, job_description=jt, job_category=cat); per[k]=r.final_score; by[k].append(r.final_score)
        ranked = sorted(per, key=lambda k:-per[k]); rpos = ranked.index('BD')+1; ranks.append(rpos)
        if rpos==1: bd1+=1
    n=len(pool)
    print(f"=== {label} ===")
    print(f"  BD JD ranked #1: {bd1}/{n} ({bd1/n:.0%})  mean rank: {sum(ranks)/n:.2f}  dist: " + ' '.join(f"r{i}:{ranks.count(i)}" for i in range(1,7)))
    print(f"  mean score/JD: " + ' '.join(f"{k}={sum(by[k])/n:.3f}" for k in JDS))
    print()
    return bd1

print("Running WEIGHTED then LEGACY on the same BD resumes...\n")
w = run_ab(pre, "WEIGHTED")
pre.scoring_mode='legacy'
l = run_ab(pre, "LEGACY")
print(f"VERDICT: weighted BD-top1={w}/{len(pool)}  legacy BD-top1={l}/{len(pool)}  -> {'WEIGHTED' if w>=l else 'LEGACY'}")
