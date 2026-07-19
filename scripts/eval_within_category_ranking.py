"""Within-category fine-ranking regression eval: weighted vs legacy, with shuffle control.

Regression check that the pipeline can ORDER resumes of the SAME category by fit to a
role-specific JD (not just pick the right category). Run from project root.

Ground truth: florex .lab sub-role labels. For a role-specific JD (job_category held
to INFORMATION-TECHNOLOGY so penalty=1.0, isolating content+cross signal):
  positives = IT-predicted pool resumes carrying that sub-role
  negatives = IT-predicted pool resumes WITHOUT that sub-role
  AUC = Mann-Whitney P(positive outscores a random negative); ties 0.5. Random=0.5.

Both scoring modes are computed from ONE rank_single call (penalty~1.0 for all):
  weighted = res.final_score  (= 0.5*content + 0.3*pen + 0.2*raw)
  legacy   = res.raw_score * penalty_factor
The shuffle control permutes the SAME score arrays 200x (no re-scoring) — if control
AUC stays ~0.5 while real AUC is high, the signal is genuine, not a metric artifact.

See memories: [pipeline-silent-random-fallback] (why checkpoint paths are passed),
              [ats-weighted-scoring] (what this eval established).
"""
import os, sys, glob, random, re
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import logging
for n in ('pipeline.inference', 'urllib3', 'httpx', 'huggingface_hub', 'transformers', 'models'):
    logging.getLogger(n).setLevel(logging.ERROR)
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from pipeline.inference import TriadRankPipeline
from data.loaders import clean_resume_text

device = 'cuda' if torch.cuda.is_available() else 'cpu'

ROLE_JDS = {
    'Database_Administrator': 'Database Administrator. Manage Microsoft SQL Server / PostgreSQL / MySQL, backup and recovery, performance tuning, indexing, replication, T-SQL and PL/SQL, ETL, data warehousing, high availability (AlwaysOn, mirroring), DBA certification preferred.',
    'Network_Administrator':  'Network Administrator. Manage LAN/WAN, Cisco routers and switches, firewalls, VPN, TCP/IP, routing protocols OSPF/BGP, network monitoring (SolarWinds, PRTG), CCNA/CCNP, Windows and Linux servers, VLANs, wireless.',
    'Security_Analyst':       'Information Security Analyst. SIEM (Splunk, QRadar), incident response, vulnerability scanning (Nessus), penetration testing, ISO 27001, NIST CSF, firewall policies, IDS/IPS, threat intelligence, endpoint detection, CISSP.',
    'Python_Developer':       'Python Developer. Python 3, Django / Flask / FastAPI, pandas / NumPy, REST APIs, PostgreSQL, Celery / Redis, Docker, unit testing (pytest), data pipelines, Airflow.',
    'Java_Developer':         'Java Developer. Java / Spring Boot, J2EE, Maven / Gradle, Hibernate, JPA, REST APIs, JUnit / Mockito, Kafka, microservices, JVM tuning, Tomcat.',
    'Front_End_Developer':    'Frontend Developer. JavaScript / TypeScript, React / Vue / Angular, HTML5 / CSS3, responsive design, webpack / Vite, REST / GraphQL APIs, accessibility (WCAG), Tailwind / SASS, Jest.',
    'Systems_Administrator':  'Systems Administrator. Linux and Windows server administration, Active Directory, Group Policy, VMware / Hyper-V, PowerShell and Bash scripting, patch management, backups, monitoring (Nagios, Zabbix), DNS, DHCP.',
}
IT = 'INFORMATION-TECHNOLOGY'
POOL_TARGET = 250      # IT-predicted resumes to pool
N_NEG_CAP = 100        # cap negatives per role
N_GENES = 200          # shuffle relabelings for control
MIN_POS = 8            # skip a role if fewer positives than this


def _strip_html(raw: str) -> str:
    t = re.sub(r"<[^>]+>", " ", raw)
    t = re.sub(r"&nbsp;|\\xa0", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _read_lab(txt_path: str):
    lab = txt_path[:-4] + '.lab'
    if not os.path.exists(lab):
        return []
    return [l.strip() for l in open(lab, errors='replace').read().splitlines() if l.strip()]


def auc(pos, neg) -> float:
    """Mann-Whitney: P(pos > neg), ties counted 0.5."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    ct = 0.0
    for p in pos:
        ct += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return ct / (len(pos) * len(neg))


# self-checks (cheap, protects against a silent AUC regression)
assert abs(auc([0.9, 0.8], [0.2, 0.1]) - 1.0) < 1e-9
assert abs(auc([0.5, 0.5], [0.5, 0.5]) - 0.5) < 1e-9
assert abs(auc([0.1, 0.2], [0.8, 0.9]) - 0.0) < 1e-9


def build_pool(pipe, max_scan=1200):
    """IT-predicted florex resumes: (stripped_text, labels) list."""
    files = sorted(glob.glob('datasets/external/resume_corpus/resumes_corpus/*.txt'))
    rng = random.Random(42)
    pool, n_scan = [], 0
    for fp in rng.sample(files, min(max_scan, len(files))):
        n_scan += 1
        lab = _read_lab(fp)
        if not lab:
            continue
        raw = open(fp, errors='replace').read()
        t = clean_resume_text(raw)
        enc = pipe._cat_tokenizer([t], truncation=True, max_length=256, padding=True, return_tensors='pt')
        with torch.no_grad():
            p = pipe.model2.predict(enc['input_ids'].to(device), enc['attention_mask'].to(device))
        if pipe.id_to_name[int(p['predicted_class'][0].item())] == IT:
            pool.append((_strip_html(raw), lab))
        if len(pool) >= POOL_TARGET:
            break
    return pool, n_scan


def shuffled_control_auc(scores, npos, n_gen, rng):
    """AUC under random positive/negative relabeling (no re-scoring)."""
    s = np.asarray(scores, float)
    out = []
    for _ in range(n_gen):
        perm = rng.sample(range(len(s)), len(s))
        sc = s[perm]
        out.append(auc(sc[:npos], sc[npos:]))
    return float(np.mean(out)), float(np.std(out))


def main():
    print("Loading pipeline with TRAINED checkpoints "
          "(passing paths — see pipeline-silent-random-fallback memory)...")
    pipe = TriadRankPipeline(
        model1_path=config['models']['cross_encoder']['save_path'],
        model2_path=config['models']['category_encoder']['save_path'],
        device=device,
    )

    print(f"Building IT-predicted florex pool (target {POOL_TARGET})...")
    pool, n_scan = build_pool(pipe)
    print(f"Pool: scanned={n_scan}, IT-predicted kept={len(pool)}\n")

    rng2 = random.Random(7)
    hdr = (f"{'role':<24} | {'n_pos':>5} {'n_neg':>5} | "
           f"{'AUC_w':>6} {'ctrl_w':>14} | {'AUC_l':>6} {'ctrl_l':>14} | "
           f"{'top10_w':>7} {'top10_l':>7} | winner")
    print(hdr)
    print('-' * 138)
    rows = []
    for R, jd in ROLE_JDS.items():
        pos_idx = [i for i, (_, lab) in enumerate(pool) if R in lab]
        neg_idx = [i for i, (_, lab) in enumerate(pool) if R not in lab]
        if len(pos_idx) < MIN_POS:
            print(f"{R:<24} | too few positives ({len(pos_idx)}); skip")
            continue
        if len(neg_idx) > N_NEG_CAP:
            neg_idx = rng2.sample(neg_idx, N_NEG_CAP)
        idx = pos_idx + neg_idx
        npos = len(pos_idx)

        # score each candidate ONCE; derive both modes from the same call
        w_scr, l_scr = [], []
        for i in idx:
            txt = pool[i][0][:4000]
            res = pipe.rank_single(resume_text=txt, job_description=jd, job_category=IT)
            pen = res.metadata.get('penalty_factor', 1.0)
            w_scr.append(res.final_score)
            l_scr.append(res.raw_score * pen)
        w_scr = np.asarray(w_scr); l_scr = np.asarray(l_scr)

        a_w = auc(w_scr[:npos], w_scr[npos:])
        a_l = auc(l_scr[:npos], l_scr[npos:])
        # control from the SAME arrays — no re-scoring
        rng_ctl = random.Random(123 + len(rows))
        cw_m, cw_s = shuffled_control_auc(w_scr, npos, min(N_GENES, 200), rng_ctl)
        cl_m, cl_s = shuffled_control_auc(l_scr, npos, min(N_GENES, 200), rng_ctl)

        def topk(scr, k=10):
            order = np.argsort(-scr)
            return sum(1 for j in order[:k] if j < npos) / min(k, len(order))
        tk_w, tk_l = topk(w_scr), topk(l_scr)

        w_valid = a_w > cw_m + 3 * cw_s
        l_valid = a_l > cl_m + 3 * cl_s
        winner = 'WEIGHTED' if a_w > a_l + 0.01 else ('LEGACY' if a_l > a_w + 0.01 else 'TIE')
        print(f"{R:<24} | {npos:>5} {len(neg_idx):>5} | "
              f"{a_w:>6.3f} {cw_m:>8.3f}+/-{cw_s:<4.3f} {'OK' if w_valid else '??'!s:<3}| "
              f"{a_l:>6.3f} {cl_m:>8.3f}+/-{cl_s:<4.3f} {'OK' if l_valid else '??'!s:<3}| "
              f"{tk_w:>7.2f} {tk_l:>7.2f} | {winner}")
        rows.append((R, a_w, a_l, w_valid, l_valid))

    if rows:
        aw = np.mean([r[1] for r in rows]); al = np.mean([r[2] for r in rows])
        allvalid = all(r[3] and r[4] for r in rows)
        print('-' * 138)
        print(f"{'MACRO MEAN':<24} | {'':>5} {'':>5} | {aw:>6.3f} {'':>14} | {al:>6.3f} {'':>14} | {'':>7} {'':>7} | "
              f"{'WEIGHTED' if aw > al + 0.01 else ('LEGACY' if al > aw + 0.01 else 'TIE')}")
        print(f"\nRandom-chance AUC = 0.5.  Weighted macro AUC={aw:.3f}, Legacy macro AUC={al:.3f}.")
        print(f"Shuffle control valid (real AUC > control_mean+3sigma) for ALL roles: {allvalid}")
        if not allvalid:
            print("WARNING: at least one role failed the shuffle control — the AUC signal there is suspect.")


if __name__ == '__main__':
    main()
