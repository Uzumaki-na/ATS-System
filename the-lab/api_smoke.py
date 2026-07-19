"""Backend readiness smoke: real truth-labeled resumes through the LIVE FastAPI.

Hits the running API (uvicorn api.main:app) with real resumes and asserts a
genuine ranking gap — not green-on-garbage. Cannot have the random-weight
foot-gun: the API loads checkpoints from config, so a passing result here is a
real trained-model result.

Positives: 3 florex IT resumes (external/eval corpus — NOT the training CSV).
Negatives: 3 Resume.csv resumes from ACCOUNTANT / CHEF / TEACHER.
Run from project root with the API up on :8000:
    ./.venv/Scripts/python.exe the-lab/api_smoke.py
"""
import urllib.request, urllib.parse, json, csv, glob, os, re, random, time

BASE = 'http://127.0.0.1:8000'
IT_ROLES = {'Python_Developer', 'Java_Developer', 'Database_Administrator',
            'Network_Administrator', 'Systems_Administrator', 'Security_Analyst',
            'Front_End_Developer', 'Software_Developer', 'Web_Developer'}

IT_JD = ("Senior Software Engineer — build scalable distributed systems. Python, Java, or Go. "
         "Cloud-native (AWS/GCP), containers (Docker/Kubernetes), CI/CD, REST APIs, "
         "SQL/PostgreSQL. Strong CS fundamentals, system design, and team collaboration.")
ACC_JD = ("Senior Accountant — GAAP, general ledger, month-end close, reconciliations, "
          "financial reporting, QuickBooks, Excel, audit support. CPA preferred.")


def read_lab(p):
    lab = p[:-4] + '.lab'
    if not os.path.exists(lab):
        return []
    return [l.strip() for l in open(lab, errors='replace').read().splitlines() if l.strip()]


def strip_html(raw):
    t = re.sub(r"<[^>]+>", " ", raw)
    t = re.sub(r"&nbsp;|\\xa0", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def wait_for_api(timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = urllib.request.urlopen(BASE + '/health', timeout=3)
            d = json.loads(r.read())
            if d['pipeline_ready'] and d['models_loaded']:
                return True
        except Exception:
            time.sleep(2)
    raise RuntimeError("API not ready on :8000 — start: uvicorn api.main:app --port 8000")


def post_rank(candidates, jd, cat, top_k=None):
    body = {"job_description": jd, "job_category": cat, "candidates": candidates}
    if top_k is not None:
        body["top_k"] = top_k
    req = urllib.request.Request(BASE + '/rank', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())


def rank_single(text, jd, cat):
    qs = urllib.parse.urlencode({'candidate_text': text, 'job_description': jd, 'job_category': cat})
    req = urllib.request.Request(BASE + '/rank/single?' + qs, data=b'', method='POST')
    return json.loads(urllib.request.urlopen(req, timeout=300).read())


def smoke_extract(text):
    """POST /extract (?resume_text=...) -> {skills, education, ...}. Returns (ok, info)."""
    import requests
    qs = urllib.parse.urlencode({'resume_text': text})
    r = requests.post(BASE + '/extract', params={'resume_text': text}, timeout=300)
    d = r.json()
    ok = r.status_code == 200 and isinstance(d, dict) and len(d) > 0
    return ok, d


def smoke_rank_pdf(it_pdf, neg_pdf, jd, cat):
    """POST /rank/pdf (multipart) with 1 IT + 1 non-IT PDF; assert IT ranks #1. Returns (ok, resp)."""
    import requests
    with open(it_pdf, 'rb') as fi, open(neg_pdf, 'rb') as fn:
        files = [('files', ('it.pdf', fi.read(), 'application/pdf')),
                 ('files', ('neg.pdf', fn.read(), 'application/pdf'))]
        r = requests.post(BASE + '/rank/pdf', timeout=300,
                          data={'job_description': jd, 'job_category': cat, 'top_k': '10'}, files=files)
    d = r.json()
    ok = r.status_code == 200 and d['returned_candidates'] >= 2 and d['results'][0]['candidate_id'] == 'it.pdf'
    return ok, d


def main():
    # 3 florex IT positives (external corpus) — HARD cap at 3.
    files = sorted(glob.glob('datasets/external/resume_corpus/resumes_corpus/*.txt'))
    its = []
    for fp in files:
        lab = read_lab(fp)
        if lab and any(r in IT_ROLES for r in lab):
            its.append(('florex', lab, strip_html(open(fp, errors='replace').read())[:4000]))
        if len(its) >= 3:
            break
    assert len(its) == 3, f"need 3 IT florex positives, got {len(its)}"

    # 3 non-IT negatives from the real Resume.csv (ACCOUNTANT / CHEF / TEACHER).
    wanted = ['ACCOUNTANT', 'CHEF', 'TEACHER']
    negs = []
    seen = set()
    with open('datasets/category_resumes/Resume.csv', encoding='utf-8', errors='replace') as f:
        for row in csv.DictReader(f):
            cat = row['Category']
            if cat in wanted and cat not in seen and row['Resume_str'].strip():
                negs.append((cat, row['Resume_str'][:4000]))
                seen.add(cat)
            if len(negs) >= 3:
                break
    assert len(negs) == 3, f"need 3 non-IT negatives, got {len(negs)}"

    wait_for_api()
    print("API ready.\n")

    candidates = [{"id": f"IT-{i+1}", "text": t} for i, (_, _, t) in enumerate(its)] + \
                 [{"id": f"NEG-{c[:4]}-{i+1}", "text": t} for i, (c, t) in enumerate(negs)]

    print("=== /rank batch (IT JD, job_category=INFORMATION-TECHNOLOGY) ===")
    print(f"candidates: {len(candidates)} | " +
          ", ".join(f"IT-{i+1}" for i in range(3)) + " (truth IT) | " +
          ", ".join(negs[i][0] for i in range(3)) + " (truth non-IT)")
    resp = post_rank(candidates, IT_JD, 'INFORMATION-TECHNOLOGY', top_k=6)
    assert resp['returned_candidates'] == 6, f"expected 6 returned, got {resp['returned_candidates']}"
    for r in resp['results']:
        truth = 'IT   ' if r['candidate_id'].startswith('IT-') else 'nonIT'
        print(f"  rank{r['rank']} {r['candidate_id']:<14} truth={truth} "
              f"final={r['final_score']:.4f} raw={r['raw_score']:.4f} "
              f"pred={r['category']['predicted']:<22} match={str(r['category']['match']):<5} "
              f"label={r['label']} skill_ov={r['skill_overlap']:.2f}")
    positions = {r['candidate_id']: r['rank'] for r in resp['results']}
    it_ranks = sorted(v for k, v in positions.items() if k.startswith('IT-'))
    neg_ranks = sorted(v for k, v in positions.items() if not k.startswith('IT-'))
    it_scores = [r['final_score'] for r in resp['results'] if r['candidate_id'].startswith('IT-')]
    neg_scores = [r['final_score'] for r in resp['results'] if not r['candidate_id'].startswith('IT-')]
    gap_ok = min(it_scores) > max(neg_scores)
    order_ok = max(it_ranks) < min(neg_ranks)
    print(f"\n  IT ranks={it_ranks} | neg ranks={neg_ranks}")
    print(f"  all-IT-above-all-nonIT? {order_ok} | gap min(IT)={min(it_scores):.4f} > max(neG)={max(neg_scores):.4f}? {gap_ok}")

    # /rank/single: same IT resume, matched vs mismatched category.
    a = rank_single(its[0][2], IT_JD, 'INFORMATION-TECHNOLOGY')
    b = rank_single(its[0][2], ACC_JD, 'ACCOUNTANT')
    print("\n=== /rank/single (florex IT resume) ===")
    print(f"  under IT  : final={a['final_score']:.4f} raw={a['raw_score']:.4f} "
          f"pred={a['category_predicted']} match={a['category_match']} skills={len(a['skills'])} exp_yr={a['experience_years']}")
    print(f"  under ACC : final={b['final_score']:.4f} raw={b['raw_score']:.4f} "
          f"pred={b['category_predicted']} match={b['category_match']}")
    matched_higher = a['final_score'] > b['final_score']
    print(f"  matched(IT) > mismatched(ACC)? {matched_higher} (gap {a['final_score']-b['final_score']:.4f})")

    # /extract on the same IT resume.
    ex_ok, ex = smoke_extract(its[0][2])
    print("\n=== /extract (florex IT resume) ===")
    if ex_ok:
        print(f"  keys={list(ex.keys())}")
        sk = ex.get('skills')
        if isinstance(sk, dict):
            n = len(sk)
            sample = list(sk.keys())[:5] if all(isinstance(k, str) for k in sk) else list(sk.items())[:3]
        elif isinstance(sk, list):
            n = len(sk)
            sample = [s.get('skill', s) if isinstance(s, dict) else s for s in sk[:5]]
        else:
            n, sample = 0, []
        print(f"  skills count={n}  sample={sample}")
        print(f"  experience_years={ex.get('experience_years')}  education_count={len(ex.get('education', []))}")
        print(f"  extract_ok={ex_ok}")
    else:
        print(f"  extract_ok={ex_ok}  resp={str(ex)[:300]}")

    # /rank/pdf: 1 IT PDF vs 1 ACCOUNTANT PDF, IT JD.
    it_pdf = 'datasets/resume_pdfs/INFORMATION-TECHNOLOGY/10089434.pdf'
    neg_pdf = 'datasets/resume_pdfs/ACCOUNTANT/10554236.pdf'
    pdf_ok, pdf_resp = smoke_rank_pdf(it_pdf, neg_pdf, IT_JD, 'INFORMATION-TECHNOLOGY')
    print("\n=== /rank/pdf (1 IT pdf + 1 ACCOUNTANT pdf, IT JD) ===")
    if pdf_ok or isinstance(pdf_resp, dict) and 'results' in pdf_resp:
        for r in pdf_resp['results']:
            print(f"  rank{r['rank']} {r['candidate_id']:<8} final={r['final_score']:.4f} "
                  f"pred={r['category']['predicted']:<22} match={str(r['category']['match']):<5}")
        pdf_ok = pdf_resp['results'][0]['candidate_id'] == 'it.pdf' and \
                 pdf_resp['results'][0]['final_score'] > pdf_resp['results'][1]['final_score']
    else:
        print(f"  pdf_ok={pdf_ok}  resp={str(pdf_resp)[:400]}")
    print(f"  pdf_ok={pdf_ok} (IT pdf ranked #1 above accountant)")

    print("\n=== VERDICT ===")
    ok = order_ok and gap_ok and matched_higher and ex_ok and pdf_ok
    print(f"  /rank batch sane (IT>nonIT+gap): {order_ok and gap_ok}")
    print(f"  /rank/single category-gate: {matched_higher}")
    print(f"  /extract returns entity dict: {ex_ok}")
    print(f"  /rank/pdf ranks IT pdf #1: {pdf_ok}")
    print(f"  OVERALL backend ready on all 5 endpoints? {'YES' if ok else 'NO'}")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
