"""Sanity check: pipeline-loaded category encoder == directly-loaded checkpoint.

History: this script originally HUNTED a discrepancy (direct load -> confident IT
0.986, pipeline -> APPAREL 0.07 on the same text). Root cause turned out to be a
test-harness foot-gun: `TriadRankPipeline(device=device)` with no checkpoint paths
silently falls back to random untrained weights (see pipeline-silent-random-fallback
memory). With checkpoint paths passed from config, pipeline.model2 MUST match a
direct CategoryEncoder.load — same state_dict, same argmax, ~identical probs.

If this ever prints NO/mismatch, the pipeline's model-loading path has drifted from
the model's own .load(); treat any divergence as a real bug, not noise.
"""
import sys, os, re
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import logging
for n in ('urllib3','httpx','huggingface_hub','transformers','models'):
    logging.getLogger(n).setLevel(logging.ERROR)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from data.loaders import clean_resume_text
from pipeline.inference import TriadRankPipeline
from models.category_encoder import CategoryEncoder
from config import config

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def strip_html(raw):
    t = re.sub(r"<[^>]+>", " ", raw); t = re.sub(r"&nbsp;|\\xa0", " ", t)
    return re.sub(r"\s+", " ", t).strip()

raw = open('datasets/external/resume_corpus/resumes_corpus/20953.txt', errors='replace').read()
text = clean_resume_text(raw)

# ---- Direct load ----
mdir = CategoryEncoder.load(config['models']['category_encoder']['save_path'], device=device); mdir.eval()
# ---- Via pipeline (pass checkpoint paths — else silent random-weight fallback) ----
pre = TriadRankPipeline(
    model1_path=config['models']['cross_encoder']['save_path'],
    model2_path=config['models']['category_encoder']['save_path'],
    device=device,
)
mpipe = pre.model2; mpipe.eval()

print("=== model objects ===")
print("  direct type:", type(mdir).__name__, "id_to_name classes:", len(getattr(mdir,'id_to_name',{})))
print("  pipe   type:", type(mpipe).__name__, "id_to_name classes:", len(getattr(mpipe,'id_to_name',{})))
print("  pre.id_to_name classes:", len(pre.id_to_name), " (config says", len(config['categories']['id_to_name']),")")
print("  direct == pipe (same object)?", mdir is mpipe)
# weight-equality probe
sd_dir = dict(mdir.state_dict()); sd_pipe = dict(mpipe.state_dict())
shared = set(sd_dir) & set(sd_pipe)
diffs = [k for k in shared if not torch.equal(sd_dir[k], sd_pipe[k]) if sd_dir[k].shape==sd_pipe[k].shape]
print("  state_dict keys equal? ", "yes" if not diffs else f"NO - {len(diffs)} differ", f" (dir {len(sd_dir)} keys, pipe {len(sd_pipe)})")

tok = pre._cat_tokenizer  # pipeline cached tokenizer
enc = tok([text], truncation=True, max_length=256, padding="max_length", return_tensors='pt')
ii, am = enc['input_ids'].to(device), enc['attention_mask'].to(device)

with torch.no_grad():
    pd = mdir.predict(ii, am)
    pp = mpipe.predict(ii, am)

import numpy as np
dprobs = pd['probabilities'][0].cpu().numpy()
pprobs = pp['probabilities'][0].cpu().numpy()
print("\n=== predictions on 20953.txt (clean_resume_text) ===")
print(f"  DIRECT: argmax idx={int(pd['predicted_class'][0].item())}  conf={float(pd['confidence'][0]):.3f}")
print(f"          probs min={dprobs.min():.4f} max={dprobs.max():.4f} argmax-prob={dprobs.max():.4f}")
print(f"          name(via dir.id_to_name): {mdir.id_to_name[int(pd['predicted_class'][0].item())]}")
print(f"  PIPE:   argmax idx={int(pp['predicted_class'][0].item())}  conf={float(pp['confidence'][0]):.3f}")
print(f"          probs min={pprobs.min():.4f} max={pprobs.max():.4f} argmax-prob={pprobs.max():.4f}")
print(f"          name(via pre.id_to_name): {pre.id_to_name.get(int(pp['predicted_class'][0].item()), 'IDX_OUT_OF_RANGE')}")
print(f"          name(via pipe.model2.id_to_name): {getattr(mpipe,'id_to_name',{}).get(int(pp['predicted_class'][0].item()),'N/A')}")
print("\n  are the prob arrays identical?", np.allclose(dprobs, pprobs, atol=1e-5))
print(f"  direct probs[diag:0..5]: {np.round(dprobs[:6],4)}")
print(f"  pipe   probs[diag:0..5]: {np.round(pprobs[:6],4)}")
