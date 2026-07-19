"""Definitive category-encoder accuracy on florex, large N, eval-matched path.

Loads via load_florex_data (stratified, proper IT/BD label mapping, _strip_html),
tokenizes exactly like the eval (max_length=256, padding=max_length), and reports:
  - accuracy (true IT/BD vs predicted)
  - predicted-category distribution (is it confident-IT/BD or collapsed-FINANCE?)
  - confidence histogram (how many resumes are collapsed <0.3?)
This settles whether the 73% eval number holds or the FINANCE-collapse is real.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import logging
for n in ('urllib3', 'httpx', 'huggingface_hub', 'transformers', 'models'):
    logging.getLogger(n).setLevel(logging.ERROR)
import torch
from collections import Counter
from transformers import AutoTokenizer
from models.category_encoder import CategoryEncoder

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
from model_eval import load_florex_data

device = 'cuda' if torch.cuda.is_available() else 'cpu'
samples = load_florex_data(max_files=400)
print(f"Loaded {len(samples)} florex samples (stratified IT/BD)")

model = CategoryEncoder.load("checkpoints/category_encoder/best_model.pt", device=device)
model.eval()
tok = AutoTokenizer.from_pretrained(model.pretrained_model_name, token=False)
id_to_name = model.id_to_name

# Map predicted 24-cat name -> IT or BD bucket (only IT and BD are florex's true categories)
IT = {"INFORMATION-TECHNOLOGY"}
BD = {"BUSINESS-DEVELOPMENT"}

pred_counter = Counter()
binned_correct = Counter()
conf_buckets = Counter()
acc_correct = 0
collapsed = 0  # confidence < 0.30
for s in samples:
    enc = tok([s["text"]], truncation=True, max_length=256, padding="max_length", return_tensors="pt")
    with torch.no_grad():
        p = model.predict(enc["input_ids"].to(device), enc["attention_mask"].to(device))
    pid = int(p["predicted_class"][0].item()); conf = float(p["confidence"][0])
    pname = id_to_name[pid]
    pred_counter[pname] += 1
    true_cat = s["category"]  # "INFORMATION-TECHNOLOGY" or "BUSINESS-DEVELOPMENT"
    # Bucket both sides to IT/BD/OTHER so the strings can be compared
    true_bucket = "IT" if true_cat in IT else ("BD" if true_cat in BD else "OTHER")
    pred_bucket = "IT" if pname in IT else ("BD" if pname in BD else "OTHER")
    ok = (pred_bucket == true_bucket)
    acc_correct += ok
    key = f"true_{true_bucket}_pred_{pred_bucket if not ok else 'OK('+pred_bucket+')'}"
    binned_correct[key] += 1
    conf_buckets[conf] = conf_buckets.get(conf, 0) + 1
    if conf < 0.30: collapsed += 1

n = len(samples)
print(f"\nAccuracy (predicted in correct IT/BD bucket): {acc_correct/n:.3f}  ({acc_correct}/{n})")
print(f"Resumes with confidence < 0.30 (collapsed): {collapsed}/{n} ({collapsed/n:.1%})")
print(f"\nPer-bucket:")
for k, v in binned_correct.items():
    print(f"  {k}: {v}")
print(f"\nPredicted-category distribution (top 8 of {len(pred_counter)}):")
for name, c in pred_counter.most_common(8):
    print(f"  {name:<24} {c:>4}  ({c/n*100:5.1f}%)")
