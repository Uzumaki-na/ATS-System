"""Census florex .lab sub-role labels to pick roles with enough samples for eval."""
import glob, os
from collections import Counter
labs = glob.glob('datasets/external/resume_corpus/resumes_corpus/*.lab')
c = Counter()
for lp in labs:
    for line in open(lp, errors='replace').read().splitlines():
        line = line.strip()
        if line: c[line] += 1
print(f"Total .lab files: {len(labs)}")
print(f"Distinct labels: {len(c)}\n")
print(f"{'count':>6}  label")
for lab, n in c.most_common():
    print(f"{n:>6}  {lab}")
