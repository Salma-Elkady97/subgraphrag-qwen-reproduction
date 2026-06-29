"""95% bootstrap confidence intervals for Hit and Macro-F1 from per-question CSVs.

Usage:
    python -m src.eval.bootstrap_ci results/webqsp_per_q.csv 1639 92.13 77.86
    python -m src.eval.bootstrap_ci results/cwq_per_q.csv    3531 66.24 53.77

This script reproduces the CIs reported in Table 2 of the paper.
"""
from __future__ import annotations
import csv, sys, random, statistics

def load(path):
    h, f = [], []
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if not r.get("hit") or not r.get("f1"): continue
            try: h.append(int(r["hit"])); f.append(float(r["f1"]))
            except (ValueError, TypeError): continue
    return h, f

def ci(vals, n_resamples=10000, alpha=0.05):
    n = len(vals); rng = random.choices
    means = sorted(sum(rng(vals, k=n))/n for _ in range(n_resamples))
    lo = means[int(alpha/2 * n_resamples)]
    hi = means[int((1-alpha/2) * n_resamples) - 1]
    return lo*100, hi*100, statistics.mean(means)*100

def main(argv):
    path, exp_n, exp_hit, exp_f1 = argv[0], int(argv[1]), float(argv[2]), float(argv[3])
    seed = int(argv[4]) if len(argv) > 4 else 20260603
    random.seed(seed)
    hits, f1s = load(path)
    n = len(hits)
    mh = sum(hits)/n*100; mf = sum(f1s)/n*100
    assert n == exp_n, f"row-count mismatch {n}!={exp_n}"
    assert abs(mh - exp_hit) < 0.05, f"Hit mismatch {mh:.2f}!={exp_hit}"
    assert abs(mf - exp_f1) < 0.05, f"F1 mismatch {mf:.2f}!={exp_f1}"
    hlo, hhi, _ = ci(hits)
    flo, fhi, _ = ci(f1s)
    print(f"{path}: Hit={mh:.2f} [{hlo:.2f}, {hhi:.2f}] | F1={mf:.2f} [{flo:.2f}, {fhi:.2f}]")

if __name__ == "__main__":
    main(sys.argv[1:])
