"""Per-question Hit and F1 using the substring-match metric defined in
``src/reasoning/repair_macro_guarded_v10.py``. This is the metric whose
aggregates match the published WebQSP (89.51 / 76.39) and CWQ (66.27 / 53.78)
numbers, verified against ``results/webqsp_predictions.json`` and
``results/cwq_predictions.json``.
"""
from __future__ import annotations
import json, re

def _norm(x: str) -> str:
    x = str(x).lower().strip()
    x = re.sub(r"[^a-z0-9\s\.\-:/|,]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def _split(x: str) -> list[str]:
    out, seen = [], set()
    for p in re.split(r"\||,| and ", str(x)):
        p = _norm(p)
        if p and p not in seen:
            seen.add(p); out.append(p)
    return out

def per_question(pred: str, golds: list[str]) -> tuple[int, float]:
    preds = _split(pred)
    g = [_norm(x) for x in golds if _norm(x)]
    hit = 0
    for gt in g:
        for pr in preds if preds else [_norm(pred)]:
            if gt == pr or gt in pr or pr in gt:
                hit = 1; break
        if hit: break
    mp, mg = set(), set()
    for i, pr in enumerate(preds):
        for j, gt in enumerate(g):
            if gt == pr or gt in pr or pr in gt:
                mp.add(i); mg.add(j)
    tp = len(mp); fp = max(0, len(preds) - tp); fn = max(0, len(g) - len(mg))
    p = tp/(tp+fp) if (tp+fp) else 0.0
    r = tp/(tp+fn) if (tp+fn) else 0.0
    f1 = 2*p*r/(p+r) if (p+r) else 0.0
    return hit, f1

def aggregate(path_jsonl: str) -> tuple[int, float, float]:
    rows = [json.loads(l) for l in open(path_jsonl, encoding="utf-8") if l.strip()]
    hits, f1s = [], []
    for r in rows:
        h, f = per_question(r["prediction"], r["ground_truth"])
        hits.append(h); f1s.append(f)
    n = len(rows)
    return n, sum(hits)/n*100, sum(f1s)/n*100

if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        n, h, f = aggregate(p)
        print(f"{p}: n={n}, Hit={h:.4f}, Macro-F1={f:.4f}")
