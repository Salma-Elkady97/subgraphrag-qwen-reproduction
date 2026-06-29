"""One-line verification entry point for both prediction files."""
from src.eval.score import aggregate
for p in ("results/webqsp_predictions.json", "results/cwq_predictions.json"):
    n, h, f = aggregate(p)
    print(f"{p}: n={n}, Hit={h:.4f}, Macro-F1={f:.4f}")
