#!/usr/bin/env bash
set -euo pipefail

# Verify Table 2 numbers from the committed per-question logs (no GPU needed).
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python -m src.eval.score    results/webqsp_predictions.json
python -m src.eval.score    results/cwq_predictions.json
python -m src.eval.bootstrap_ci results/webqsp_per_q.csv 1639 92.13 77.86
python -m src.eval.bootstrap_ci results/cwq_per_q.csv    3531 66.24 53.77
