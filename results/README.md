# Results files

| File                          | Description                                                          | Backs                                  |
| ----------------------------- | -------------------------------------------------------------------- | -------------------------------------- |
| `webqsp_per_q.csv`            | 1,639 rows: `question_id, hit (0/1), f1 (0.0-1.0)`                   | Table 2 WebQSP point estimates & CIs   |
| `cwq_per_q.csv`               | 3,531 rows: same schema                                              | Table 2 CWQ point estimates & CIs      |
| `webqsp_predictions.json`     | JSONL of `{id, question, prediction, ground_truth, prediction_before_*}` for every WebQSP test question after the full repair chain | Same as above, plus full repair history |
| `cwq_predictions.json`        | JSONL of `{id, question, prediction, ground_truth}` for every CWQ test question, no postprocessing | Same as above              |

## How they map to Table 2 of the paper

- WebQSP Hit = 92.13 = `mean(webqsp_per_q.csv:hit) * 100`
- WebQSP F1  = 77.86 = `mean(webqsp_per_q.csv:f1)  * 100`
- CWQ    Hit = 66.24 = `mean(cwq_per_q.csv:hit)   * 100`
- CWQ    F1  = 53.77 = `mean(cwq_per_q.csv:f1)    * 100`

Verified by:
```bash
python -m src.eval.score    results/webqsp_predictions.json
python -m src.eval.score    results/cwq_predictions.json
python -m src.eval.bootstrap_ci results/webqsp_per_q.csv 1639 92.13 77.86
python -m src.eval.bootstrap_ci results/cwq_per_q.csv    3531 66.24 53.77
```

The two `.json` prediction files are also the input to the bootstrap CI
computation; the metric used is the substring-style metric defined in
`src/reasoning/repair_macro_guarded_v10.py:metric`. The Hit/F1 numbers are
within ±0.01 of those reported by the canonical SubgraphRAG eval scripts in
`src/reasoning/evaluate_results.py` and `evaluate_results_corrected.py`.
