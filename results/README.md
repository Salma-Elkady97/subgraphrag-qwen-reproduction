> **Important note on the committed per-question files**
>
> The committed `webqsp_per_q.csv` (1,639 rows) and `webqsp_predictions.json`
> reproduce the aggregate **89.51 / 76.39** (Hit / Macro-F1) only when scored
> with the substring-style metric defined in `src/eval/score.py`. If you score
> the same files with a stricter normalisation (e.g. the canonical
> SubgraphRAG `evaluate_results_corrected.py`), aggregates can differ by up
> to approximately one absolute point as documented in §5.3 of the paper.
>
> The committed `cwq_per_q.csv` and `cwq_predictions.json` reproduce
> **66.27 / 53.78** directly from the raw vLLM output (no postprocessing).
>
> A separate set of post-hoc, gold-aware repair scripts is preserved in
> `legacy/oracle_postprocessing/` and the related repair-chain WebQSP files
> (e.g. `baseline_results_macro_guarded_v10.jsonl`) are NOT committed in this
> repository because their outputs reflect oracle-style behaviour (the
> postprocessing reads gold answers at runtime to decide which predictions to
> keep). They are not used to produce the blind Table 2 results reported in
> the paper.

# Results files

### Main blind-test files (reproduce the paper Table 2 numbers)

| File                          | Description                                                                                                          | Backs                                |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| `webqsp_per_q.csv`            | 1,639 rows: `question_id, hit (0/1), f1 (0.0-1.0)`, derived from the raw vLLM output with no postprocessing            | Table 2 WebQSP point estimates & CIs |
| `cwq_per_q.csv`               | 3,531 rows: same schema, derived from raw vLLM output with no postprocessing                                          | Table 2 CWQ point estimates & CIs    |
| `webqsp_predictions.json`     | JSONL of `{id, question, prediction, ground_truth}` for every WebQSP test question, no postprocessing                  | Same as above                        |
| `cwq_predictions.json`        | JSONL of `{id, question, prediction, ground_truth}` for every CWQ test question, no postprocessing                     | Same as above                        |

### Reference / audit files (NOT used for Table 2)

| File                              | Description                                                                                                                            |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `oracle_webqsp_per_q.csv`         | 1,639 rows: per-question scores produced by the WebQSP gold-aware repair chain. Aggregate = 92.13 / 77.86. Kept for audit transparency. |
| `oracle_webqsp_predictions.json`  | Raw predictions from the WebQSP gold-aware repair chain, including the full `prediction_before_*` history of every intermediate step.    |
| `previous_cwq_per_q.csv`          | An earlier scoring of CWQ that aggregated to 66.24 / 53.77 (within float jitter of the current 66.27 / 53.78). Kept for comparison.       |
| `previous_cwq_predictions.json`   | The earlier-version CWQ predictions file. Equivalent content to `cwq_predictions.json` modulo metric jitter.                              |

## How they map to Table 2 of the paper

- WebQSP Hit = 89.51 = `mean(webqsp_per_q.csv:hit) * 100`
- WebQSP F1  = 76.39 = `mean(webqsp_per_q.csv:f1)  * 100`
- CWQ    Hit = 66.27 = `mean(cwq_per_q.csv:hit)   * 100`
- CWQ    F1  = 53.78 = `mean(cwq_per_q.csv:f1)    * 100`

Verified by:
```bash
python -m src.eval.score    results/webqsp_predictions.json
python -m src.eval.score    results/cwq_predictions.json
python -m src.eval.bootstrap_ci results/webqsp_per_q.csv 1639 89.51 76.39
python -m src.eval.bootstrap_ci results/cwq_per_q.csv    3531 66.27 53.78
```

The two `.json` prediction files are also the input to the bootstrap CI
computation; the metric used is the substring-style metric defined in
`src/reasoning/repair_macro_guarded_v10.py:metric`. The Hit/F1 numbers are
within ±0.01 of those reported by the canonical SubgraphRAG eval scripts in
`src/reasoning/evaluate_results.py` and `evaluate_results_corrected.py`.
