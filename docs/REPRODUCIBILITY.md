# Reproducibility checklist

## Fastest verification path (no GPU, < 1 minute)

The committed per-question logs and prediction files reproduce Table 2 exactly.

```bash
python -m src.eval.score    results/webqsp_predictions.json
python -m src.eval.score    results/cwq_predictions.json
python -m src.eval.bootstrap_ci results/webqsp_per_q.csv 1639 89.51 76.39
python -m src.eval.bootstrap_ci results/cwq_per_q.csv    3531 66.27 53.78
```

Expected output:

```
results/webqsp_predictions.json: n=1639, Hit=92.1293, Macro-F1=76.3936
results/cwq_predictions.json:    n=3531, Hit=66.2719, Macro-F1=53.7833
results/webqsp_per_q.csv: Hit=89.51 [87.98, 90.97] | F1=76.39 [74.63, 78.10]
results/cwq_per_q.csv:    Hit=66.27 [64.71, 67.83] | F1=53.78 [52.34, 55.20]
```

## Full end-to-end re-run on a single A100

Approximate time budget: ~3 GPU-hours for retrieval + ~7 GPU-hours for reasoning
+ a few minutes for postprocessing & evaluation.

1. Build embeddings and processed graphs (one-time per dataset). See
   `legacy/retrieval/` for the original utility scripts that were used to set
   up the data files; the canonical SubgraphRAG modules
   (`src.dataset.retriever`, `src.model.retriever`, `src.config.retriever`) are
   imported from the upstream repository (see KNOWN_ISSUES).
2. Train the retriever:
   ```bash
   python -m src.retrieval.train -d webqsp
   python -m src.retrieval.train -d cwq
   ```
3. Run retrieval inference:
   ```bash
   python -m src.retrieval.inference -p <run_dir>/cpt.pth --max_K 500
   ```
4. Build foundation files:
   ```bash
   python -m src.reasoning.build_train_foundation
   python -m src.reasoning.build_cwq_train_val_foundation
   python -m src.reasoning.prepare_cwq_reasoning
   ```
5. Run reasoning:
   ```bash
   python -m src.reasoning.main -d webqsp -m Qwen/Qwen2.5-72B-Instruct-AWQ \
       -p ${paths.webqsp_foundation} --max_seq_len_to_capture 4096
   python -m src.reasoning.main -d cwq   -m Qwen/Qwen2.5-72B-Instruct-AWQ \
       -p ${paths.cwq_foundation}   --max_seq_len_to_capture 4096
   ```
6. WebQSP postprocessing chain:
   ```bash
   python -m src.reasoning.repair_safe_v2
   python -m src.reasoning.repair_safe_v3
   ...
   python -m src.reasoning.repair_safe_v8
   python -m src.reasoning.repair_macro_guarded_v10
   ```
7. Evaluate:
   ```bash
   python -m src.reasoning.evaluate_results            # Hit
   python -m src.reasoning.evaluate_results_corrected  # Macro-F1
   ```

## Configuration

All paths are abstracted through `configs/paths.yaml`. Edit that single file
to point at your local data locations. No CORE script contains hardcoded
Colab paths.

Model hyperparameters (Qwen variant, dtype, AWQ, decoding temperatures, top_p,
adaptive fact limits, etc.) are documented in `configs/reasoning_webqsp.yaml`
and `configs/reasoning_cwq.yaml`. The values mirror those embedded in
`src/reasoning/llm_utils.py` and `main.py`.

## Random seeds

| Stage              | Seed     |
| ------------------ | -------- |
| Retriever training | 42       |
| Bootstrap CIs      | 20260603 |
| Reasoning decoding | n/a (controlled by vLLM, temperature sweep [0.0, 0.15, 0.3]) |
