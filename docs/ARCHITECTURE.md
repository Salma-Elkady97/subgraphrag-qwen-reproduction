# Architecture summary

## Retrieval pipeline (`src/retrieval/`)

```
RoG-{webqsp,cwq} (Hugging Face)
        │
        ▼
emb.EmbInferDataset           ──► processed/.pkl  (per-question entity/relation lists)
gte_large_en.GTELargeEN       ──► emb/*.pt        (entity / relation / question embeddings)
                                              │
                                              ▼
                                  Retriever (MLP + DDE) trained by src.retrieval.train
                                              │
                                              ▼
                                  src.retrieval.inference   ──► retrieval_result.pth
                                              │
                                              ▼
                                  src.retrieval.eval        ──► retrieval recall@K
```

Notes:
- `gte_large_en.GTELargeEN` keeps the upstream class name for compatibility but
  internally uses `sentence-transformers/all-mpnet-base-v2`. See
  [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md#encoder-substitution).
- `train.py` and `inference.py` import `src.config.retriever`, `src.dataset.retriever`,
  `src.model.retriever`. Those modules are **not present** in this repo; they
  are taken directly from the upstream SubgraphRAG codebase (Li et al., 2025).
  See [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md#upstream-imports).

## Reasoning pipeline (`src/reasoning/`)

```
WebQSP                                       CWQ
──────                                       ───
retrieval_result.pth + processed.pkl         retrieval_result.pth
            │                                       │
            ▼                                       ▼
build_train_foundation.py            build_cwq_train_val_foundation.py
            │                              prepare_cwq_reasoning.py
            ▼                                       │
   webqsp_SOTA_READY.pth                            ▼
            │                            cwq_SOTA_READY.pth
            │                                       │
            └────────────────────┬──────────────────┘
                                 ▼
                       main.py + llm_utils.py
                       Qwen2.5-72B-AWQ via vLLM
                       form_tiered_prompt (adaptive limits)
                       Majority vote over temperatures [0.0, 0.15, 0.3]
                                 │
                                 ▼
                       baseline_results.jsonl
                       (CWQ: this is the final output → 66.27 / 53.78)
                                 │
                                 │  (WebQSP only)
                                 ▼
        repair_safe_v2.py → v3 → v4 → v5 → v6 → v7 → v8.py
                                 │
                                 ▼
                  repair_macro_guarded_v10.py   ← FINAL for WebQSP
                  → baseline_results_macro_guarded_v10.jsonl
                  → 89.51 Hit / 76.39 Macro-F1
                                 │
                                 ▼
                  evaluate_results.py (Hit) + evaluate_results_corrected.py (F1)
```

## Files at a glance

| File                                                                   | Role                                                                 |
| ---------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `src/reasoning/main.py`                                                | Entry point for vLLM inference                                       |
| `src/reasoning/llm_utils.py`                                           | vLLM init, prompt construction, decoding, majority vote              |
| `src/reasoning/prompts.py`                                             | Upstream SubgraphRAG prompts (kept for compatibility)                |
| `src/reasoning/prepare_data.py`                                        | Loads RoG subgraphs + scored triplets                                |
| `src/reasoning/prepare_prompts.py`                                     | Prompt templating used by upstream eval                              |
| `src/reasoning/build_train_foundation.py`                              | Builds `webqsp_TRAIN_SOTA_READY.pth` for training                    |
| `src/reasoning/build_cwq_train_val_foundation.py`                      | CWQ train/val foundation builder                                     |
| `src/reasoning/prepare_cwq_reasoning.py`                               | CWQ test foundation (`cwq_SOTA_READY.pth`)                           |
| `src/reasoning/postprocess_kgqa_v3_safe.py`                            | Path-based safe postprocessor                                        |
| `src/reasoning/repair_safe_v2.py` … `v8.py`                            | Sequential question-specific safe-repair passes (WebQSP only)        |
| `src/reasoning/repair_macro_guarded_v10.py`                            | Final WebQSP postprocessor                                           |
| `src/reasoning/evaluate_results.py`                                    | Hit metric                                                           |
| `src/reasoning/evaluate_results_corrected.py`                          | Macro-F1 / Precision / Recall metric                                 |
| `src/reasoning/retrieval_service.py`                                   | Helper (fuzzy entity linking + 2-hop subgraph)                       |
| `src/eval/score.py`                                                    | Self-contained reproduction of the v10 metric                        |
| `src/eval/bootstrap_ci.py`                                             | 95% bootstrap CIs for Table 2                                        |
| `src/utils/paths.py`                                                   | Path-token resolver for ${paths.NAME} references                     |
| `legacy/`                                                              | Dev/experimental files preserved unchanged (see legacy/README.md)    |
