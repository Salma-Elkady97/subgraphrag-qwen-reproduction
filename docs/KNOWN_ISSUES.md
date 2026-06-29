# Known issues, deviations from the upstream repository, and reproducibility caveats

This document records every meaningful deviation between the code committed in
this repository and the canonical SubgraphRAG implementation, plus methodological
caveats a reviewer should be aware of. **None of these issues was fixed during
the cleanup** — files were preserved exactly as they were used to produce the
published numbers, per the user's instruction.

## 1. WebQSP postprocessing chain is gold-answer-aware

The WebQSP published numbers (89.51 Hit / 76.39 Macro-F1) are produced by
running `main.py` and then passing the output through a chain of postprocessors:

```
baseline_results.jsonl
    → repair_safe_v2.py
    → repair_safe_v3.py
    → repair_safe_v4.py
    → repair_safe_v5.py
    → repair_safe_v6.py
    → repair_safe_v7.py
    → repair_safe_v8.py
    → repair_macro_guarded_v10.py
    → baseline_results_macro_guarded_v10.jsonl
```

Two characteristics of this chain warrant disclosure:

a) **`repair_macro_guarded_v10.py` reads the ground-truth answer set for each
   question and only accepts a proposed alternative prediction if it does not
   reduce Hit or Macro-F1 relative to the original prediction.** This is
   implemented in the `metric()` function and the guard
   `if h >= best_hit and f1 >= best_f1` on the test set. By construction this
   guarantees that the chain never lowers either metric on the reported test
   split; the chain can therefore only equal or exceed the raw model output.

b) **`repair_safe_v2.py` through `repair_safe_v8.py` and the legacy
   `repair_to_92.py` contain question-specific string overrides** (for example
   "anakin skywalker" → "Ted Bracewell", "australian dollar called" → "AUD",
   "william taft famous for" → "lawyer | judge | jurist"). These overrides are
   only triggered when the substring of a specific test question is present.

These two properties mean that the WebQSP F1 and Hit numbers as reported are
sensitive to the test-set distribution and should not be interpreted as
estimates of generalisation performance from a held-out evaluation that is
fully blind to test-set answers. The CWQ pipeline does **not** include this
postprocessing chain — CWQ numbers (66.27 / 53.78) come directly from
`main.py` and are not subject to the same caveat.

## 2. Encoder substitution

`src/retrieval/gte_large_en.py` retains the class name `GTELargeEN` and is
imported as the "GTE-Large-EN" encoder, but the underlying transformer is
`sentence-transformers/all-mpnet-base-v2` (768-dim mean-pooled). The original
SubgraphRAG codebase uses the actual `Alibaba-NLP/gte-large-en-v1.5` encoder
(1024-dim, requires `trust_remote_code=True` and `xformers`). The substitution
was made because the original encoder requires environment configuration that
was not available in the Colab runtime used for this work. The retriever was
trained and evaluated end-to-end with this substituted encoder, so the
retrieval upper bound reported in the paper (WebQSP 93.65 %, CWQ 79.38 %) is
specific to the all-mpnet-base-v2 embeddings, not the GTE-Large-EN ones.

## 3. Upstream module imports

`src/retrieval/train.py` and `src/retrieval/inference.py` import from:
- `src.config.retriever`
- `src.dataset.retriever`
- `src.model.retriever`

None of these modules are present in this repository. They are the canonical
SubgraphRAG modules and were taken directly from the upstream repository at
https://github.com/<upstream>/<repo>. To reproduce the retrieval stage,
clone the upstream repository alongside this one and copy or symlink those
three modules into `src/`.

## 4. Missing intermediate `.pth` files

The pipeline depends on several `.pth` artefacts that are too large to commit:
- `webqsp_SOTA_READY.pth` (test foundation)
- `webqsp_TRAIN_SOTA_READY.pth` (training foundation)
- `cwq_SOTA_READY.pth` (CWQ test foundation)
- `retrieval_result.pth` (per dataset)
- `cpt.pth` (retriever checkpoint)

These must be regenerated locally by running stages 1–4 of
[`docs/REPRODUCIBILITY.md`](REPRODUCIBILITY.md). The fast-verification path
documented in REPRODUCIBILITY.md uses the committed `*_per_q.csv` and
`*_predictions.json` files and does not need any of these `.pth` artefacts.

## 5. CWQ training does not appear to be exercised end-to-end

`build_cwq_train_val_foundation.py` builds CWQ train/val foundation files, but
no `train.py` invocation for CWQ retriever training is preserved in the repo
history. The CWQ retrieval results used for evaluation were produced with a
retriever trained as documented in the upstream SubgraphRAG paper; we have
not independently verified that the train/val foundation files committed here
exactly reproduce the upstream retriever state.

## 6. `legacy/` directory

`legacy/` preserves files that are not part of the final pipeline:
- intermediate postprocessing versions (`postprocess_kgqa.py`,
  `postprocess_kgqa_v2.py`, `repair_macro_v9.py`,
  `repair_macro_guarded_v11.py`),
- alternative repair scripts (`repair_results.py`,
  `repair_results_focused.py`, `repair_to_92.py`, `undo_harmful_focus.py`),
- a verifier / reranker attempt that the paper reports as negative results
  (`train_verifier.py`, `run_trained_verifier.py`, `verifier_rerank.py`),
- a development monkey-patch (`patch_trim.py`),
- retrieval-side development utilities (`fix_cwq_processed.py`,
  `make_cwq_processed_only.py`, `rebuild_cwq_processed_only.py`,
  `rebuild_cwq_triple_scores_only.py`, `emb_cwq_safe.py`,
  `inference_split.py`).

These files are kept for transparency and to allow others to inspect the
development history. They are not invoked by any CORE script.
