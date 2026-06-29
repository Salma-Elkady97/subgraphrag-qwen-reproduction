# Legacy directory

This directory preserves source files that are not part of the final pipeline
that produced the published Table 2 numbers, but that are kept for full
transparency about the development process.

## What's here

### `legacy/reasoning/`
- `postprocess_kgqa.py`, `postprocess_kgqa_v2.py` — earlier postprocessor
  iterations; superseded by `src/reasoning/postprocess_kgqa_v3_safe.py`.
- `repair_macro_v9.py` — earlier version of the macro-guarded repair; superseded
  by `src/reasoning/repair_macro_guarded_v10.py`.
- `repair_macro_guarded_v11.py` — uses v10 as input and applies further
  question-specific repairs; was NOT used to produce the published numbers.
- `repair_results.py`, `repair_results_focused.py` — alternative repair paths
  that were explored but discarded.
- `repair_to_92.py` — additional question-specific overrides explored but not
  used in the final reported run.
- `undo_harmful_focus.py` — utility for reverting a focused repair.
- `patch_trim.py` — one-off Colab script that monkey-patches `llm_utils.py`
  in place. Documents a context-trim variant that was tried during development.
- `train_verifier.py`, `run_trained_verifier.py`, `verifier_rerank.py` — a
  verifier / re-ranker experiment that the paper reports as negative results
  (it reduced final accuracy and was therefore not adopted).

### `legacy/retrieval/`
- `inference_split.py` — an alternative inference loop that was not used.
- `emb_cwq_safe.py`, `fix_cwq_processed.py`, `make_cwq_processed_only.py`,
  `rebuild_cwq_processed_only.py`, `rebuild_cwq_triple_scores_only.py` — CWQ
  data-preparation utilities used during initial setup; once
  `cwq_SOTA_READY.pth` exists they are not needed again.

## Why we kept them

The user's mandate during cleanup was to "preserve the original implementation
and report the issue rather than make assumptions" whenever a file's role was
ambiguous. Rather than delete files we were not 100% sure are unused, we
moved them here. None of these files is imported or executed by any CORE
script under `src/`.
