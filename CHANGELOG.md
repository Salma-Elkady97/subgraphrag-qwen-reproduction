# Changelog

## 1.0.0 — 2026-06-20

### Repository cleanup (this commit)

All changes below are STRUCTURAL ONLY. No algorithm, prompt, hyperparameter, or
numeric value has been modified. Verification against the original
`results/*_per_q.csv` and `results/*_predictions.json` files reproduces the
published Table 2 numbers (WebQSP 92.13 / 77.86, CWQ 66.24 / 53.77) exactly.

#### Added
- Top-level metadata: `README.md`, `LICENSE` (MIT), `CITATION.cff`,
  `requirements.txt`, `environment.yml`, `.gitignore`.
- Documentation under `docs/`: `ARCHITECTURE.md`, `REPRODUCIBILITY.md`,
  `HARDWARE.md`, `DATA.md`, `KNOWN_ISSUES.md`.
- `configs/paths.yaml` — single file mapping every path token used in CORE
  scripts. Edit this file to point at your local data layout.
- `configs/retrieval_{webqsp,cwq}.yaml` — retriever configs reflecting the
  values used during this work.
- `configs/reasoning_{webqsp,cwq}.yaml` — reasoning configs documenting model,
  prompt, decoding, and postprocessing parameters embedded in `llm_utils.py`
  and `main.py`.
- `src/utils/paths.py` — resolves `${paths.NAME}` tokens via `configs/paths.yaml`.
- `src/eval/score.py` — minimal self-contained reproduction of the v10 metric.
- `src/eval/bootstrap_ci.py` — 95% bootstrap CIs for Table 2.
- `scripts/reproduce_table2.sh` and `scripts/verify_predictions.py` — fast
  verification entry points.
- `legacy/README.md` — explains every file moved to `legacy/`.

#### Changed
- All `/content/drive/...` Colab paths in CORE pipeline files replaced with
  `${paths.NAME}` tokens (resolved through `src/utils/paths.py`). No other
  source modifications. See `substitution_log.json` for the per-file count.

#### Moved (no edits)
The following were moved unchanged from `src/reasoning/` to
`legacy/reasoning/`:
- `postprocess_kgqa.py`, `postprocess_kgqa_v2.py`
- `repair_macro_v9.py`, `repair_macro_guarded_v11.py`
- `repair_results.py`, `repair_results_focused.py`
- `repair_to_92.py`, `undo_harmful_focus.py`, `patch_trim.py`
- `train_verifier.py`, `run_trained_verifier.py`, `verifier_rerank.py`

The following were moved unchanged from `src/retrieval/` to `legacy/retrieval/`:
- `inference_split.py`, `emb_cwq_safe.py`, `fix_cwq_processed.py`,
  `make_cwq_processed_only.py`, `rebuild_cwq_processed_only.py`,
  `rebuild_cwq_triple_scores_only.py`.

#### Removed
- None. Everything from the source ZIP was preserved either under `src/`
  (CORE pipeline) or under `legacy/` (development history).

#### Known issues documented
- WebQSP postprocessing chain reads ground-truth answers and is partly
  question-specific. See `docs/KNOWN_ISSUES.md` §1.
- `GTELargeEN` class uses `sentence-transformers/all-mpnet-base-v2` rather
  than the upstream GTE-Large-EN v1.5. See `docs/KNOWN_ISSUES.md` §2.
- `src/retrieval/train.py` and `inference.py` import canonical SubgraphRAG
  modules that are not committed here; they must be copied from the upstream
  repository. See `docs/KNOWN_ISSUES.md` §3.
