# Efficient and Reproducible Graph-RAG with Open-Source LLMs for Multi-Hop KGQA

Reproduction and extension of the **SubgraphRAG** framework (Li et al., ICLR 2025)
on the WebQSP and ComplexWebQuestions (CWQ) benchmarks, using the fully
open-source `Qwen/Qwen2.5-72B-Instruct-AWQ` model served by vLLM on a single
NVIDIA A100-SXM4-80GB.

## Headline results

| Method                                | LLM Access      | WebQSP Hit | WebQSP F1 | CWQ Hit | CWQ F1 |
| ------------------------------------- | --------------- | ---------- | --------- | ------- | ------ |
| SubgraphRAG + Llama3.1-70B            | Open-weight     | 86.24      | 74.70     | 57.89   | 51.78  |
| SubgraphRAG + GPT-4o-mini (500)       | Proprietary API | 91.22      | 77.67     | 64.97   | 55.41  |
| SubgraphRAG + GPT-4o                  | Proprietary API | 90.91      | 78.24     | 67.49   | 59.42  |
| **This framework**                    | **Open-weight** | **89.51 [87.98, 90.97]** | **76.39 [74.63, 78.10]** | **66.27 [64.71, 67.83]** | **53.78 [52.34, 55.20]** |

95% confidence intervals are 10,000-replicate non-parametric bootstrap (seed = 20260603).

## Quick start

```bash
git clone https://github.com/Salma-Elkady97/subgraphrag-qwen-reproduction.git
cd subgraphrag-qwen-reproduction
conda env create -f environment.yml && conda activate subgraphrag-qwen

# Reproduce the published Table 2 numbers from the committed per-question logs
python -m src.eval.score results/webqsp_predictions.json results/cwq_predictions.json
python -m src.eval.bootstrap_ci results/webqsp_per_q.csv 1639 89.51 76.39
python -m src.eval.bootstrap_ci results/cwq_per_q.csv    3531 66.27 53.78
```

The two `*_per_q.csv` files and the two `*_predictions.json` files in `results/`
back every number in Table 2 of the paper and reproduce them exactly without
running any model inference.

## Full pipeline (end-to-end re-run)

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md). High-level order:

1. **Retrieval setup**: build embeddings & processed graphs from RoG-WebQSP /
   RoG-CWQ (Hugging Face). See `src/retrieval/`.
2. **Retriever training**: `python -m src.retrieval.train -d webqsp`
3. **Retriever inference**: `python -m src.retrieval.inference -p <cpt.pth>` →
   `retrieval_result.pth`
4. **Foundation data**: `python -m src.reasoning.build_train_foundation`,
   `prepare_cwq_reasoning`, `build_cwq_train_val_foundation`
5. **Reasoning inference**: `python -m src.reasoning.main -d webqsp -m Qwen/Qwen2.5-72B-Instruct-AWQ -p <foundation.pth>` produces `baseline_results.jsonl`. **This is the file whose aggregate is the blind-test result reported in Table 2 of the paper (WebQSP 89.51 / 76.39, CWQ 66.27 / 53.78). No postprocessing is applied for the published numbers.**
6. **Evaluation**: `python -m src.reasoning.evaluate_results` (Hit) and
   `evaluate_results_corrected` (F1) from the SubgraphRAG eval suite.

CWQ uses **no postprocessing**. WebQSP also uses no postprocessing for the
blind-test numbers reported in the paper. A separate chain of
`repair_safe_v2.py → … → repair_safe_v8.py → repair_macro_guarded_v10.py`
exists in `src/reasoning/` and was used during development to produce an
oracle-style result of 92.13 / 77.86 on WebQSP, but those scripts read the
test-set gold answers at runtime and their outputs are therefore NOT reported
as blind-evaluation results. See `docs/KNOWN_ISSUES.md` for a full discussion
of this distinction.

## Hardware

- NVIDIA A100-SXM4-80GB (40 GB also works at smaller context)
- CUDA 13.0, PyTorch 2.11.0+cu128, vLLM 0.11.2
- Peak inference memory: **38.76 GiB**
- WebQSP runtime: **2.53 GPU-hours**, throughput 10.79 q/min
- CWQ runtime: **4.46 GPU-hours**, throughput 13.19 q/min

Full details in [`docs/HARDWARE.md`](docs/HARDWARE.md).

## Datasets

WebQSP and CWQ are NOT redistributed in this repository. See
[`docs/DATA.md`](docs/DATA.md) for the official download sources.

## Known issues and limitations

A reproducibility-critical concern about the WebQSP postprocessing chain is
documented in [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md). Please read it
before drawing conclusions from comparisons against the WebQSP F1 number.

## Architecture summary

[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) describes the retrieval and
reasoning pipelines, key file responsibilities, and the data-flow diagram.

## Citing

If you use this repository, please cite both the original SubgraphRAG paper and
this reproduction. See [`CITATION.cff`](CITATION.cff).

## License

[MIT](LICENSE).
