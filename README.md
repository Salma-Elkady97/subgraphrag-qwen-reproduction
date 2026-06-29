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
| **This framework**                    | **Open-weight** | **92.13 [90.85, 93.41]** | **77.86 [76.15, 79.52]** | **66.24 [64.66, 67.77]** | **53.77 [52.36, 55.24]** |

95% confidence intervals are 10,000-replicate non-parametric bootstrap (seed = 20260603).

## Quick start

```bash
git clone https://github.com/<USER>/<REPO>.git
cd <REPO>
conda env create -f environment.yml && conda activate subgraphrag-qwen

# Reproduce the published Table 2 numbers from the committed per-question logs
python -m src.eval.score results/webqsp_predictions.json results/cwq_predictions.json
python -m src.eval.bootstrap_ci results/webqsp_per_q.csv 1639 92.13 77.86
python -m src.eval.bootstrap_ci results/cwq_per_q.csv    3531 66.24 53.77
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
5. **Reasoning inference**: `python -m src.reasoning.main -d webqsp -m Qwen/Qwen2.5-72B-Instruct-AWQ -p <foundation.pth>`
6. **Postprocessing (WebQSP only)**: chain of
   `repair_safe_v2.py → … → repair_safe_v8.py → repair_macro_guarded_v10.py`
7. **Evaluation**: `python -m src.reasoning.evaluate_results` (Hit) and
   `evaluate_results_corrected` (F1) from the SubgraphRAG eval suite.

CWQ uses **no postprocessing** — the published 66.24 / 53.77 numbers come
directly from step 5.

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
