# Datasets

This repository does **not** redistribute the underlying benchmark datasets.
They must be obtained from the official upstream sources.

## WebQSP
- Original: Yih et al., 2016, "The Value of Semantic Parse Labeling for KBQA".
- Processed variant used here: `rmanluo/RoG-webqsp` on Hugging Face
  (https://huggingface.co/datasets/rmanluo/RoG-webqsp), loaded by
  `src/retrieval/emb.py` and `src/reasoning/prepare_data.py`.
- Test split: 1,639 questions.

## ComplexWebQuestions (CWQ)
- Original: Talmor & Berant, 2018, "The Web as a Knowledge-Base for
  Answering Complex Questions".
- Processed variant used here: `rmanluo/RoG-cwq` on Hugging Face
  (https://huggingface.co/datasets/rmanluo/RoG-cwq).
- Test split: 3,531 questions.

## Freebase
The underlying Freebase entity identifiers (`m.xxxxx`, `g.xxxxx`) appear in the
processed subgraphs. The `entity_identifiers` whitelist used by
`src/retrieval/emb.py` should be provided locally (path documented in the
`emb` config of the upstream SubgraphRAG repository).

## Model weights
The Qwen2.5-72B-Instruct-AWQ checkpoint is downloaded automatically by
vLLM from Hugging Face on first run. It is approximately 38 GB after
AWQ-Marlin 4-bit quantization.
