
import torch

from src.utils.paths import resolve as _resolve
SRC = _resolve("${paths.cwq_results_dir}/cwq_retrieval_result.pth")
OUT = _resolve("${paths.cwq_results_dir}/cwq_SOTA_READY.pth")

data = torch.load(SRC, map_location="cpu")

out = {}

for qid, item in data.items():
    triples = item.get("scored_triplets", None)
    if triples is None:
        triples = item.get("scored_triples", [])

    out[qid] = {
        "id": qid,
        "question": item.get("question", item.get("q_text", "")),
        "scored_triplets": triples,
        "a_entity": item.get("a_entity", []),
        "q_entity": item.get("q_entity", []),
    }

torch.save(out, OUT)

print("Saved:", OUT)
print("Items:", len(out))

k = list(out.keys())[0]
print("Sample key:", k)
print("Question:", out[k]["question"])
print("Gold:", out[k]["a_entity"])
print("Num triples:", len(out[k]["scored_triplets"]))
print("First triples:", out[k]["scored_triplets"][:3])