
import os
import pickle
import torch

from src.utils.paths import resolve as _resolve
BASE = _resolve("${paths.cwq_data_root}")
OUT_DIR = _resolve("${paths.cwq_results_dir}")

os.makedirs(OUT_DIR, exist_ok=True)


def entity_name(item, eid):
    text_entities = item["text_entity_list"]
    non_text_entities = item["non_text_entity_list"]

    if eid < len(text_entities):
        return text_entities[eid]

    j = eid - len(text_entities)
    if 0 <= j < len(non_text_entities):
        return non_text_entities[j]

    return str(eid)


def build_split(split):
    processed_path = f"{BASE}/processed/{split}.pkl"
    score_path = f"{BASE}/triple_scores/{split}.pth"
    out_path = f"{OUT_DIR}/cwq_{split.upper()}_READY.pth"

    print("=" * 100)
    print("Split:", split)
    print("Processed:", processed_path)
    print("Scores:", score_path)

    with open(processed_path, "rb") as f:
        processed = pickle.load(f)

    score_data = torch.load(score_path, map_location="cpu")

    foundation = {}

    for item in processed:
        qid = item["id"]

        if qid not in score_data:
            continue

        scores = score_data[qid].get("triple_scores", None)

        h_ids = item["h_id_list"]
        r_ids = item["r_id_list"]
        t_ids = item["t_id_list"]
        rels = item["relation_list"]

        triples_with_scores = []

        for i, (h, r, t) in enumerate(zip(h_ids, r_ids, t_ids)):
            try:
                h_name = entity_name(item, int(h))
                r_name = rels[int(r)]
                t_name = entity_name(item, int(t))
                score = float(scores[i]) if scores is not None and i < len(scores) else 0.0
                triples_with_scores.append((score, h_name, r_name, t_name))
            except:
                pass

        triples_with_scores = sorted(triples_with_scores, key=lambda x: x[0], reverse=True)
        scored_triplets = [(h, r, t) for score, h, r, t in triples_with_scores]

        foundation[qid] = {
            "id": qid,
            "question": item.get("question", ""),
            "scored_triplets": scored_triplets,
            "a_entity": item.get("a_entity", []),
            "q_entity": item.get("q_entity", []),
        }

    torch.save(foundation, out_path)

    print("Saved:", out_path)
    print("Items:", len(foundation))

    k = list(foundation.keys())[0]
    print("Sample:", k)
    print("Q:", foundation[k]["question"])
    print("Gold:", foundation[k]["a_entity"])
    print("Triples:", len(foundation[k]["scored_triplets"]))
    print("Top triples:", foundation[k]["scored_triplets"][:3])


for split in ["train", "val"]:
    build_split(split)