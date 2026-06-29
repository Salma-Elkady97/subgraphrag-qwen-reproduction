
import torch, pickle

from src.utils.paths import resolve as _resolve
PROCESSED = _resolve("${paths.webqsp_processed}/train.pkl")
TRIPLE_SCORES = _resolve("${paths.webqsp_triple_scores}/train.pth")
OUT = _resolve("${paths.webqsp_train_foundation}")

with open(PROCESSED, "rb") as f:
    processed = pickle.load(f)

score_data = torch.load(TRIPLE_SCORES, map_location="cpu")


def entity_name(item, eid):
    text_entities = item["text_entity_list"]
    non_text_entities = item["non_text_entity_list"]

    if eid < len(text_entities):
        return text_entities[eid]

    j = eid - len(text_entities)
    if 0 <= j < len(non_text_entities):
        return non_text_entities[j]

    return str(eid)


foundation = {}

for item in processed:
    qid = item["id"]

    h_ids = item["h_id_list"]
    r_ids = item["r_id_list"]
    t_ids = item["t_id_list"]
    rels = item["relation_list"]

    scores = None
    if qid in score_data:
        scores = score_data[qid].get("triple_scores", None)

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

    # Sort descending by retrieval score
    triples_with_scores = sorted(triples_with_scores, key=lambda x: x[0], reverse=True)

    scored_triplets = [(h, r, t) for score, h, r, t in triples_with_scores]

    foundation[qid] = {
        "id": qid,
        "question": item.get("question", ""),
        "scored_triplets": scored_triplets,
        "a_entity": item.get("a_entity", []),
        "q_entity": item.get("q_entity", []),
    }

torch.save(foundation, OUT)

print("Saved:", OUT)
print("Items:", len(foundation))

first_key = list(foundation.keys())[0]
first = foundation[first_key]

print("First key:", first_key)
print("Question:", first["question"])
print("Gold:", first["a_entity"])
print("Num triples:", len(first["scored_triplets"]))
print("Top 10 triples:")
for t in first["scored_triplets"][:10]:
    print(t)