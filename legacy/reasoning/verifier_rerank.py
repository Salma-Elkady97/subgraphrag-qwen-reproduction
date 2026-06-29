
import os
import re
import json
import torch
import argparse
from sentence_transformers import CrossEncoder


def norm(x):
    x = str(x).lower().strip()
    x = re.sub(r"[^a-z0-9\s\.\-:/|,]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def clean_rel(r):
    return str(r).split(".")[-1].replace("_", " ").strip()


def build_name_map(triples):
    mp = {}
    for t in triples:
        try:
            if len(t) >= 3 and "type.object.name" in str(t[1]):
                mp[str(t[0]).strip()] = str(t[2]).strip()
        except:
            pass
    return mp


def is_bad_candidate(x):
    x = norm(x)
    return (not x) or x.startswith("m.") or x in ["none", "unknown", "null"]


def is_multi_question(q):
    q = norm(q)
    return any(x in q for x in [
        "who are", "what are", "which are", "countries",
        "representatives", "senators", "children", "parents",
        "works of art", "members", "religions"
    ])


def generate_candidates(question, triples, top_k=250):
    name_map = build_name_map(triples)
    cand = {}

    for rank, t in enumerate(triples[:top_k]):
        try:
            if len(t) < 3:
                continue

            s0, r0, o0 = str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()

            if "type.object.name" in r0:
                continue

            s = name_map.get(s0, s0)
            r = clean_rel(r0)
            o = name_map.get(o0, o0)

            if is_bad_candidate(o):
                continue

            k = norm(o)

            if k not in cand:
                cand[k] = {
                    "candidate": o,
                    "evidence": [],
                    "rank": rank
                }

            if len(cand[k]["evidence"]) < 6:
                cand[k]["evidence"].append(f"{s} -- {r} --> {o}")

        except:
            pass

    return list(cand.values())[:80]


def build_pair(question, c):
    evidence = "\n".join(c["evidence"])
    text = (
        f"Question: {question}\n"
        f"Candidate answer: {c['candidate']}\n"
        f"Evidence:\n{evidence}\n"
        f"Is this candidate the correct answer?"
    )
    return [question, text]


def select_answers(question, ranked):
    if not ranked:
        return ""

    if not is_multi_question(question):
        return ranked[0]["candidate"]

    top = ranked[0]["score"]
    selected = []

    for r in ranked:
        if len(selected) >= 8:
            break
        if r["score"] >= top - 0.35:
            selected.append(r["candidate"])

    if not selected:
        selected = [ranked[0]["candidate"]]

    return " | ".join(selected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pth", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reranker", default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    parser.add_argument("--top_k_triples", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    print("Loading:", args.input_pth)
    foundation = torch.load(args.input_pth)
    keys = list(foundation.keys())

    print("Loading CrossEncoder:", args.reranker)
    model = CrossEncoder(args.reranker, max_length=512)

    rows = []

    for idx, key in enumerate(keys):
        item = foundation[key]
        question = item.get("question", item.get("q_text", ""))
        triples = item.get("scored_triplets", [])
        gold = item.get("a_entity", [])

        candidates = generate_candidates(question, triples, args.top_k_triples)

        if not candidates:
            pred = ""
        else:
            pairs = [build_pair(question, c) for c in candidates]
            scores = model.predict(pairs, batch_size=args.batch_size, show_progress_bar=False)

            ranked = []
            for c, s in zip(candidates, scores):
                ranked.append({
                    "candidate": c["candidate"],
                    "score": float(s) - 0.002 * c["rank"]
                })

            ranked = sorted(ranked, key=lambda x: x["score"], reverse=True)
            pred = select_answers(question, ranked)

        rows.append({
            "id": item.get("id", idx),
            "question": question,
            "prediction": pred,
            "ground_truth": gold
        })

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx+1}/{len(keys)}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("Saved:", args.output)


if __name__ == "__main__":
    main()
