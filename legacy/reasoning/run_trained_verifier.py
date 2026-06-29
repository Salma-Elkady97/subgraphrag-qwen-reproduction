
import os, re, json, argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


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


def is_multi_question(q):
    q = norm(q)
    keys = [
        "who are", "what are", "which are", "countries",
        "representatives", "senators", "children",
        "parents", "works of art", "members"
    ]
    return any(k in q for k in keys)


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

            if not o:
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


def build_text(question, cand):
    ev = "\n".join(cand["evidence"])
    return f"""Question: {question}
Candidate answer: {cand['candidate']}
Evidence:
{ev}
Is this candidate likely the correct answer?"""


def choose_answer(question, cands, probs):
    rows = []

    for c, p in zip(cands, probs):
        rows.append({
            "candidate": c["candidate"],
            "score": float(p),
            "rank": c["rank"]
        })

    rows = sorted(rows, key=lambda x: x["score"], reverse=True)

    if not rows:
        return ""

    if not is_multi_question(question):
        return rows[0]["candidate"]

    top = rows[0]["score"]
    out = []

    for r in rows:
        if len(out) >= 8:
            break
        if r["score"] >= top - 0.25:
            out.append(r["candidate"])

    if not out:
        out = [rows[0]["candidate"]]

    return " | ".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_pth", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top_k", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=384)
    args = parser.parse_args()

    foundation = torch.load(args.test_pth)
    keys = list(foundation.keys())

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    results = []

    for idx, key in enumerate(tqdm(keys)):
        item = foundation[key]

        question = item.get("question", item.get("q_text", ""))
        triples = item.get("scored_triplets", item.get("scored_triples", []))
        gold = item.get("a_entity", [])

        cands = generate_candidates(question, triples, args.top_k)

        if not cands:
            pred = ""
        else:
            texts = [build_text(question, c) for c in cands]

            probs = []

            for i in range(0, len(texts), args.batch_size):
                batch = texts[i:i+args.batch_size]

                enc = tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    max_length=args.max_len,
                    return_tensors="pt"
                ).to(device)

                with torch.no_grad():
                    logits = model(**enc).logits
                    p = torch.softmax(logits, dim=-1)[:,1]
                    probs.extend(p.cpu().tolist())

            pred = choose_answer(question, cands, probs)

        results.append({
            "id": item.get("id", idx),
            "question": question,
            "prediction": pred,
            "ground_truth": gold
        })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("Saved:", args.output)


if __name__ == "__main__":
    main()
