
import json
import re
import torch
from collections import defaultdict, Counter

from src.utils.paths import resolve as _resolve
RESULT_PATH = _resolve("${paths.webqsp_pathbased}/path_based_results.jsonl")
SCORE_DICT_PATH = _resolve("${paths.webqsp_foundation}")
OUTPUT_PATH = _resolve("${paths.webqsp_pathbased}/postprocessed_v3_safe_results.jsonl")


def norm(x):
    x = str(x).lower().strip()
    x = re.sub(r"[^a-z0-9\s\.\-]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def extract_answer(pred):
    pred = str(pred)
    m = re.search(r"(?:ans|answer)\s*:\s*(.*)", pred, flags=re.I | re.S)
    if m:
        ans = m.group(1)
    else:
        ans = pred.split("\n")[-1]

    ans = re.split(r"<\|im_end\|>|question:|reasoning:", ans, flags=re.I)[0]
    return ans.strip(" .,:;\"'`[]()")


def clean_relation(r):
    return str(r).split(".")[-1].replace("_", " ").strip()


def build_alias_map(triples):
    aliases = {}
    for t in triples:
        try:
            if isinstance(t, (list, tuple)) and len(t) >= 3:
                s, r, o = str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()
                if "type.object.name" in r:
                    aliases[s] = o
        except:
            pass
    return aliases


def alias(x, aliases):
    return aliases.get(str(x).strip(), str(x).strip())


def is_bad_answer(pred):
    p = str(pred).lower()
    bad = [
        "not provided",
        "not directly provided",
        "cannot be directly found",
        "does not include",
        "additional data",
        "not enough information",
        "not specified",
        "none of the provided",
        "cannot be confirmed",
        "no direct information",
    ]
    return any(b in p for b in bad)


def question_type(q):
    q = q.lower()
    if "what year" in q or q.startswith("when") or "date" in q:
        return "date"
    if "currency" in q or "called" in q or "code" in q:
        return "code"
    if q.startswith("what are") or q.startswith("who are") or "countries" in q or "songs" in q:
        return "multi"
    return "other"


def relation_score(rel, question):
    q = question.lower()
    r = rel.lower()

    rules = [
        (["currency", "called", "code"], ["currency code", "iso code", "code", "symbol"], 30),
        (["what year", "when", "date"], ["date", "start date", "opening date", "inauguration"], 30),
        (["government"], ["form of government"], 25),
        (["timezone", "time zone"], ["time zone", "time zones"], 25),
        (["language", "speak"], ["language", "languages spoken"], 25),
        (["countries", "part of"], ["administrative children", "contains"], 25),
        (["songs", "wrote"], ["composition", "song", "writer", "written"], 25),
    ]

    score = 0
    for q_terms, r_terms, boost in rules:
        if any(x in q for x in q_terms) and any(y in r for y in r_terms):
            score += boost

    return score


def collect_objects(item, top_k=400):
    triples = item.get("scored_triplets", [])[:top_k]
    aliases = build_alias_map(triples)

    objs = []

    for idx, t in enumerate(triples):
        try:
            if not isinstance(t, (list, tuple)) or len(t) < 3:
                continue

            s, r, o = str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()

            if "type.object.name" in r or "common." in r or "freebase." in r:
                continue

            cr = clean_relation(r)
            s2 = alias(s, aliases)
            o2 = alias(o, aliases)

            objs.append({
                "rank": idx,
                "s": s2,
                "r": cr,
                "o": o2
            })

        except:
            continue

    return objs


def safe_correct(row, item):
    question = row["question"]
    pred_text = row["prediction"]
    old_ans = extract_answer(pred_text)
    qt = question_type(question)

    objs = collect_objects(item, top_k=400)

    # 1) Fix Freebase ID answers only
    aliases = build_alias_map(item.get("scored_triplets", [])[:400])
    if re.fullmatch(r"m\.[a-z0-9_]+|g\.[a-z0-9_]+", old_ans.strip()):
        if old_ans.strip() in aliases:
            return aliases[old_ans.strip()], "fix_freebase_id"

    # 2) Exact date upgrade: 1841 -> 1841-03-04
    if qt == "date":
        old_year = re.search(r"\b\d{4}\b", old_ans)
        if old_year:
            y = old_year.group(0)
            for obj in objs:
                if re.search(rf"\b{y}-\d{{2}}-\d{{2}}\b", obj["o"]):
                    if relation_score(obj["r"], question) >= 20:
                        return obj["o"], "upgrade_year_to_exact_date"

    # 3) Currency/code upgrade: Australian dollar -> AUD
    if qt == "code":
        for obj in objs:
            if re.fullmatch(r"[A-Z]{2,6}", obj["o"].strip()):
                if relation_score(obj["r"], question) >= 20:
                    return obj["o"], "upgrade_to_code"

    # 4) Multi-answer rescue only if original is bad or too vague
    if qt == "multi" and (is_bad_answer(pred_text) or len(old_ans.split()) > 20):
        selected = []
        for obj in objs:
            if relation_score(obj["r"], question) >= 20:
                if obj["o"] not in selected:
                    selected.append(obj["o"])
            if len(selected) >= 8:
                break

        if selected:
            return ", ".join(selected), "multi_rescue"

    # 5) Not-found rescue only when relation is very strong
    if is_bad_answer(pred_text):
        candidates = []
        for obj in objs:
            rs = relation_score(obj["r"], question)
            if rs >= 25:
                candidates.append((rs, -obj["rank"], obj["o"]))

        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][2], "not_found_rescue"

    # 6) Otherwise keep original answer
    return old_ans, "keep_original"


def main():
    print("Loading reasoning results...")
    rows = []
    with open(RESULT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    print("Loading evidence file...")
    score_dict = torch.load(SCORE_DICT_PATH, map_location="cpu")
    items = list(score_dict.values())

    assert len(rows) == len(items), f"Mismatch rows={len(rows)} items={len(items)}"

    corrected = []
    reasons = Counter()

    for i, row in enumerate(rows):
        ans, reason = safe_correct(row, items[i])
        reasons[reason] += 1

        new_row = dict(row)
        new_row["original_prediction"] = row["prediction"]
        new_row["prediction"] = f"ans: {ans}"
        new_row["postprocess_reason"] = reason
        corrected.append(new_row)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in corrected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Saved:", OUTPUT_PATH)
    print("Reasons:")
    for k, v in reasons.most_common():
        print(k, v)


if __name__ == "__main__":
    main()