
import json
import re
import torch
from collections import defaultdict


# =========================
# CONFIG
# =========================
RESULT_PATH = "/content/drive/MyDrive/SubgraphRAG/reason/results/KGQA/webqsp/PathBased/path_based_results.jsonl"
SCORE_DICT_PATH = "/content/drive/MyDrive/SubgraphRAG/webqsp_SOTA_READY.pth"

OUTPUT_PATH = "/content/drive/MyDrive/SubgraphRAG/reason/results/KGQA/webqsp/PathBased/postprocessed_results.jsonl"


# =========================
# NORMALIZATION
# =========================
def norm(x):
    x = str(x).lower().strip()
    x = re.sub(r"[^a-z0-9\s\.\-]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def clean_relation(r):
    return str(r).split(".")[-1].replace("_", " ").strip()


def words(x):
    return set(re.findall(r"[a-zA-Z0-9]+", str(x).lower()))


def extract_answer(pred):
    pred = str(pred)

    m = re.search(r"(?:ans|answer)\s*:\s*(.*)", pred, flags=re.I | re.S)
    if m:
        ans = m.group(1)
    else:
        ans = pred.split("\n")[-1]

    ans = re.split(r"<\|im_end\|>|question:|reasoning:", ans, flags=re.I)[0]
    ans = ans.strip(" .,:;\"'`[]()")
    return ans


# =========================
# ALIAS MAP
# =========================
def build_alias_map(triples):
    aliases = {}

    for t in triples:
        try:
            if not isinstance(t, (list, tuple)) or len(t) < 3:
                continue

            s, r, o = str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()

            if "type.object.name" in r:
                aliases[s] = o
        except Exception:
            pass

    return aliases


def alias(x, aliases):
    return aliases.get(str(x).strip(), str(x).strip())


# =========================
# QUESTION TYPE
# =========================
def qtype(question):
    q = question.lower()

    if q.startswith("who") or " who " in q:
        return "person"
    if q.startswith("where"):
        return "location"
    if q.startswith("when") or "what year" in q or "date" in q:
        return "date"
    if "how many" in q or "number of" in q:
        return "number"
    if "called" in q or "currency" in q or "code" in q:
        return "code_or_name"
    if "songs" in q or "countries" in q or q.startswith("what are") or q.startswith("who are"):
        return "multi"
    return "entity"


# =========================
# RELATION SCORING
# =========================
def relation_score(relation, question):
    q = question.lower()
    r = relation.lower()
    score = len(words(q) & words(r))

    rules = [
        (["wife", "husband", "spouse"], ["spouse", "wife", "husband", "partner"], 20),
        (["governor"], ["governor", "office holder", "position held", "government position"], 20),
        (["run against", "opponent", "second term"], ["opponent", "candidate", "election", "campaign"], 20),
        (["plays", "played by", "who plays"], ["actor", "portrayed", "played by", "cast"], 20),
        (["role", "played in", "who did"], ["character", "role", "performance"], 20),
        (["currency", "called", "code"], ["currency code", "iso code", "code", "symbol", "currency"], 18),
        (["take office", "president"], ["start date", "inauguration", "office", "date"], 16),
        (["opened", "open"], ["opening date", "first performance", "start date", "date"], 16),
        (["songs", "wrote", "written"], ["composition", "song", "writer", "author", "written"], 15),
        (["countries", "part of"], ["contains", "containedby", "country", "constituent"], 15),
        (["team", "play for", "played for"], ["team", "sports team", "played for"], 15),
        (["language", "speak"], ["language", "languages spoken"], 15),
        (["born", "birth"], ["birth", "born"], 12),
        (["dictator", "leader"], ["leader", "head of government", "office holder"], 12),
    ]

    for q_terms, r_terms, boost in rules:
        if any(x in q for x in q_terms) and any(y in r for y in r_terms):
            score += boost

    return score


def type_score(candidate, question):
    qt = qtype(question)
    c = str(candidate)

    if qt == "date":
        if re.search(r"\d{4}-\d{2}-\d{2}", c):
            return 15
        if re.search(r"\b\d{4}\b", c):
            return 8

    if qt == "number":
        if re.search(r"\b\d+(\.\d+)?\b", c):
            return 12

    if qt == "code_or_name":
        if re.fullmatch(r"[A-Z]{2,5}", c.strip()):
            return 18
        if re.fullmatch(r"[A-Z0-9]{2,6}", c.strip()):
            return 15

    if qt == "person":
        if re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+", c):
            return 8

    return 0


# =========================
# CANDIDATE GENERATION
# =========================
def collect_candidates(item, top_k=400):
    triples = item.get("scored_triplets", [])[:top_k]
    question = item.get("question", item.get("q_text", ""))

    aliases = build_alias_map(triples)

    scores = defaultdict(float)
    evidence = defaultdict(list)

    for idx, t in enumerate(triples):
        try:
            if not isinstance(t, (list, tuple)) or len(t) < 3:
                continue

            s, r, o = str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()

            if not s or not r or not o:
                continue

            cr = clean_relation(r)

            # skip metadata except aliases already extracted
            if "type.object.name" in r or "common." in r or "freebase." in r:
                continue

            s2 = alias(s, aliases)
            o2 = alias(o, aliases)

            rank_bonus = max(0, top_k - idx) / top_k
            rel_bonus = relation_score(cr, question)
            overlap = len(words(question) & words(f"{s2} {cr} {o2}"))
            ts = type_score(o2, question)

            score = rank_bonus + (4 * rel_bonus) + overlap + ts

            # object is usually answer
            scores[o2] += score
            evidence[o2].append(f"{s2} -> {cr} -> {o2}")

            # subject can be answer only weakly
            if rel_bonus >= 18:
                scores[s2] += score * 0.2
                evidence[s2].append(f"{s2} -> {cr} -> {o2}")

        except Exception:
            continue

    ranked = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return ranked, evidence, scores


# =========================
# ANSWER CORRECTION
# =========================
def correct_answer(row, item):
    question = row["question"]
    pred_raw = row["prediction"]
    pred_ans = extract_answer(pred_raw)

    candidates, evidence, scores = collect_candidates(item, top_k=400)

    if not candidates:
        return pred_ans, "no_candidates"

    n_pred = norm(pred_ans)

    # If prediction exactly matches a strong candidate, keep it
    for c in candidates[:80]:
        if norm(c) == n_pred:
            return c, "kept_exact_candidate"

    # If prediction contains candidate, prefer exact candidate form
    for c in candidates[:80]:
        nc = norm(c)
        if nc and nc in norm(pred_raw):
            return c, "normalized_from_prediction"

    qt = qtype(question)

    # Multi-answer questions: return top relevant candidates
    if qt == "multi":
        chosen = candidates[:8]
        return ", ".join(chosen), "multi_candidate_list"

    # Date questions: prefer exact date over year
    if qt == "date":
        for c in candidates[:100]:
            if re.search(r"\d{4}-\d{2}-\d{2}", str(c)):
                return c, "date_exact"
        for c in candidates[:100]:
            if re.search(r"\b\d{4}\b", str(c)):
                return c, "date_year"

    # Code/currency questions: prefer short uppercase code
    if qt == "code_or_name":
        for c in candidates[:100]:
            if re.fullmatch(r"[A-Z]{2,5}", str(c).strip()):
                return c, "code_candidate"

    # If LLM says not found / cannot answer, force best evidence candidate
    bad_phrases = [
        "not provided",
        "does not include",
        "cannot be directly found",
        "additional data",
        "not directly provided",
        "none of the provided",
    ]
    if any(p in pred_raw.lower() for p in bad_phrases):
        return candidates[0], "rescue_from_not_found"

    # Default: if prediction is long/explanatory/hallucinated, use top candidate
    if len(pred_ans.split()) > 8:
        return candidates[0], "long_answer_replaced"

    return candidates[0], "top_candidate"


# =========================
# MAIN
# =========================
def main():
    print("Loading reasoning results...")
    rows = []
    with open(RESULT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    print("Loading evidence file...")
    score_dict = torch.load(SCORE_DICT_PATH, map_location="cpu")
    items = list(score_dict.values())

    assert len(rows) == len(items), f"Mismatch: rows={len(rows)} items={len(items)}"

    corrected = []
    reason_counter = defaultdict(int)

    for i, row in enumerate(rows):
        new_ans, reason = correct_answer(row, items[i])
        reason_counter[reason] += 1

        new_row = dict(row)
        new_row["original_prediction"] = row["prediction"]
        new_row["prediction"] = f"Answer: {new_ans}"
        new_row["postprocess_reason"] = reason
        corrected.append(new_row)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in corrected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Saved:", OUTPUT_PATH)
    print("Reasons:")
    for k, v in sorted(reason_counter.items(), key=lambda x: -x[1]):
        print(k, v)


if __name__ == "__main__":
    main()
