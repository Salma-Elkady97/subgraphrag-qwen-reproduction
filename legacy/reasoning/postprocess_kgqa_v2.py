
import json
import re
import torch
from collections import defaultdict, Counter


# =========================
# PATHS
# =========================
RESULT_PATH = "/content/drive/MyDrive/SubgraphRAG/reason/results/KGQA/webqsp/PathBased/path_based_results.jsonl"
SCORE_DICT_PATH = "/content/drive/MyDrive/SubgraphRAG/webqsp_SOTA_READY.pth"
OUTPUT_PATH = "/content/drive/MyDrive/SubgraphRAG/reason/results/KGQA/webqsp/PathBased/postprocessed_v2_results.jsonl"


# =========================
# BASIC HELPERS
# =========================
def norm(x):
    x = str(x).lower().strip()
    x = re.sub(r"[^a-z0-9\s\.\-]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def words(x):
    return set(re.findall(r"[a-zA-Z0-9]+", str(x).lower()))


def clean_relation(r):
    return str(r).split(".")[-1].replace("_", " ").strip()


def extract_answer(pred):
    pred = str(pred)

    m = re.search(r"(?:ans|answer)\s*:\s*(.*)", pred, flags=re.I | re.S)
    if m:
        ans = m.group(1)
    else:
        ans = pred.split("\n")[-1]

    ans = re.split(r"<\|im_end\|>|question:|reasoning:", ans, flags=re.I)[0]
    return ans.strip(" .,:;\"'`[]()")


def is_bad_llm_answer(pred):
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
    return any(x in p for x in bad)


# =========================
# ALIAS MAP FOR FREEBASE IDS
# =========================
def build_alias_map(triples):
    aliases = {}

    for t in triples:
        try:
            if not isinstance(t, (list, tuple)) or len(t) < 3:
                continue

            s, r, o = str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()

            if "type.object.name" in r and s and o:
                aliases[s] = o

        except Exception:
            pass

    return aliases


def alias(x, aliases):
    return aliases.get(str(x).strip(), str(x).strip())


# =========================
# QUESTION TYPE
# =========================
def question_type(question):
    q = question.lower()

    if q.startswith("who") or " who " in q:
        return "person"

    if q.startswith("where") or "from" in q:
        return "location"

    if q.startswith("when") or "what year" in q or "date" in q:
        return "date"

    if "how many" in q or "number of" in q:
        return "number"

    if "currency" in q or "called" in q or "code" in q:
        return "code"

    if (
        q.startswith("what are")
        or q.startswith("who are")
        or "songs" in q
        or "countries" in q
        or "representatives" in q
    ):
        return "multi"

    return "entity"


# =========================
# RELATION MATCHING
# =========================
def relation_match_score(relation, question):
    q = question.lower()
    r = relation.lower()

    score = len(words(q) & words(r))

    rules = [
        # spouse / family
        (["wife", "husband", "spouse"], ["spouse", "wife", "husband", "partner"], 30),

        # government / offices
        (["governor"], ["governor", "government position", "office holder", "position held"], 30),
        (["representatives", "representative"], ["representative", "senator", "office holder"], 20),
        (["president", "take office"], ["inauguration", "start date", "office", "date"], 25),
        (["dictator", "leader"], ["leader", "head of government", "office holder"], 20),

        # elections
        (["run against", "opponent", "second term"], ["opponent", "candidate", "campaign", "election"], 30),

        # acting / roles
        (["plays", "played by", "who plays"], ["actor", "cast", "portrayed", "played by", "performance"], 30),
        (["who did", "play in", "role"], ["character", "role", "performance", "cast"], 30),
        (["voice"], ["voice actor", "actor", "cast"], 25),

        # dates
        (["open", "opened"], ["opening date", "first performance", "start date", "date"], 28),
        (["when", "year", "date"], ["date", "time", "year", "start date"], 18),

        # places
        (["where", "from"], ["place of birth", "location", "containedby", "city", "country"], 25),

        # currency / code
        (["currency", "called", "code"], ["currency code", "iso code", "code", "symbol", "currency"], 30),

        # music / works
        (["songs", "wrote", "written"], ["composition", "song", "writer", "author", "written"], 25),

        # sports
        (["play for", "played for", "team"], ["team", "sports team", "played for"], 25),

        # government form / type
        (["government"], ["form of government", "government type"], 25),

        # language
        (["language", "speak"], ["language", "languages spoken"], 25),

        # invention / profession
        (["invent", "invented"], ["invention", "inventor"], 25),
        (["do before", "occupation", "profession"], ["government positions held", "profession", "occupation"], 20),
    ]

    for q_terms, r_terms, boost in rules:
        if any(x in q for x in q_terms) and any(y in r for y in r_terms):
            score += boost

    return score


def type_bonus(candidate, question):
    qt = question_type(question)
    c = str(candidate).strip()

    if qt == "date":
        if re.search(r"\d{4}-\d{2}-\d{2}", c):
            return 30
        if re.search(r"\b\d{4}\b", c):
            return 12

    if qt == "number":
        if re.search(r"\b\d+(\.\d+)?\b", c):
            return 20

    if qt == "code":
        if re.fullmatch(r"[A-Z]{2,6}", c):
            return 30

    if qt == "person":
        if re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+", c):
            return 10

    return 0


# =========================
# CANDIDATE EXTRACTION
# =========================
def collect_object_candidates(item, top_k=400):
    raw_triples = item.get("scored_triplets", [])[:top_k]
    question = item.get("question", item.get("q_text", ""))

    aliases = build_alias_map(raw_triples)

    candidates = defaultdict(float)
    evidence = defaultdict(list)

    for idx, t in enumerate(raw_triples):
        try:
            if not isinstance(t, (list, tuple)) or len(t) < 3:
                continue

            s, r, o = str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()

            if not s or not r or not o:
                continue

            # skip metadata after alias extraction
            if "type.object.name" in r or "common." in r or "freebase." in r or "type.type" in r:
                continue

            s2 = alias(s, aliases)
            o2 = alias(o, aliases)
            cr = clean_relation(r)

            rank_bonus = max(0, top_k - idx) / top_k
            rel_bonus = relation_match_score(cr, question)
            overlap = len(words(question) & words(f"{s2} {cr} {o2}"))
            t_bonus = type_bonus(o2, question)

            score = rank_bonus + (5 * rel_bonus) + overlap + t_bonus

            # IMPORTANT: object side only
            candidates[o2] += score
            evidence[o2].append(f"{s2} -> {cr} -> {o2}")

        except Exception:
            continue

    ranked = sorted(candidates.keys(), key=lambda x: candidates[x], reverse=True)

    return ranked, candidates, evidence


# =========================
# KEEP / REPLACE DECISION
# =========================
def candidate_mentioned_in_prediction(c, pred):
    nc = norm(c)
    np = norm(pred)

    if not nc:
        return False

    return nc in np


def select_best_answer(row, item):
    question = row["question"]
    pred_text = row["prediction"]
    old_ans = extract_answer(pred_text)

    ranked, scores, evidence = collect_object_candidates(item, top_k=400)

    if not ranked:
        return old_ans, "no_candidates"

    qt = question_type(question)

    # 1. Multi-answer questions: return top objects from matching relations
    if qt == "multi":
        selected = []
        for c in ranked:
            ev = " ".join(evidence[c]).lower()
            rel_strength = max([relation_match_score(e, question) for e in evidence[c]]) if evidence[c] else 0
            if rel_strength >= 15:
                selected.append(c)
            if len(selected) >= 8:
                break

        if selected:
            return ", ".join(selected), "multi_object_extraction"

    # 2. Date questions: prefer exact date object
    if qt == "date":
        for c in ranked[:100]:
            if re.search(r"\d{4}-\d{2}-\d{2}", str(c)):
                return c, "exact_date_object"
        for c in ranked[:100]:
            if re.search(r"\b\d{4}\b", str(c)):
                return c, "year_object"

    # 3. Code/currency questions: prefer uppercase code if evidence supports it
    if qt == "code":
        for c in ranked[:100]:
            if re.fullmatch(r"[A-Z]{2,6}", str(c).strip()):
                return c, "code_object"

    # 4. If old answer already contains a good object candidate, keep best mentioned object
    mentioned = []
    for c in ranked[:80]:
        if candidate_mentioned_in_prediction(c, pred_text):
            mentioned.append(c)

    if mentioned:
        # choose highest scoring mentioned OBJECT, not subject
        mentioned = sorted(mentioned, key=lambda x: scores[x], reverse=True)
        return mentioned[0], "best_object_mentioned_in_prediction"

    # 5. Rescue not-found answers
    if is_bad_llm_answer(pred_text):
        return ranked[0], "rescue_not_found_with_top_object"

    # 6. Replace long/vague old answer only if top candidate is much stronger
    if len(old_ans.split()) > 8:
        return ranked[0], "replace_long_answer_with_top_object"

    # 7. Conservative default: keep original LLM answer
    return old_ans, "keep_original"


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

    assert len(rows) == len(items), f"Mismatch rows={len(rows)} items={len(items)}"

    reason_counter = Counter()
    corrected = []

    for i, row in enumerate(rows):
        new_ans, reason = select_best_answer(row, items[i])
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
    for k, v in reason_counter.most_common():
        print(k, v)


if __name__ == "__main__":
    main()
