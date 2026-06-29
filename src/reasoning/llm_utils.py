
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import re
import torch
from collections import Counter


def llm_init(model_name_or_path, max_seq_len_to_capture=4096):
    quant = None
    m = model_name_or_path.lower()

    if "awq" in m:
        quant = "awq"
    elif "gptq" in m:
        quant = "gptq"

    print("GPU:", torch.cuda.get_device_name(0))
    print("Model:", model_name_or_path)
    print("Quantization:", quant)

    llm = LLM(
        model=model_name_or_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_model_len=max_seq_len_to_capture,
        gpu_memory_utilization=0.82,
        quantization="awq_marlin",
        dtype="float16",
        enforce_eager=False,
        disable_log_stats=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=True
    )

    return llm, tokenizer


def clean_rel(r):
    return str(r).split(".")[-1].replace("_", " ").strip()


def normalize_text(x):
    x = str(x).strip()
    x = re.sub(r"<\|.*?\|>", " ", x)
    x = re.sub(r"^(ans|answer|final answer)\s*:\s*", "", x, flags=re.I).strip()
    x = x.split("\n")[0].strip()
    x = re.sub(r"[^a-zA-Z0-9\.\-:/|, ]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x.lower()


def build_name_map(raw_triplets):
    name_map = {}

    for item in raw_triplets:
        try:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue

            s = str(item[0]).strip()
            r = str(item[1]).strip()
            o = str(item[2]).strip()

            if "type.object.name" in r and s and o:
                name_map[s] = o
        except:
            continue

    return name_map


def replace_id(x, name_map):
    x = str(x).strip()
    return name_map.get(x, x)


def is_bad_answer(x):
    bad = [
        "date if exists",
        "date if applicable",
        "answer entity",
        "final answer",
        "provided facts",
        "information is missing",
        "do not explain",
        "use only",
        "copy the instructions",
        "cannot determine",
        "not available",
        "unknown"
    ]

    low = str(x).lower()
    return any(b in low for b in bad)


def adaptive_fact_limits(question):
    """
    A100-safe CWQ version.
    Keeps retrieval order, but allows more evidence than the 3072-token run.
    """
    q_len = len(str(question).split())

    if q_len <= 10:
        return 95, 70
    elif q_len <= 18:
        return 80, 60
    elif q_len <= 28:
        return 65, 50
    else:
        return 50, 38


def form_tiered_prompt(qa_pair):
    question = qa_pair["question"]
    raw_triplets = qa_pair.get("scored_triplets", [])

    max_facts, max_candidates = adaptive_fact_limits(question)
    name_map = build_name_map(raw_triplets)

    facts = []
    candidates = []
    seen_facts = set()
    seen_candidates = set()

    # Keep original retrieval order. This is the part that gave your best CWQ score.
    scan_limit = min(len(raw_triplets), max_facts * 3)

    for item in raw_triplets[:scan_limit]:
        try:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue

            s_raw = str(item[0]).strip()
            r_raw = str(item[1]).strip()
            o_raw = str(item[2]).strip()

            if not s_raw or not r_raw or not o_raw:
                continue

            if "type.object.name" in r_raw:
                continue

            s = replace_id(s_raw, name_map)
            r = clean_rel(r_raw)
            o = replace_id(o_raw, name_map)

            key = (s, r, o)
            if key in seen_facts:
                continue

            seen_facts.add(key)

            if len(facts) < max_facts:
                facts.append(f"{len(facts)+1}. {s} -- {r} --> {o}")

            # Keep tail candidates only. Previous head+tail candidate attempts reduced Hit.
            if o and o not in seen_candidates and len(candidates) < max_candidates:
                seen_candidates.add(o)
                candidates.append(o)

            if len(facts) >= max_facts and len(candidates) >= max_candidates:
                break

        except:
            continue

    facts_text = "\n".join(facts)
    candidates_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidates)])

    system_msg = (
        "You are a strict Knowledge Graph QA answer selector. "
        "Use only the provided facts and candidate answers. "
        "Return only the final answer text. "
        "Never explain. "
        "Never invent an answer. "
        "Never output Freebase IDs if readable names exist. "
        "For complex multi-hop questions, follow the facts carefully and return the final requested entity, not an intermediate entity. "
        "If multiple answers are correct, separate them using |."
    )

    user_msg = (
        f"Question:\n{question}\n\n"
        f"Facts:\n{facts_text}\n\n"
        f"Candidate Answers:\n{candidates_text}\n\n"
        "Return only the final answer text."
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]


form_path_prompt = form_tiered_prompt
form_hybrid_prompt = form_tiered_prompt


def build_prompt_with_template(messages, tokenizer):
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except:
        return (
            f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
            f"<|im_start|>user\n{messages[1]['content']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )


def force_token_safe_prompt(prompt, tokenizer, max_input_tokens=3800):
    ids = tokenizer.encode(prompt, add_special_tokens=False)

    if len(ids) <= max_input_tokens:
        return prompt

    keep_start = int(max_input_tokens * 0.78)
    keep_end = max_input_tokens - keep_start

    new_ids = ids[:keep_start] + ids[-keep_end:]
    return tokenizer.decode(new_ids, skip_special_tokens=False)


def extract_answer(txt):
    txt = str(txt).strip()

    if is_bad_answer(txt):
        return ""

    txt = re.sub(r"^(ans|answer|final answer)\s*:\s*", "", txt, flags=re.I).strip()
    txt = txt.split("\n")[0].strip()

    if is_bad_answer(txt):
        return ""

    return normalize_text(txt)


def majority_vote(items):
    items = [x for x in items if x and not is_bad_answer(x)]

    if not items:
        return ""

    return Counter(items).most_common(1)[0][0]


def llm_inf_all(llm, tokenizer, data):
    messages = [form_tiered_prompt(qa) for qa in data]
    prompts = [build_prompt_with_template(m, tokenizer) for m in messages]

    prompts = [
        force_token_safe_prompt(p, tokenizer, max_input_tokens=3800)
        for p in prompts
    ]

    lengths = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
    print(f"Prompt tokens | max={max(lengths)} | avg={sum(lengths)//len(lengths)} | n={len(lengths)}")
    print("Debug token count > 3600:", sum(1 for x in lengths if x > 3600))
    print("Debug token count > 3800:", sum(1 for x in lengths if x > 3800))

    temps = [0.0, 0.15, 0.3]
    all_preds = []

    for temp in temps:
        print(f"Running temperature={temp}")

        params = SamplingParams(
            temperature=temp,
            top_p=0.90,
            max_tokens=64,
            repetition_penalty=1.03,
            stop=["<|im_end|>", "\nQuestion:", "\nFacts:", "\nCandidate"]
        )

        outputs = llm.generate(prompts, params)
        preds = [extract_answer(o.outputs[0].text) for o in outputs]
        all_preds.append(preds)

    final_preds = []

    for i in range(len(data)):
        final_preds.append(
            majority_vote([
                all_preds[0][i],
                all_preds[1][i],
                all_preds[2][i]
            ])
        )

    return final_preds
