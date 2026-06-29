
path = "/content/drive/MyDrive/SubgraphRAG/reason/llm_utils.py"

with open(path, "r", encoding="utf-8") as f:
    txt = f.read()

start = txt.index("def llm_inf_all")
new_func = r'''
def hard_trim_prompt(prompt, tokenizer, max_tokens=2850):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return prompt

    # Keep beginning instructions + ending candidates/question
    keep_start = int(max_tokens * 0.35)
    keep_end = max_tokens - keep_start
    new_ids = ids[:keep_start] + ids[-keep_end:]
    return tokenizer.decode(new_ids, skip_special_tokens=False)


def llm_inf_all(llm, tokenizer, data):
    all_candidates = [build_candidates(qa)[1] for qa in data]
    messages = [form_tiered_prompt(qa) for qa in data]
    prompts = [build_prompt_with_template(m, tokenizer) for m in messages]

    # IMPORTANT: prevent vLLM context crash
    prompts = [hard_trim_prompt(p, tokenizer, max_tokens=2850) for p in prompts]

    lengths = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
    print(f"Prompt tokens | max={max(lengths)} | avg={sum(lengths)//len(lengths)} | n={len(lengths)}")

    temps = [0.0, 0.2]
    all_runs = []

    for temp in temps:
        params = SamplingParams(
            temperature=temp,
            top_p=0.90,
            max_tokens=40,
            repetition_penalty=1.02,
            stop=["<|im_end|>", "\nQuestion:", "\nFacts:", "\nCandidate"]
        )

        outputs = llm.generate(prompts, params)

        preds = []
        for i, out in enumerate(outputs):
            raw = clean_model_output(out.outputs[0].text)
            mapped = match_to_candidate(raw, all_candidates[i])

            if mapped:
                preds.append(mapped)
            else:
                preds.append(deterministic_fallback(data[i]["question"], all_candidates[i]))

        all_runs.append(preds)

    final = []
    for i in range(len(data)):
        voted = majority_vote([all_runs[j][i] for j in range(len(all_runs))])
        if voted:
            final.append(voted)
        else:
            final.append(deterministic_fallback(data[i]["question"], all_candidates[i]))

    return final
'''

txt = txt[:start] + new_func

with open(path, "w", encoding="utf-8") as f:
    f.write(txt)

print("Patched:", path)
