
import os
import argparse
import json
import torch
from llm_utils import llm_init, llm_inf_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="webqsp")
    parser.add_argument("-m", "--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("-p", "--score_dict_path", type=str, required=True)
    parser.add_argument("--max_seq_len_to_capture", type=int, default=4096)
    args = parser.parse_args()

    print(f"Loading Foundation Data: {args.score_dict_path}")
    foundation_data = torch.load(args.score_dict_path)

    data = []
    for qid in foundation_data:
        item = foundation_data[qid]
        item["question"] = item.get("question", item.get("q_text", ""))
        data.append(item)

    llm, tokenizer = llm_init(args.model, args.max_seq_len_to_capture)
    predictions = llm_inf_all(llm, tokenizer, data)

    output_dir = os.path.join("results", "KGQA", args.dataset, "Baseline")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "baseline_results.jsonl")

    with open(output_path, "w", encoding="utf-8") as f:
        for i, qa_pair in enumerate(data):
            f.write(json.dumps({
                "id": qa_pair.get("id", i),
                "question": qa_pair["question"],
                "prediction": predictions[i],
                "ground_truth": qa_pair.get("a_entity", []),
            }, ensure_ascii=False) + "\n")

    print(f"Inference Finished. Results: {output_path}")


if __name__ == "__main__":
    main()
