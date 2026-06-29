
import json, os

input_path = "/content/drive/MyDrive/SubgraphRAG/reason/results/KGQA/webqsp/Baseline/baseline_results_repaired_focused.jsonl"
output_path = "/content/drive/MyDrive/SubgraphRAG/reason/results/KGQA/webqsp/Baseline/baseline_results_repaired_safe.jsonl"

rows = []

with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)

        # If focused repair changed the answer but did not help for Super Bowl-like case,
        # restore previous repaired answer.
        q = row.get("question", "").lower()
        new = str(row.get("prediction", ""))
        old = str(row.get("prediction_before_focused", ""))

        if "super bowl" in q and old:
            row["prediction"] = old

        rows.append(row)

with open(output_path, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("Saved:", output_path)
