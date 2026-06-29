
import os
import pickle
from datasets import load_dataset
from tqdm import tqdm

BASE = "data_files/cwq"
OUT = f"{BASE}/processed"

os.makedirs(OUT, exist_ok=True)

dataset = load_dataset("rmanluo/RoG-cwq")

mapping = {
    "train": "train",
    "validation": "val",
    "test": "test"
}

for hf_split, out_split in mapping.items():

    print("=" * 100)
    print("Processing:", hf_split)

    rows = []

    for item in tqdm(dataset[hf_split]):

        row = {
            "id": item["id"],
            "question": item["question"],

            "q_entity": item.get("q_entity", []),
            "q_entity_id_list": item.get("q_entity_id_list", []),

            "text_entity_list": item.get("text_entity_list", []),
            "non_text_entity_list": item.get("non_text_entity_list", []),

            "relation_list": item.get("relation_list", []),

            "h_id_list": item.get("h_id_list", []),
            "r_id_list": item.get("r_id_list", []),
            "t_id_list": item.get("t_id_list", []),

            "a_entity": item.get("a_entity", []),
            "a_entity_id_list": item.get("a_entity_id_list", []),
        }

        rows.append(row)

    out_path = f"{OUT}/{out_split}.pkl"

    with open(out_path, "wb") as f:
        pickle.dump(rows, f)

    print("Saved:", out_path)
    print("Rows:", len(rows))

print("\nALL DONE")
