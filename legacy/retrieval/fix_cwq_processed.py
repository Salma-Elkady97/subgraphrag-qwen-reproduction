
import os
import pickle
from tqdm import tqdm

from src.dataset.retriever import RetrieverDataset

OUT_DIR = "data_files/cwq/processed"
os.makedirs(OUT_DIR, exist_ok=True)

for split in ["train", "val", "test"]:
    print(f"\nProcessing {split}...")

    dataset = RetrieverDataset(
        config={"dataset_name": "cwq"},
        split=split
    )

    data = dataset.processed_dict_list

    out_path = os.path.join(OUT_DIR, f"{split}.pkl")

    with open(out_path, "wb") as f:
        pickle.dump(data, f)

    print(f"Saved: {out_path} | items={len(data)}")

print("\nDONE")
