
import os
import torch
import pickle
from tqdm import tqdm

from src.config.emb import load_yaml
from src.model.text_encoders import GTELargeEN


DATASET = "cwq"
CONFIG_FILE = f"configs/emb/gte-large-en-v1.5/{DATASET}.yaml"

PROCESSED_DIR = f"data_files/{DATASET}/processed"
SAVE_DIR = f"data_files/{DATASET}/emb/gte-large-en-v1.5"

os.makedirs(SAVE_DIR, exist_ok=True)

config = load_yaml(CONFIG_FILE)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
text_encoder = GTELargeEN(device)


def move_cpu(x):
    if torch.is_tensor(x):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: move_cpu(v) for k, v in x.items()}
    if isinstance(x, list):
        return [move_cpu(v) for v in x]
    return x


def encode_split(split):
    pkl_path = os.path.join(PROCESSED_DIR, f"{split}.pkl")
    save_path = os.path.join(SAVE_DIR, f"{split}.pth")

    print("=" * 100)
    print("Split:", split)
    print("Loading:", pkl_path)

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(pkl_path)

    with open(pkl_path, "rb") as f:
        subset = pickle.load(f)

    print("Items:", len(subset))

    emb_dict = {}

    for sample in tqdm(subset, desc=f"Encoding {split}"):
        qid = sample["id"]
        q_text = sample["question"]
        text_entity_list = sample["text_entity_list"]
        relation_list = sample["relation_list"]

        q_emb, entity_embs, relation_embs = text_encoder(
            q_text,
            text_entity_list,
            relation_list
        )

        emb_dict[qid] = {
            "q_emb": move_cpu(q_emb),
            "entity_embs": move_cpu(entity_embs),
            "relation_embs": move_cpu(relation_embs),
        }

    print("Saving:", save_path)
    torch.save(emb_dict, save_path)

    if not os.path.exists(save_path):
        raise RuntimeError(f"Save failed: {save_path}")

    print("Saved OK:", save_path)
    print("File size MB:", os.path.getsize(save_path) / (1024 * 1024))


for split in ["train", "val", "test"]:
    encode_split(split)

print("ALL DONE")
