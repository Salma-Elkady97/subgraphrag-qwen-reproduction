
import os
from datasets import load_dataset
from src.config.emb import load_yaml
from src.dataset.emb import EmbInferDataset

dataset = "cwq"
config_file = f"configs/emb/gte-large-en-v1.5/{dataset}.yaml"
config = load_yaml(config_file)

input_file = os.path.join("rmanluo", "RoG-cwq")

train_set = load_dataset(input_file, split="train")
val_set = load_dataset(input_file, split="validation")
test_set = load_dataset(input_file, split="test")

entity_identifiers = []
with open(config["entity_identifier_file"], "r") as f:
    for line in f:
        entity_identifiers.append(line.strip())
entity_identifiers = set(entity_identifiers)

save_dir = f"data_files/{dataset}/processed"
os.makedirs(save_dir, exist_ok=True)

print("Saving processed train...")
train_processed = EmbInferDataset(
    train_set,
    entity_identifiers,
    os.path.join(save_dir, "train.pkl")
)

print("Saving processed val...")
val_processed = EmbInferDataset(
    val_set,
    entity_identifiers,
    os.path.join(save_dir, "val.pkl")
)

print("Saving processed test...")
test_processed = EmbInferDataset(
    test_set,
    entity_identifiers,
    os.path.join(save_dir, "test.pkl"),
    skip_no_topic=False,
    skip_no_ans=False
)

print("DONE")
print("train:", len(train_processed))
print("val:", len(val_processed))
print("test:", len(test_processed))
print("Files saved to:", save_dir)
