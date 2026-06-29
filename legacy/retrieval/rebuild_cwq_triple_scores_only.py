
import os
import pickle
import torch
import networkx as nx
from tqdm import tqdm

BASE = "data_files/cwq"
PROCESSED_DIR = f"{BASE}/processed"
OUT_DIR = f"{BASE}/triple_scores"

os.makedirs(OUT_DIR, exist_ok=True)


def get_nx_g(h_id_list, r_id_list, t_id_list):
    g = nx.DiGraph()

    for i, (h, r, t) in enumerate(zip(h_id_list, r_id_list, t_id_list)):
        try:
            g.add_edge(int(h), int(t), triple_id=i, relation_id=int(r))
        except:
            pass

    return g


def shortest_paths(g, q_entity_id, a_entity_id):
    paths = []

    try:
        paths.extend(list(nx.all_shortest_paths(g, q_entity_id, a_entity_id)))
    except:
        pass

    try:
        paths.extend(list(nx.all_shortest_paths(g, a_entity_id, q_entity_id)))
    except:
        pass

    if not paths:
        return []

    min_len = min(len(p) for p in paths)
    return [p for p in paths if len(p) == min_len]


def extract_paths_and_score(sample):
    h_ids = sample["h_id_list"]
    r_ids = sample["r_id_list"]
    t_ids = sample["t_id_list"]

    g = get_nx_g(h_ids, r_ids, t_ids)

    path_list = []

    for qid in sample.get("q_entity_id_list", []):
        for aid in sample.get("a_entity_id_list", []):
            try:
                paths = shortest_paths(g, int(qid), int(aid))
                path_list.extend(paths)
            except:
                pass

    triple_scores = torch.zeros(len(h_ids))

    max_path_length = None

    for path in path_list:
        if len(path) < 2:
            continue

        num_triples = len(path) - 1
        max_path_length = num_triples if max_path_length is None else max(max_path_length, num_triples)

        for i in range(num_triples):
            h = path[i]
            t = path[i + 1]

            try:
                triple_id = g[h][t]["triple_id"]
                triple_scores[triple_id] = 1.0
            except:
                pass

    return triple_scores, max_path_length


def build_split(split):
    pkl_path = f"{PROCESSED_DIR}/{split}.pkl"
    out_path = f"{OUT_DIR}/{split}.pth"

    print("=" * 100)
    print("Split:", split)
    print("Loading:", pkl_path)

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    result = {}

    for sample in tqdm(data, desc=f"Scoring {split}"):
        qid = sample["id"]
        triple_scores, max_path_length = extract_paths_and_score(sample)

        result[qid] = {
            "triple_scores": triple_scores,
            "max_path_length": max_path_length
        }

    torch.save(result, out_path)

    print("Saved:", out_path)
    print("Items:", len(result))


for split in ["train", "val"]:
    build_split(split)

print("DONE")
