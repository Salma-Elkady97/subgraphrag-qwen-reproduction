
import os
import argparse
import torch
from tqdm import tqdm

from src.dataset.retriever import RetrieverDataset
from src.model.retriever import Retriever


def inference(model, dataset, device, max_K=200):
    model.eval()

    retrieval_result = {}

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )

    with torch.no_grad():
        for batch in tqdm(loader):

            qid = batch["id"][0]

            out = model(batch)

            triple_scores = out["triple_score"][0].cpu()

            topk = min(max_K, len(triple_scores))

            vals, idxs = torch.topk(triple_scores, k=topk)

            scored_triplets = []

            h_list = batch["h_id_list"][0]
            r_list = batch["r_id_list"][0]
            t_list = batch["t_id_list"][0]

            text_entity_list = batch["text_entity_list"][0]
            non_text_entity_list = batch["non_text_entity_list"][0]
            relation_list = batch["relation_list"][0]

            def entity_name(eid):
                eid = int(eid)

                if eid < len(text_entity_list):
                    return text_entity_list[eid]

                j = eid - len(text_entity_list)

                if 0 <= j < len(non_text_entity_list):
                    return non_text_entity_list[j]

                return str(eid)

            for rank_idx in idxs.tolist():

                try:
                    h = entity_name(h_list[rank_idx])
                    r = relation_list[int(r_list[rank_idx])]
                    t = entity_name(t_list[rank_idx])

                    scored_triplets.append((h, r, t))

                except:
                    pass

            retrieval_result[qid] = {
                "question": batch["question"][0],
                "scored_triplets": scored_triplets,
                "q_entity": batch["q_entity"][0],
                "a_entity": batch["a_entity"][0],
            }

    return retrieval_result


def main(args):

    checkpoint = torch.load(args.path, map_location="cpu")

    config = checkpoint["config"]

    dataset_name = config["dataset"]["name"]

    config["dataset"]["split"] = args.split

    dataset = RetrieverDataset(
        config=config,
        split=args.split
    )

    model = Retriever(config)
    model.load_state_dict(checkpoint["model_state_dict"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    result = inference(
        model,
        dataset,
        device=device,
        max_K=args.max_K
    )

    out_dir = os.path.dirname(args.path)

    out_path = os.path.join(
        out_dir,
        f"retrieval_result_{args.split}.pth"
    )

    torch.save(result, out_path)

    print("=" * 100)
    print("Saved:", out_path)
    print("Items:", len(result))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p",
        "--path",
        required=True
    )

    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "val", "test"]
    )

    parser.add_argument(
        "--max_K",
        type=int,
        default=200
    )

    args = parser.parse_args()

    main(args)
