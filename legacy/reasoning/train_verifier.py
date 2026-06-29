
import os, re, json, argparse, random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def norm(x):
    x = str(x).lower().strip()
    x = re.sub(r"[^a-z0-9\s\.\-:/|,]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def clean_rel(r):
    return str(r).split(".")[-1].replace("_", " ").strip()


def build_name_map(triples):
    mp = {}
    for t in triples:
        try:
            if len(t) >= 3 and "type.object.name" in str(t[1]):
                mp[str(t[0]).strip()] = str(t[2]).strip()
        except:
            pass
    return mp


def is_match(candidate, golds):
    c = norm(candidate)
    for g in golds:
        g = norm(g)
        if g and (g == c or g in c or c in g):
            return True
    return False


def generate_candidates(question, triples, golds=None, top_k=250):
    name_map = build_name_map(triples)
    cand = {}

    for rank, t in enumerate(triples[:top_k]):
        try:
            if len(t) < 3:
                continue

            s0, r0, o0 = str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()

            if "type.object.name" in r0:
                continue

            s = name_map.get(s0, s0)
            r = clean_rel(r0)
            o = name_map.get(o0, o0)

            if not o or norm(o).startswith("m."):
                continue

            k = norm(o)
            if k not in cand:
                cand[k] = {
                    "candidate": o,
                    "evidence": [],
                    "rank": rank
                }

            if len(cand[k]["evidence"]) < 6:
                cand[k]["evidence"].append(f"{s} -- {r} --> {o}")

        except:
            pass

    return list(cand.values())[:80]


def build_text(question, cand):
    ev = "\n".join(cand["evidence"][:6])
    return (
        f"Question: {question}\n"
        f"Candidate answer: {cand['candidate']}\n"
        f"Evidence:\n{ev}\n"
        f"Label whether the candidate is the correct answer."
    )


def load_items(path):
    obj = torch.load(path)
    items = []
    for k in obj:
        x = obj[k]
        q = x.get("question", x.get("q_text", ""))
        triples = x.get("scored_triplets", [])
        golds = x.get("a_entity", [])
        items.append((q, triples, golds))
    return items


def build_training_examples(items, max_neg=8, top_k=250):
    examples = []

    for q, triples, golds in tqdm(items, desc="Building train examples"):
        cands = generate_candidates(q, triples, golds, top_k=top_k)

        positives = []
        negatives = []

        for c in cands:
            label = 1 if is_match(c["candidate"], golds) else 0
            if label:
                positives.append(c)
            else:
                negatives.append(c)

        if not positives:
            continue

        random.shuffle(negatives)

        for p in positives:
            examples.append((build_text(q, p), 1))

        for n in negatives[:max_neg * max(1, len(positives))]:
            examples.append((build_text(q, n), 0))

    random.shuffle(examples)
    return examples


class VerifierDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len=384):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text, label = self.examples[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "labels": torch.tensor(label, dtype=torch.long)
        }


def train(args):
    random.seed(42)
    torch.manual_seed(42)

    print("Loading train:", args.train_pth)
    items = load_items(args.train_pth)

    train_items, dev_items = train_test_split(items, test_size=0.1, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2
    )

    train_examples = build_training_examples(train_items, args.max_neg, args.top_k)
    dev_examples = build_training_examples(dev_items, args.max_neg, args.top_k)

    print("Train examples:", len(train_examples))
    print("Dev examples:", len(dev_examples))

    train_ds = VerifierDataset(train_examples, tokenizer, args.max_len)
    dev_ds = VerifierDataset(dev_examples, tokenizer, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    best_acc = -1

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} train"):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(dev_loader, desc=f"Epoch {epoch+1} dev"):
                labels = batch["labels"].to(device)
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total if total else 0
        print(f"Epoch {epoch+1} loss={total_loss/len(train_loader):.4f} dev_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            os.makedirs(args.output_dir, exist_ok=True)
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print("Saved best:", args.output_dir)

    print("Best dev acc:", best_acc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pth", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_model", default="microsoft/deberta-v3-base")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=384)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--top_k", type=int, default=250)
    parser.add_argument("--max_neg", type=int, default=8)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
