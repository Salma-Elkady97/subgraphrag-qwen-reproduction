import torch
from transformers import AutoTokenizer, AutoModel


class GTELargeEN:
    """
    SAFE encoder replacement (no Alibaba gte-large-en-v1.5, no trust_remote_code, no xformers).

    Keeps the same interface expected by SubgraphRAG:
      __call__(q_text, text_entity_list, relation_list) -> (q_emb, entity_embs, relation_embs)

    Uses: sentence-transformers/all-mpnet-base-v2
    Adds batching to avoid GPU OOM when encoding many entities/relations.
    """

    def __init__(self, device, max_length=256, batch_size=16):
        self.device = device

        # Stable encoder (does NOT require xformers / remote code)
        self.model_name = "sentence-transformers/all-mpnet-base-v2"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        # Keep these conservative to avoid OOM on long entity lists
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)

        print(f"✅ Using encoder: {self.model_name} | device={self.device} | max_length={self.max_length} | batch_size={self.batch_size}")

    @torch.no_grad()
    def embed(self, texts):
        """
        Returns: torch.FloatTensor [N, H] on CPU
        """
        if not texts:
            return torch.zeros((0, 768), dtype=torch.float32)

        outs = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]

            batch = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            last_hidden = self.model(**batch).last_hidden_state  # [B, L, H]
            emb = last_hidden.mean(dim=1)  # [B, H] mean pooling
            outs.append(emb.detach().cpu())

        return torch.cat(outs, dim=0)

    def __call__(self, q_text, text_entity_list, relation_list):
        q_emb = self.embed([q_text])[0]
        entity_embs = self.embed(text_entity_list)
        relation_embs = self.embed(relation_list)
        return q_emb, entity_embs, relation_embs