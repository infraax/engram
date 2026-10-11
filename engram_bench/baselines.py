"""MEANING-sense baselines: dedicated text-embedding models (the H5 null hypothesis)."""

from __future__ import annotations

import numpy as np

EMBEDDERS = {
    "minilm": {"id": "sentence-transformers/all-MiniLM-L6-v2", "q": "", "d": "", "kw": {}},
    "bge-small": {"id": "BAAI/bge-small-en-v1.5",
                  "q": "Represent this sentence for searching relevant passages: ", "d": "", "kw": {}},
    "nomic": {"id": "nomic-ai/nomic-embed-text-v1.5", "q": "search_query: ", "d": "search_document: ",
              "kw": {"trust_remote_code": True}},
}


class Embedder:
    def __init__(self, key: str, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        spec = EMBEDDERS[key]
        self.key, self.spec = key, spec
        self.model = SentenceTransformer(spec["id"], device=device, **spec["kw"])

    def _enc(self, texts: list[str], prefix: str, bs: int = 32) -> np.ndarray:
        return self.model.encode([prefix + t for t in texts], batch_size=bs, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    def docs(self, texts: list[str]) -> np.ndarray:
        return self._enc(texts, self.spec["d"])

    def queries(self, texts: list[str]) -> np.ndarray:
        return self._enc(texts, self.spec["q"])
