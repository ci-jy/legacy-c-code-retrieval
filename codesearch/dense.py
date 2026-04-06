"""Dense embeddings from a local sentence-transformers model on the CPU."""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

DEFAULT_MODEL = os.environ.get("CODESEARCH_MODEL", "BAAI/bge-small-en-v1.5")

# Some retrieval models expect an instruction in front of queries.
QUERY_PREFIXES = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


@lru_cache(maxsize=4)
def _load(model_name: str):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(0)
    try:
        # Use the local Hugging Face cache without touching the network when possible.
        model = SentenceTransformer(model_name, device="cpu", local_files_only=True)
    except OSError:
        model = SentenceTransformer(model_name, device="cpu")
    model.eval()
    return model


class Embedder:
    """Wraps a SentenceTransformer; vectors are L2-normalised so dot product is cosine."""

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size

    @property
    def model(self):
        return _load(self.model_name)

    def encode_documents(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        return self._encode(texts, show_progress)

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode_queries([query])[0]

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        prefix = QUERY_PREFIXES.get(self.model_name, "")
        return self._encode([prefix + q for q in queries], False)

    def _encode(self, texts: list[str], show_progress: bool) -> np.ndarray:
        import torch

        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        with torch.inference_mode():
            vecs = self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=show_progress,
            )
        return np.asarray(vecs, dtype=np.float32)
