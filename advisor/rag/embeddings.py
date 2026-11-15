"""Embedding backends: Vertex AI (text-embedding-005 / gemini-embedding) or an offline hashing embedder."""
from __future__ import annotations

import hashlib
import re

import numpy as np

from advisor.config import settings

_TOKEN = re.compile(r"[a-z0-9]+")
STOP = set("a an the of to and or in on for is are be by with as at it this that from how what why".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOP]


class HashingEmbedder:
    """Deterministic, dependency-free embedder (unigrams + bigrams hashed into a fixed space).

    Only for offline tests/CI -- quality is far below a real embedding model.
    """

    name = "offline-hashing-1024"

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        toks = tokenize(text)
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        v = np.zeros(self.dim, dtype=np.float32)
        for f in feats:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed(self, texts: list[str], task: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts])


class VertexEmbedder:
    def __init__(self, model: str | None = None):
        from advisor.llm import genai_client

        self.client = genai_client()
        self.name = model or settings().embedding_model

    def embed(self, texts: list[str], task: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        from google.genai import types

        vecs = []
        for i in range(0, len(texts), 32):  # batch to stay under request limits
            resp = self.client.models.embed_content(
                model=self.name,
                contents=texts[i : i + 32],
                config=types.EmbedContentConfig(task_type=task),
            )
            vecs.extend(e.values for e in resp.embeddings)
        arr = np.asarray(vecs, dtype=np.float32)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def get_embedder():
    return HashingEmbedder() if settings().offline else VertexEmbedder()
