"""Tiny persistent vector store (numpy + JSON). Swap for Vertex AI Vector Search at scale."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from advisor.rag.chunker import Chunk


class VectorStore:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder_name: str):
        self.chunks, self.vectors, self.embedder_name = chunks, vectors, embedder_name

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        (directory / "chunks.json").write_text(
            json.dumps({"embedder": self.embedder_name, "chunks": [c.to_dict() for c in self.chunks]}, indent=1)
        )

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        meta = json.loads((directory / "chunks.json").read_text())
        return cls([Chunk(**c) for c in meta["chunks"]], np.load(directory / "vectors.npy"), meta["embedder"])

    def search(self, qvec: np.ndarray, k: int) -> list[tuple[int, float]]:
        sims = self.vectors @ qvec
        idx = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in idx]
