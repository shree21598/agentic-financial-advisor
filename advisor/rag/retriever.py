"""Hybrid retriever: dense vectors + BM25, fused with reciprocal-rank fusion (RRF)."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from advisor.config import CORPUS_DIR, INDEX_DIR, settings
from advisor.rag.chunker import Chunk, chunk_corpus
from advisor.rag.embeddings import get_embedder, tokenize
from advisor.rag.store import VectorStore


@dataclass
class Hit:
    chunk: Chunk
    score: float

    def to_dict(self) -> dict:
        return {
            "source": f"{self.chunk.doc} > {self.chunk.section}",
            "chunk_id": self.chunk.id,
            "score": round(self.score, 4),
            "text": self.chunk.text,
        }


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.toks = [tokenize(d) for d in docs]
        self.k1, self.b = k1, b
        self.avg = sum(map(len, self.toks)) / max(len(self.toks), 1)
        df = Counter(t for doc in self.toks for t in set(doc))
        n = len(self.toks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in self.toks]

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for tf, doc in zip(self.tf, self.toks):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(doc) / self.avg))
            out.append(s)
        return out


class HybridRetriever:
    def __init__(self, store: VectorStore, embedder):
        self.store, self.embedder = store, embedder
        self.bm25 = BM25([c.text for c in store.chunks])

    def search(self, query: str, top_k: int = 4, rrf_k: int = 60) -> list[Hit]:
        n = len(self.store.chunks)
        qv = self.embedder.embed([query], task="RETRIEVAL_QUERY")[0]
        dense = self.store.search(qv, min(n, 20))
        bm = sorted(enumerate(self.bm25.scores(query)), key=lambda x: -x[1])[:20]
        fused: dict[int, float] = {}
        for ranking in (dense, bm):
            for rank, (i, _) in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1.0 / (rrf_k + rank + 1)
        best = sorted(fused.items(), key=lambda x: -x[1])[:top_k]
        return [Hit(self.store.chunks[i], s) for i, s in best]


def build_index(force: bool = False) -> VectorStore:
    embedder = get_embedder()
    target = INDEX_DIR / embedder.name.replace("/", "_")
    if target.exists() and not force:
        return VectorStore.load(target)
    chunks = chunk_corpus(CORPUS_DIR)
    vecs = embedder.embed([c.text for c in chunks], task="RETRIEVAL_DOCUMENT")
    store = VectorStore(chunks, vecs, embedder.name)
    store.save(target)
    return store


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    return HybridRetriever(build_index(), get_embedder())


if __name__ == "__main__":  # python -m advisor.rag.retriever "query"
    import sys

    s = build_index(force="--rebuild" in sys.argv)
    print(f"index: {len(s.chunks)} chunks ({s.embedder_name}) top_k={settings().top_k}")
    q = " ".join(a for a in sys.argv[1:] if not a.startswith("--"))
    if q:
        for h in get_retriever().search(q):
            print(f"{h.score:.4f}  {h.chunk.id}  {h.chunk.text[:90]}")
