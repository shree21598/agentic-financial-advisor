from advisor.config import CORPUS_DIR
from advisor.rag.chunker import chunk_corpus
from advisor.rag.retriever import get_retriever


def test_chunks_are_self_describing():
    chunks = chunk_corpus(CORPUS_DIR)
    assert len(chunks) > 30
    assert all(" - " in c.text and c.id for c in chunks)


def test_retrieval_finds_relevant_doc():
    r = get_retriever()
    cases = {
        "what is the 4% rule": "safe_withdrawal.md",
        "avalanche vs snowball": "debt_payoff.md",
        "wash sale rule": "taxes_investing.md",
        "why do bond prices fall when rates rise": "bonds.md",
    }
    for q, doc in cases.items():
        assert doc in {h.chunk.doc for h in r.search(q, top_k=3)}, q
