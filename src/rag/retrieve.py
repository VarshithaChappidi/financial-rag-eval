import json
import pickle
import sys
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

from rag.index import CHUNKS, INDEX, MODEL, tokenize


class Retriever:
    def __init__(self) -> None:
        self.chunks = [json.loads(l) for l in CHUNKS.open(encoding="utf-8")]
        self.emb = np.load(INDEX / "emb.npy")
        with (INDEX / "bm25.pkl").open("rb") as f:
            self.bm25 = pickle.load(f)
        self.model = TextEmbedding(MODEL)

    def _mask(self, ticker: str | None) -> np.ndarray:
        if not ticker:
            return np.zeros(len(self.chunks))
        return np.array([0 if c["ticker"] == ticker else -1e9 for c in self.chunks])

    def dense(self, q: str, k: int = 30, ticker: str | None = None) -> list[int]:
        qv = np.array(list(self.model.query_embed(q))[0], dtype=np.float32)
        scores = self.emb @ qv + self._mask(ticker)
        return np.argsort(-scores)[:k].tolist()

    def sparse(self, q: str, k: int = 30, ticker: str | None = None) -> list[int]:
        scores = np.array(self.bm25.get_scores(tokenize(q))) + self._mask(ticker)
        return np.argsort(-scores)[:k].tolist()

    def hybrid(self, q: str, k: int = 5, pool: int = 30, ticker: str | None = None, rrf_k: int = 60) -> list[dict]:
        fused: dict[int, float] = {}
        for ranking in (self.dense(q, pool, ticker), self.sparse(q, pool, ticker)):
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [self.chunks[i] for i in top]


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "What are JPMorgan's main liquidity risks?"
    for c in Retriever().hybrid(query):
        print(f"\n[{c['ticker']}] {c['type']} | {c['section']}\n{c['text'][:400]}")