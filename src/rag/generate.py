import os
import sys

import anthropic
from dotenv import load_dotenv
from fastembed.rerank.cross_encoder import TextCrossEncoder

from rag.retrieve import Retriever

load_dotenv()
MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-5-5")
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

NAMES = {
    "JPM": ["jpmorgan", "jp morgan", "chase"],
    "BAC": ["bank of america", "bofa"],
    "MS": ["morgan stanley"],
    "GS": ["goldman"],
    "C": ["citi", "citigroup"],
}

SYSTEM = (
    "You are a financial analyst assistant. Answer using ONLY the numbered sources. "
    "Cite every claim like [1]. Quote numbers exactly as written and state the fiscal "
    "year and the basis (for example Standardized vs Advanced). If the sources do not "
    "contain the answer, reply exactly: Not found in the provided filings."
)


def detect_ticker(q: str) -> str | None:
    ql = q.lower()
    hits = [t for t, names in NAMES.items() if any(n in ql for n in names)]
    return hits[0] if len(hits) == 1 else None


class Reranker:
    def __init__(self) -> None:
        self.model = TextCrossEncoder(model_name=RERANK_MODEL)

    def rerank(self, q: str, chunks: list[dict], k: int = 5) -> list[dict]:
        docs = [f"{c['section']}\n{c['text']}" for c in chunks]
        scores = list(self.model.rerank(q, docs))
        order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:k]
        return [chunks[i] for i in order]


def answer(q: str, retriever: Retriever, reranker: Reranker) -> tuple[str, list[dict]]:
    ticker = detect_ticker(q)
    candidates = retriever.hybrid(q, k=20, pool=40, ticker=ticker)
    top = reranker.rerank(q, candidates, k=5)
    context = "\n\n".join(
        f"[{i + 1}] ({c['ticker']} | {c['section']})\n{c['text']}" for i, c in enumerate(top)
    )
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=MODEL,
        max_tokens=500,
        system=SYSTEM,
        messages=[{"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {q}"}],
    )
    return msg.content[0].text, top


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What was Goldman Sachs CET1 capital ratio?"
    text, sources = answer(question, Retriever(), Reranker())
    print("\nANSWER:\n" + text)
    print("\nSOURCES:")
    for i, c in enumerate(sources):
        print(f"[{i + 1}] {c['ticker']} | {c['type']} | {c['section']}")