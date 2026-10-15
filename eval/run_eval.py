import json
import re
import time
from pathlib import Path

from rag.generate import Reranker
from rag.retrieve import Retriever

K = 5
QUESTIONS = Path("eval/questions.jsonl")
OUT = Path("eval/results.md")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower())


def hit_rank(chunks: list[dict], q: dict) -> int:
    options = [q["evidence"]] + q.get("alt", [])
    for i, c in enumerate(chunks):
        text = norm(c["text"])
        for opt in options:
            if all(norm(e) in text for e in opt):
                return i + 1
    return 0


def main() -> None:
    qs = [json.loads(l) for l in QUESTIONS.open(encoding="utf-8") if l.strip()]
    r, rr = Retriever(), Reranker()
    configs = {
        "dense only": lambda q, t: [r.chunks[i] for i in r.dense(q, K, t)],
        "BM25 only": lambda q, t: [r.chunks[i] for i in r.sparse(q, K, t)],
        "hybrid (RRF)": lambda q, t: r.hybrid(q, k=K, ticker=t),
        "hybrid, no company filter": lambda q, t: r.hybrid(q, k=K, ticker=None),
        "hybrid + reranker": lambda q, t: rr.rerank(q, r.hybrid(q, k=20, pool=40, ticker=t), k=K),
    }
    rows = []
    for name, fn in configs.items():
        ranks, start = [], time.time()
        for q in qs:
            ranks.append(hit_rank(fn(q["question"], q["ticker"]), q))
        misses = [q["id"] for q, x in zip(qs, ranks) if x == 0]
        print(f"{name} misses ({len(misses)}): {misses}")
        ms = (time.time() - start) * 1000 / len(qs)
        n = len(qs)
        rows.append((
            name,
            sum(1 for x in ranks if x == 1) / n,
            sum(1 for x in ranks if x > 0) / n,
            sum(1 / x for x in ranks if x > 0) / n,
            ms,
        ))
    lines = [
        f"Questions: {len(qs)}\n",
        "| Config | Hit@1 | Hit@5 | MRR@5 | Avg latency (ms) |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| {n} | {h1:.2f} | {h5:.2f} | {m:.2f} | {ms:.0f} |" for n, h1, h5, m, ms in rows]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()