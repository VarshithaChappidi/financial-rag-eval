import json
import pickle
import re
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi

CHUNKS = Path("data/chunks.jsonl")
INDEX = Path("data/index")
MODEL = "BAAI/bge-small-en-v1.5"


def tokenize(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


def main() -> None:
    INDEX.mkdir(parents=True, exist_ok=True)
    chunks = [json.loads(l) for l in CHUNKS.open(encoding="utf-8")]
    docs = [f"{c['ticker']} | {c['section']}\n{c['text']}" for c in chunks]

    print("building BM25...")
    bm25 = BM25Okapi([tokenize(d) for d in docs])
    with (INDEX / "bm25.pkl").open("wb") as f:
        pickle.dump(bm25, f)

    print(f"embedding {len(docs)} chunks (a few minutes on CPU)...")
    model = TextEmbedding(MODEL)
    emb = np.array(list(model.embed(docs, batch_size=64)), dtype=np.float32)
    np.save(INDEX / "emb.npy", emb)
    print("done:", emb.shape)


if __name__ == "__main__":
    main()