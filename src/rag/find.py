import json
import sys

from rag.index import CHUNKS


def main() -> None:
    ticker, phrase = sys.argv[1], " ".join(sys.argv[2:]).lower()
    for line in CHUNKS.open(encoding="utf-8"):
        c = json.loads(line)
        low = c["text"].lower()
        if c["ticker"] == ticker and phrase in low:
            i = low.index(phrase)
            print(f"[{c['id']}] {c['type']} | {c['section']}")
            print(f"   ...{c['text'][max(0, i - 80): i + 200]}...\n")


main()