import json
from pathlib import Path

PROC = Path("data/processed")
OUT = Path("data/chunks.jsonl")
MAX_CHARS = 1200
TABLE_MAX = 2500


def split_table(text: str) -> list[str]:
    if len(text) <= TABLE_MAX:
        return [text]
    rows = text.split("\n")
    header, parts, cur = rows[0], [], []
    for row in rows[1:]:
        if sum(len(r) + 1 for r in cur) + len(row) > TABLE_MAX and cur:
            parts.append("\n".join([header, *cur]))
            cur = []
        cur.append(row)
    if cur:
        parts.append("\n".join([header, *cur]))
    return parts


def chunk_file(path: Path) -> list[dict]:
    chunks, buf, buf_key = [], [], None

    def flush():
        nonlocal buf
        if buf:
            t, s = buf_key
            chunks.append({"ticker": t, "type": "text", "section": s, "text": "\n".join(buf)})
            buf = []

    for line in path.open(encoding="utf-8"):
        b = json.loads(line)
        key = (b["ticker"], b["section"])
        if b["type"] == "table":
            flush()
            for part in split_table(b["text"]):
                chunks.append({"ticker": b["ticker"], "type": "table", "section": b["section"], "text": part})
            continue
        if key != buf_key:
            flush()
            buf_key = key
        if buf and sum(len(x) for x in buf) + len(b["text"]) > MAX_CHARS:
            flush()
            buf_key = key
        buf.append(b["text"])
    flush()
    return chunks


def main() -> None:
    all_chunks = []
    for path in sorted(PROC.glob("*.jsonl")):
        all_chunks.extend(chunk_file(path))
    with OUT.open("w", encoding="utf-8") as f:
        for i, c in enumerate(all_chunks):
            c["id"] = i
            f.write(json.dumps(c) + "\n")
    tables = sum(c["type"] == "table" for c in all_chunks)
    print(f"{len(all_chunks)} chunks ({tables} table chunks)")


if __name__ == "__main__":
    main()