import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

RAW = Path("data/raw")
OUT = Path("data/processed")


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()


def table_to_text(tbl) -> str:
    rows = []
    for tr in tbl.find_all("tr"):
        cells = [clean(c.get_text(" ")) for c in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c and c not in {"$", ")", "%", "("}]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def is_heading(el, text: str) -> bool:
    if not 3 <= len(text) <= 120 or text.endswith("."):
        return False
    for node in [el, *el.find_all(True)]:
        style = node.get("style", "").lower().replace(" ", "")
        if "font-weight:700" in style or "font-weight:bold" in style:
            if len(clean(node.get_text(" "))) >= 0.9 * len(text):
                return True
    return False


def parse_filing(path: Path) -> list[dict]:
    ticker = path.stem
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "lxml")
    for h in soup.find_all("ix:header"):
        h.decompose()

    blocks, section = [], "Front matter"
    for el in soup.find_all(["div", "p", "table"]):
        if el.name == "table":
            if el.find_parent("table"):
                continue
            text = table_to_text(el)
            if text.count("\n") >= 1 and re.search(r"\d", text):
                blocks.append({"ticker": ticker, "type": "table", "section": section, "text": text})
            continue
        if el.find_parent("table") or el.find(["div", "p", "table"]):
            continue
        text = clean(el.get_text(" "))
        if len(text) < 3 or re.fullmatch(r"\d{1,3}", text):
            continue
        if is_heading(el, text):
            section = text
        elif len(text) > 40:
            blocks.append({"ticker": ticker, "type": "text", "section": section, "text": text})
    return blocks


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in sorted(RAW.glob("*.html")):
        blocks = parse_filing(path)
        with (OUT / f"{path.stem}.jsonl").open("w", encoding="utf-8") as f:
            for b in blocks:
                f.write(json.dumps(b) + "\n")
        n_tables = sum(b["type"] == "table" for b in blocks)
        print(f"{path.stem}: {len(blocks)} blocks ({n_tables} tables)")


if __name__ == "__main__":
    main()