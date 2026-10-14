import json
import time
from pathlib import Path

import requests

HEADERS = {"User-Agent": "Varshitha Chappidi your.email@example.com"}
TICKERS = ["JPM", "BAC", "MS", "GS", "C"]
OUT = Path("data/raw")


def get_cik_map() -> dict:
    r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=HEADERS, timeout=30)
    r.raise_for_status()
    return {v["ticker"]: str(v["cik_str"]).zfill(10) for v in r.json().values()}


def latest_10k(cik: str) -> tuple[str, str]:
    r = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=HEADERS, timeout=30)
    r.raise_for_status()
    recent = r.json()["filings"]["recent"]
    for form, acc, doc in zip(recent["form"], recent["accessionNumber"], recent["primaryDocument"]):
        if form == "10-K":
            return acc, doc
    raise ValueError(f"No 10-K found for CIK {cik}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ciks = get_cik_map()
    meta = {}
    for t in TICKERS:
        cik = ciks[t]
        acc, doc = latest_10k(cik)
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{doc}"
        html = requests.get(url, headers=HEADERS, timeout=60)
        html.raise_for_status()
        (OUT / f"{t}.html").write_text(html.text, encoding="utf-8")
        meta[t] = {"cik": cik, "accession": acc, "url": url}
        print(f"saved {t}: {url}")
        time.sleep(0.2)
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()