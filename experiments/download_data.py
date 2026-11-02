"""Download public held-out WikiText-2 and GSM8K through the dataset API."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def fetch(url, params=None):
    session = requests.Session()
    session.trust_env = False
    response = session.get(url, params=params, timeout=(20, 90))
    response.raise_for_status()
    return response.json()


root = Path("data")
root.mkdir(exist_ok=True)
sources = []
for dataset, config, filename in (
    ("Salesforce/wikitext", "wikitext-2-raw-v1", "wikitext2_test.jsonl"),
    ("openai/gsm8k", "main", "gsm8k_test.jsonl"),
):
    revision = fetch(f"https://huggingface.co/api/datasets/{dataset}")["sha"]
    params = {"dataset": dataset, "config": config, "split": "test", "length": 100}
    url = "https://datasets-server.huggingface.co/rows"
    first = fetch(url, dict(params, offset=0))
    count = first["num_rows_total"]

    def batch(offset):
        return fetch(url, dict(params, offset=offset))["rows"]

    rows = first["rows"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for chunk in pool.map(batch, range(100, count, 100)):
            rows.extend(chunk)
    with (root / filename).open("w") as out:
        for row in rows:
            out.write(json.dumps(row["row"], ensure_ascii=False) + "\n")
    sources.append(
        {
            "dataset": dataset,
            "config": config,
            "split": "test",
            "revision": revision,
            "rows": len(rows),
            "file": filename,
            "endpoint": url,
        }
    )
    (root / "sources.json").write_text(json.dumps(sources, indent=2))
    print(filename, len(rows), flush=True)
