"""Fetch the public ParoQuant checkpoints using resumable byte-range files."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="z-lab/Qwen3-4B-PARO")
args = parser.parse_args()
root = Path("models") / args.model.split("/")[-1]
root.mkdir(parents=True, exist_ok=True)
session = requests.Session()
session.trust_env = False
listing = session.get(
    f"https://modelscope.cn/api/v1/models/{args.model}/repo/files",
    params={"Revision": "master", "Recursive": "true"},
    timeout=30,
)
listing.raise_for_status()
files = [
    {"path": f["Path"], "size": f["Size"], "revision": f["Revision"]}
    for f in listing.json()["Data"]["Files"]
    if f["Type"] == "blob" and f["Path"] != ".gitattributes"
]
(root / "source.json").write_text(
    json.dumps({"model": args.model, "provider": "modelscope.cn", "files": files}, indent=2)
)
for file in sorted(files, key=lambda f: f["size"]):
    path = root / file["path"]
    if path.exists() and path.stat().st_size == file["size"]:
        continue
    url = f"https://modelscope.cn/models/{args.model}/resolve/{file['revision']}/{file['path']}"
    print("Downloading", file["path"], file["size"], flush=True)
    if file["size"] <= 64 * 1024 * 1024:
        response = session.get(url, timeout=(30, 120))
        response.raise_for_status()
        path.write_bytes(response.content)
        continue
    part_size = 64 * 1024 * 1024
    starts = list(range(0, file["size"], part_size))

    def get_part(start):
        end = min(start + part_size, file["size"]) - 1
        part = path.with_suffix(path.suffix + f".part-{start}")
        if part.exists() and part.stat().st_size == end - start + 1:
            return part
        transfer = requests.Session()
        transfer.trust_env = False
        response = transfer.get(url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=(30, 120))
        response.raise_for_status()
        if response.status_code != 206:
            raise RuntimeError("Checkpoint server did not accept the requested byte range")
        with part.open("wb") as out:
            for chunk in response.iter_content(4 * 1024 * 1024):
                out.write(chunk)
        if part.stat().st_size != end - start + 1:
            raise RuntimeError(f"Incomplete checkpoint range at {start}")
        return part

    with ThreadPoolExecutor(max_workers=8) as pool:
        parts = list(pool.map(get_part, starts))
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as out:
        for part in parts:
            with part.open("rb") as source:
                while chunk := source.read(4 * 1024 * 1024):
                    out.write(chunk)
            part.unlink()
    temporary.replace(path)
    print("Downloaded", path, flush=True)
