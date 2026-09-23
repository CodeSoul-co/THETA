#!/usr/bin/env python3
"""Download Qwen3-Embedding-0.6B from ModelScope into models/qwen3_embedding_0.6B."""
import json, sys, urllib.request, pathlib

REPO = "Qwen/Qwen3-Embedding-0.6B"
TARGET = pathlib.Path("models/qwen3_embedding_0.6B")
TARGET.mkdir(parents=True, exist_ok=True)
API = f"https://www.modelscope.cn/api/v1/models/{REPO}/repo/files?Revision=master&Recursive=true"

def fetch(url, binary=False):
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read() if binary else response.read().decode("utf-8")

listing = json.loads(fetch(API))
files = listing.get("Data", {}).get("Files", [])
wanted = [f for f in files if f.get("Type") == "blob" and not str(f.get("Path", "")).endswith((".md", ".gitattributes"))]
print("remote files:", len(wanted), flush=True)
for item in wanted:
    name = item["Path"]
    size = int(item.get("Size") or 0)
    target = TARGET / name
    if target.is_file() and target.stat().st_size == size and size > 0:
        print("skip", name, flush=True); continue
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://www.modelscope.cn/api/v1/models/{REPO}/repo?Revision=master&FilePath={urllib.parse.quote(name)}"
    data = fetch(url, binary=True)
    target.write_bytes(data)
    print("saved", name, len(data), flush=True)
print("done", flush=True)
