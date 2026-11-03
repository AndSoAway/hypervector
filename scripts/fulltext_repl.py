#!/usr/bin/env python3
"""Interactive full-text search REPL.

Type a query, press Enter, see BM25 hits from a running HyperVec HTTP server.
Uses only the stdlib (urllib) so it runs anywhere, no pyhypervec needed.

Usage:
    python3 scripts/fulltext_repl.py [--url URL] [--collection NAME] [--limit N]

Defaults: --url http://127.0.0.1:8099  --collection demo  --limit 5
Commands inside the REPL:  :limit N  (change limit)   :quit / Ctrl-D  (exit)
"""
import argparse
import json
import sys
import urllib.request
import urllib.error


def search(url: str, collection: str, query: str, limit: int) -> list[dict]:
    body = json.dumps(
        {"query": query, "limit": limit, "output_fields": ["contents"]}
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{url}/collections/{collection}/search-text",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read()).get("results", [])


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive full-text search REPL.")
    ap.add_argument("--url", default="http://127.0.0.1:8099")
    ap.add_argument("--collection", default="demo")
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()
    limit = args.limit

    print(f"连接 {args.url}  collection={args.collection}  limit={limit}")
    print("输入中文 query 回车检索；:limit N 改数量；:quit 或 Ctrl-D 退出\n")

    while True:
        try:
            q = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q in (":quit", ":q", ":exit"):
            break
        if q.startswith(":limit"):
            parts = q.split()
            if len(parts) == 2 and parts[1].isdigit():
                limit = int(parts[1])
                print(f"  limit = {limit}")
            else:
                print("  用法: :limit N")
            continue

        try:
            hits = search(args.url, args.collection, q, limit)
        except urllib.error.HTTPError as e:
            print(f"  [HTTP {e.code}] {e.read().decode('utf-8', 'ignore')[:200]}")
            continue
        except Exception as e:  # noqa: BLE001
            print(f"  [错误] {e}")
            continue

        if not hits:
            print("  (无命中)")
            continue
        for i, h in enumerate(hits, 1):
            text = (h.get("entity") or {}).get("contents", "")
            if len(text) > 60:
                text = text[:60] + "…"
            print(f"  {i}. id={h['id']}  BM25={h['distance']:.3f}  {text}")
        print()


if __name__ == "__main__":
    main()
