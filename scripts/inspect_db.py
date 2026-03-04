"""
List what's in the vector DB: count and optionally scroll through chunks.
Usage:
  python scripts/inspect_db.py
  python scripts/inspect_db.py --limit 10
  python scripts/inspect_db.py --file sample
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.ingestion.vector_store import count, get_client, list_chunks


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect LegacyLens vector DB")
    parser.add_argument("--limit", type=int, default=20, help="Max chunks to list (default 20)")
    parser.add_argument("--file", type=str, default=None, help="Filter by substring in file path")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--tags", action="store_true", help="Count unique tags in DB")
    args = parser.parse_args()

    client = get_client()
    n = count(client)
    print(f"Total chunks in DB: {n}")
    if n == 0:
        return 0

    if args.tags:
        all_tags: set[str] = set()
        offset = None
        while True:
            chunks, next_offset = list_chunks(client, limit=500, offset=offset)
            for c in chunks:
                for t in c.get("tags") or []:
                    all_tags.add(t)
            if next_offset is None:
                break
            offset = next_offset
        tags_sorted = sorted(all_tags)
        print(f"Unique tags: {len(tags_sorted)}")
        for t in tags_sorted:
            print(f"  {t}")
        return 0

    chunks, next_offset = list_chunks(client, limit=args.limit, file_path_contains=args.file)
    if args.json:
        print(json.dumps({"total": n, "chunks": chunks, "next_offset": next_offset}, indent=2))
        return 0

    for i, c in enumerate(chunks, 1):
        path = c.get("file_path", "?")
        start = c.get("start_line", "?")
        end = c.get("end_line", "?")
        para = c.get("paragraph_name") or "(no paragraph)"
        snippet = (c.get("code_snippet") or "")[:80].replace("\n", " ")
        print(f"  {i}. {path} L{start}-{end} | {para}")
        print(f"     {snippet}...")
    if next_offset:
        print(f"  ... more (use --limit {args.limit + len(chunks)} or --json to paginate)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
