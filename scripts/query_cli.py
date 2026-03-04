"""
Query the vector DB via API or in-process.
Usage:
  python scripts/query_cli.py "Where is CALCULATE-INTEREST?"
  QUERY_URL=http://localhost:8000 python scripts/query_cli.py "file I/O"
"""
import json
import os
import sys
from pathlib import Path

# Run from repo root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/query_cli.py <query> [top_k]", file=sys.stderr)
        return 1
    query = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    url = os.environ.get("QUERY_URL", "http://localhost:8000")
    if url:
        try:
            import httpx
            r = httpx.post(f"{url.rstrip('/')}/query", json={"query": query, "top_k": top_k}, timeout=30.0)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print("API request failed:", e, file=sys.stderr)
            print("Falling back to in-process search...", file=sys.stderr)
            url = None
    if not url:
        from backend.ingestion.embedder import Embedder
        from backend.ingestion.vector_store import get_vector_store, search
        embedder = Embedder()
        client = get_vector_store()
        vectors = embedder.embed_texts([query])
        hits = search(client, vectors[0], limit=top_k)
        data = {"query": query, "results": [{"id": h["id"], "score": h["score"], **h["payload"]} for h in hits]}
    print(json.dumps(data, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
