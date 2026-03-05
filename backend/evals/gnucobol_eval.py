import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import httpx


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw = path.read_text(encoding="utf-8")
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _recall_at_k(expected: set[str], ranked_paths: list[str], k: int) -> float:
    if not expected:
        return 0.0
    top = ranked_paths[:k]
    hit = sum(1 for p in expected if p in top)
    return hit / max(1, len(expected))


def _hit_at_k(expected: set[str], ranked_paths: list[str], k: int) -> int:
    top = set(ranked_paths[:k])
    return 1 if expected.intersection(top) else 0


def _mrr(expected: set[str], ranked_paths: list[str]) -> float:
    for i, p in enumerate(ranked_paths, 1):
        if p in expected:
            return 1.0 / i
    return 0.0


def run_retrieval_eval(
    client: httpx.Client,
    dataset: list[dict[str, Any]],
    *,
    top_k: int = 10,
    score_threshold: float = 0.0,
) -> dict[str, Any]:
    recalls: list[float] = []
    hits: list[int] = []
    mrrs: list[float] = []
    latencies_ms: list[float] = []
    failures: list[dict[str, Any]] = []

    for row in dataset:
        query = (row.get("query") or "").strip()
        expected_paths = set(row.get("expected_paths") or [])
        if not query:
            continue
        t0 = time.perf_counter()
        resp = client.post(
            "/query",
            json={
                "query": query,
                "top_k": top_k,
                "score_threshold": score_threshold,
                "source_type": row.get("source_type", "all"),
                "tags": row.get("tags"),
            },
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(elapsed_ms)
        if resp.status_code >= 400:
            failures.append({"query": query, "status": resp.status_code, "detail": resp.text[:200]})
            continue
        data = resp.json()
        ranked_paths = [r.get("file_path", "") for r in (data.get("results") or [])]
        recalls.append(_recall_at_k(expected_paths, ranked_paths, top_k))
        hits.append(_hit_at_k(expected_paths, ranked_paths, top_k))
        mrrs.append(_mrr(expected_paths, ranked_paths))

    def pct(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        arr = sorted(values)
        idx = int(round((p / 100.0) * (len(arr) - 1)))
        return arr[max(0, min(idx, len(arr) - 1))]

    return {
        "queries_total": len(dataset),
        "queries_scored": len(recalls),
        "queries_failed": len(failures),
        "hit_rate_at_k": round(sum(hits) / max(1, len(hits)), 4),
        "recall_at_k": round(sum(recalls) / max(1, len(recalls)), 4),
        "mrr": round(sum(mrrs) / max(1, len(mrrs)), 4),
        "latency_ms_avg": round(sum(latencies_ms) / max(1, len(latencies_ms)), 1),
        "latency_ms_p95": round(pct(latencies_ms, 95), 1),
        "latency_ms_p99": round(pct(latencies_ms, 99), 1),
        "latency_ms_stdev": round(statistics.pstdev(latencies_ms), 1) if len(latencies_ms) > 1 else 0.0,
        "failures": failures[:20],
    }


def run_answer_smoke_eval(
    client: httpx.Client,
    dataset: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Lightweight answer eval:
    - requires all phrases in answer_must_contain
    - rejects if any phrase in answer_must_not_contain appears
    """
    scored = 0
    passed = 0
    failures: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    for row in dataset:
        must = [s.lower() for s in (row.get("answer_must_contain") or []) if str(s).strip()]
        must_not = [s.lower() for s in (row.get("answer_must_not_contain") or []) if str(s).strip()]
        if not must and not must_not:
            continue
        query = (row.get("query") or "").strip()
        if not query:
            continue
        t0 = time.perf_counter()
        resp = client.post("/query/chat", json={"query": query, "top_k": row.get("chat_top_k", 8), "source_type": row.get("source_type", "all")})
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        if resp.status_code >= 400:
            failures.append({"query": query, "status": resp.status_code, "detail": resp.text[:200]})
            continue
        scored += 1
        answer = ((resp.json() or {}).get("answer") or "").lower()
        ok = all(s in answer for s in must) and all(s not in answer for s in must_not)
        if ok:
            passed += 1
        else:
            failures.append({"query": query, "reason": "answer phrase check failed"})
    return {
        "queries_scored": scored,
        "pass_rate": round(passed / max(1, scored), 4),
        "latency_ms_avg": round(sum(latencies_ms) / max(1, len(latencies_ms)), 1),
        "failures": failures[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="GNUCOBOL-oriented eval runner for LegacyLens.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to JSONL eval dataset")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000", help="LegacyLens API base URL")
    parser.add_argument("--top-k", type=int, default=10, help="Retriever top_k for /query")
    parser.add_argument("--score-threshold", type=float, default=0.0, help="Retriever score threshold")
    parser.add_argument("--skip-answer-eval", action="store_true", help="Skip /query/chat smoke checks")
    args = parser.parse_args()

    rows = _load_jsonl(args.dataset)
    if not rows:
        print("No dataset rows found.")
        return 1

    with httpx.Client(base_url=args.base_url, timeout=120.0) as client:
        retrieval = run_retrieval_eval(
            client,
            rows,
            top_k=args.top_k,
            score_threshold=args.score_threshold,
        )
        answer_eval = None if args.skip_answer_eval else run_answer_smoke_eval(client, rows)

    print(json.dumps({"retrieval": retrieval, "answer_smoke": answer_eval}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
