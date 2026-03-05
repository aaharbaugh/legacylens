# GNUCOBOL Evals

This folder contains a lightweight evaluation harness for LegacyLens focused on GNUCOBOL retrieval quality and latency.

## Dataset format (JSONL)

Each row supports:

- `query` (required): user question
- `expected_paths` (required for retrieval scoring): list of expected `file_path` values
- `source_type` (optional): `all|code|docs`
- `tags` (optional): retrieval tags filter
- `chat_top_k` (optional): top-k for `/query/chat` in answer smoke checks
- `answer_must_contain` (optional): list of required phrases in answer text
- `answer_must_not_contain` (optional): list of forbidden phrases in answer text

See `datasets/gnucobol_eval_sample.jsonl` for examples.

## Run

Start API first:

`uvicorn backend.api.main:app --reload`

Then run evals:

`python -m backend.evals.gnucobol_eval --dataset backend/evals/datasets/gnucobol_eval_sample.jsonl --base-url http://127.0.0.1:8000`

Skip answer smoke checks if LLM is disabled:

`python -m backend.evals.gnucobol_eval --dataset backend/evals/datasets/gnucobol_eval_sample.jsonl --skip-answer-eval`

## Metrics reported

- Retrieval:
  - `hit_rate_at_k`
  - `recall_at_k`
  - `mrr`
  - average/P95/P99 latency for `/query`
- Answer smoke:
  - phrase-based pass rate for `/query/chat`
  - average answer latency

## Building a stronger GNUCOBOL dataset

Use sources from your indexed corpus and GNUCOBOL test material:

- `tests/cobol85/*` scenarios (NIST-style COBOL85 coverage)
- runtime behavior questions (`libcob/*`)
- compiler parsing/options questions (`cobc/*`)
- docs and release artifacts (`README`, `THANKS`, `NEWS`, `COPYING`)

Aim for at least 200 labeled queries split by category (runtime, parser, docs, syntax, build/config) to track regressions meaningfully.
