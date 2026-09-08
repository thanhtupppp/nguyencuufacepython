# P1.1 Retrieval benchmark runbook

## Goal

Measure retrieval correctness before changing the production person-level pgvector architecture. This benchmark uses deterministic synthetic unit vectors only; it is not a face-recognition accuracy benchmark.

## Current harness

`tools/benchmark_pgvector_retrieval.py` creates a reproducible gallery, computes an exact person-level top-K reference, and measures a candidate-first HNSW-style query with template oversampling followed by person de-duplication.

## Required comparisons

1. Exact person-level reference: best template per person, then rank persons.
2. Current production person-level SQL: best template per person in SQL, then rank persons.
3. Candidate-first HNSW: ANN template candidates, de-duplicate persons, then optional exact rerank.
4. Candidate-first + HNSW iterative scan where supported by the installed pgvector version.

## Matrix

- 100 / 1,000 / 10,000 persons.
- 1 / 3 / 10 / 30 templates per person.
- active selectivity: 100%, 50%, 10%.
- model-version filtering.
- HNSW `ef_search`: 40 / 80 / 160 / 320.
- iterative scan: off / strict_order / relaxed_order when supported.

## Metrics

- recall@1, recall@5, recall@10 against the exact reference.
- missing-result rate.
- person-level duplicate rate (must be zero after de-duplication).
- p50/p95/p99 query latency.
- `EXPLAIN (ANALYZE, BUFFERS)` for representative cases.

## Decision gate

Do not change production architecture unless the candidate architecture preserves the required recall and false-match safety while providing a meaningful latency or scale benefit. Never lower face-recognition threshold/margin to compensate for retrieval misses.

## Next empirical step

Run the harness against the repository's `pgvector/pgvector:pg16` database image, then add the current production person-level SQL and iterative-scan variants to the same measurement script so all comparisons share the same gallery, queries, and exact reference.
