# P1.1 Person-level retrieval benchmark

## Goal

Validate that person-level ranking remains correct and fast as enrollment template counts grow. The benchmark must compare the current correctness-first query against an ANN architecture before any production architecture change.

## Architectures to compare

### A — Current correctness-first query

`face_embeddings` -> filter `model_version`/active -> best template per `person_id` -> rank people -> LIMIT K.

This is the correctness reference. It must never be replaced merely because ANN is faster.

### B — Candidate-first HNSW

HNSW over all templates -> retrieve an oversampled candidate set -> deduplicate by `person_id` -> rerank candidates with exact cosine distance -> top-K people.

Sweep candidate multiplier and `hnsw.ef_search`. Record missed-person cases explicitly.

### C — Per-person prototype/index candidate

Evaluate a separate one-vector-per-person table (prototype/representative embedding) with its own HNSW index. Keep enrollment templates in `face_embeddings` for audit/quality and exact verification. Candidate people are retrieved from the prototype index, then reranked against the person's stored templates.

This architecture is only a candidate until benchmark evidence shows acceptable recall and accuracy. It must not replace A automatically.

## Dataset matrix

- 100 / 1,000 / 10,000 persons.
- 1 / 3 / 10 / 30 templates per person.
- Query identities sampled separately from enrollment templates.
- Active selectivity: 100%, 50%, 10%.
- Multiple `model_version` values with only one version eligible for each query.
- Fixed random seed and persisted benchmark manifest.

## Metrics

### Retrieval correctness

- recall@1, recall@5, recall@10 against Architecture A exact reference.
- missed eligible-person rate.
- duplicate-person rate (must be zero after person-level deduplication).
- top1/top2 identity agreement.

### Performance

- p50/p95/p99 query latency.
- queries/sec where meaningful.
- rows/tuples visited when available from `EXPLAIN (ANALYZE, BUFFERS)`.
- index size and build time for ANN variants.

### Recognition safety

Do not retune recognition thresholds to compensate for retrieval misses. A retrieval miss is a retrieval failure and must remain visible in the report.

## pgvector controls

For HNSW variants benchmark at minimum:

- `hnsw.ef_search`: 40, 80, 160, 320.
- `hnsw.iterative_scan`: off, `strict_order`, `relaxed_order` where supported.
- `hnsw.max_scan_tuples` when iterative scan is enabled.

Use `EXPLAIN (ANALYZE, BUFFERS)` for representative queries.

## Acceptance gate

Do not change production architecture until the benchmark proves:

1. person-level recall meets the project target on every filtering scenario;
2. no duplicate `person_id` appears in final candidates;
3. top1/top2 semantics remain person-level;
4. p95 latency improvement is material enough to justify added complexity;
5. false-match/false-acceptance risk is not increased.

If no ANN design meets the gate, keep Architecture A and optimize PostgreSQL/query/schema instead.
