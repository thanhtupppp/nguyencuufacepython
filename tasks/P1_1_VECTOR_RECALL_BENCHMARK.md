# P1.1 Vector-search recall benchmark

## Why this task exists

The correctness-first person-level SQL now ranks the best template per `person_id` before `LIMIT`. This fixes the semantic bug where one person with many templates could hide the second person.

A second risk must be measured before tuning HNSW: pgvector approximate indexes can lose recall when `WHERE` filters are applied to the ANN scan. The production query filters `model_version` and optionally active persons.

## Goal

Measure whether person-level top-K remains correct as the gallery grows and as the fraction of eligible templates changes.

## Protocol

For every scenario:

1. Build a deterministic gallery with multiple templates per person.
2. Keep a locked exact-search reference using PostgreSQL exact nearest-neighbor semantics or an equivalent brute-force reference.
3. Run the production person-level query.
4. Compare person IDs, not raw template IDs.
5. Report recall@1, recall@5, recall@10, missing-result rate, p50/p95/p99 latency.
6. Record PostgreSQL/pgvector version, HNSW parameters, `hnsw.ef_search`, `hnsw.iterative_scan`, and table size.

## Scenarios

- 100 people × 1 template
- 100 people × 10 templates
- 1,000 people × 10 templates
- 10,000 people × 10 templates
- mixed template counts (1, 3, 10, 30)
- inactive-person selectivity: 0%, 50%, 90%
- model-version selectivity: one model dominates the table, then balanced models

## Acceptance criteria

- Exact reference is the correctness gate.
- No silent person-level false-negative caused by the approximate index.
- Any recall loss must be visible in the benchmark report and blocked from production acceptance.
- Threshold/margin calibration is not changed to compensate for vector-index recall loss.

## Tuning order

1. correctness-first person-level semantics
2. exact-search baseline
3. HNSW default
4. `ef_search`
5. pgvector >= 0.8 iterative scan (`strict_order` first)
6. only then consider schema/index changes

Use `EXPLAIN (ANALYZE, BUFFERS)` for each production candidate query.

## Related

- `src/database/person_level_search.sql`
- `src/database/person_search.py`
- GitHub issue #2: wire person-level search into `DatabaseClient.search_top_k()`

## Completion rule

This task is DONE only when a reproducible benchmark report shows the production configuration's recall and latency at each required gallery size. No public benchmark number is substituted for project measurements.
