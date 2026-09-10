from benchmarks.pgvector_filtered_hnsw_benchmark import EXACT_SQL, HNSW_SQL


def test_hnsw_benchmark_has_person_grouping_and_query_exclusion():
    assert "e.id <> %s" in HNSW_SQL
    assert "GROUP BY person_id" in HNSW_SQL
    assert "ORDER BY e.embedding <=> %s::vector ASC" in HNSW_SQL


def test_exact_reference_excludes_query_and_selects_one_template_per_person():
    assert "e.id <> %s" in EXACT_SQL
    assert "PARTITION BY e.person_id" in EXACT_SQL
    assert "ROW_NUMBER() OVER" in EXACT_SQL


def test_iterative_scan_modes_are_explicit_in_benchmark_contract():
    source = open("benchmarks/pgvector_filtered_hnsw_benchmark.py", encoding="utf-8").read()
    assert 'off", "strict_order", "relaxed_order' in source
    assert "hnsw.iterative_scan" in source
    assert "hnsw.max_scan_tuples" in source
