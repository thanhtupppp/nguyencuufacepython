from benchmarks.pgvector_leave_one_out_benchmark import EXACT_SQL, HNSW_SQL


def test_benchmark_excludes_query_embedding_from_exact_gallery():
    assert "AND e.id <> %s" in EXACT_SQL
    assert "PARTITION BY e.person_id" in EXACT_SQL
    assert "WHERE rn = 1" in EXACT_SQL


def test_benchmark_excludes_query_embedding_from_hnsw_gallery():
    assert "AND e.id <> %s" in HNSW_SQL
    assert "ORDER BY e.embedding <=> %s::vector ASC" in HNSW_SQL
    assert "GROUP BY person_id" in HNSW_SQL
