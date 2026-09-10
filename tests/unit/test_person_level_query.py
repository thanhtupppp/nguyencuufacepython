from src.database.person_level_query import build_person_level_search_sql


def test_person_level_query_groups_before_limit():
    sql = build_person_level_search_sql(active_only=True)
    assert "PARTITION BY e.person_id" in sql
    assert "WHERE rn = 1" in sql
    assert "ORDER BY similarity DESC" in sql
    assert "LIMIT %s" in sql
    assert "p.status = 'active'" in sql


def test_person_level_query_can_include_inactive_people():
    sql = build_person_level_search_sql(active_only=False)
    assert "p.status = 'active'" not in sql
