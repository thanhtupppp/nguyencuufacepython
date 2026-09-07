"""
Unit tests for DatabaseClient and Vector Search operations.
"""

import numpy as np
import pytest
from src.database.client import DatabaseClient, RecognitionDecision


@pytest.fixture
def db_client():
    """Provides an in-memory DatabaseClient instance for deterministic testing."""
    return DatabaseClient(force_memory=True)


def test_person_crud(db_client):
    # 1. Create
    p = db_client.create_person("user_001", "Nguyen Van A", department="AI Research", metadata={"level": 5})
    assert p["person_id"] == "user_001"
    assert p["name"] == "Nguyen Van A"

    # 2. Get
    fetched = db_client.get_person("user_001")
    assert fetched is not None
    assert fetched["name"] == "Nguyen Van A"
    assert fetched["metadata"]["level"] == 5

    # 3. List
    all_p = db_client.list_persons()
    assert len(all_p) == 1

    # 4. Delete
    deleted = db_client.delete_person("user_001")
    assert deleted is True
    assert db_client.get_person("user_001") is None


def test_vector_similarity_search(db_client):
    # Setup 2 users with known orthogonal vectors
    db_client.create_person("user_A", "Alice")
    db_client.create_person("user_B", "Bob")

    vec_a = np.zeros(512, dtype=np.float32)
    vec_a[0] = 1.0  # Unit vector along axis 0

    vec_b = np.zeros(512, dtype=np.float32)
    vec_b[1] = 1.0  # Unit vector along axis 1

    db_client.add_embedding("user_A", vec_a, model_version="arcface_v1")
    db_client.add_embedding("user_B", vec_b, model_version="arcface_v1")

    # Query vector very close to user_A
    query = np.zeros(512, dtype=np.float32)
    query[0] = 0.99
    query[1] = 0.05
    query /= np.linalg.norm(query)

    candidates = db_client.search_top_k(query, model_version="arcface_v1", top_k=2)
    assert len(candidates) == 2
    assert candidates[0].person_id == "user_A"
    assert candidates[0].similarity > 0.95
    assert candidates[1].person_id == "user_B"
    assert candidates[1].similarity < 0.10


def test_recognize_with_margin(db_client):
    db_client.create_person("user_A", "Alice")
    db_client.create_person("user_B", "Bob")

    vec_a = np.zeros(512, dtype=np.float32)
    vec_a[0] = 1.0
    vec_b = np.zeros(512, dtype=np.float32)
    vec_b[1] = 1.0

    db_client.add_embedding("user_A", vec_a, model_version="arcface_v1")
    db_client.add_embedding("user_B", vec_b, model_version="arcface_v1")

    # 1. Clear Match: close to Alice
    query_alice = np.zeros(512, dtype=np.float32)
    query_alice[0] = 1.0
    dec1 = db_client.recognize_with_margin(query_alice, threshold=0.60, margin=0.08)
    assert dec1.status == "MATCHED"
    assert dec1.person_id == "user_A"
    assert dec1.margin > 0.8

    # 2. Ambiguous Match: 45 degree vector between Alice and Bob
    query_split = np.zeros(512, dtype=np.float32)
    query_split[0] = 0.7071
    query_split[1] = 0.7071
    dec2 = db_client.recognize_with_margin(query_split, threshold=0.60, margin=0.08)
    # Scores: S1 ~ 0.7071, S2 ~ 0.7071 -> margin ~ 0.0 < 0.08 -> Ambiguous!
    assert dec2.status == "AMBIGUOUS_MATCH"
    assert dec2.person_id is None

    # 3. Unknown: orthogonal to both
    query_unknown = np.zeros(512, dtype=np.float32)
    query_unknown[2] = 1.0
    dec3 = db_client.recognize_with_margin(query_unknown, threshold=0.60, margin=0.08)
    assert dec3.status == "UNKNOWN"
    assert dec3.person_id is None


def test_sqlite_persistence(tmp_path):
    """Verify SQLite persists persons, embeddings, and access logs across sessions."""
    db_file = tmp_path / "test_faces.db"

    # Session 1: Create client, enroll person and embedding
    client1 = DatabaseClient(sqlite_path=db_file)
    client1.create_person("user_001", "Charlie", department="Security", metadata={"badge": "A1"})
    vec = np.zeros(512, dtype=np.float32)
    vec[0] = 1.0
    client1.add_embedding("user_001", vec, model_version="arcface_v1")
    client1.log_access("user_001", "cam_01", 0.95, 0.20, "MATCHED")
    client1.close()

    # Session 2: Reopen SQLite DB from same path
    client2 = DatabaseClient(sqlite_path=db_file)
    p = client2.get_person("user_001")
    assert p is not None
    assert p["name"] == "Charlie"
    assert p["metadata"]["badge"] == "A1"

    # Verify vector search retrieves the persisted identity
    query = np.zeros(512, dtype=np.float32)
    query[0] = 0.98
    query[1] = 0.05
    query /= np.linalg.norm(query)
    candidates = client2.search_top_k(query, top_k=1)
    assert len(candidates) == 1
    assert candidates[0].person_id == "user_001"
    assert candidates[0].similarity > 0.95

    # Test delete cascade
    assert client2.delete_person("user_001") is True
    assert client2.get_person("user_001") is None
    assert len(client2.search_top_k(query, top_k=1)) == 0
    client2.close()


def test_person_status_and_active_scoping(db_client):
    """Verify person status scoping and active_only filter for vector searches."""
    db_client.create_person("emp_active", "Active User")
    db_client.create_person("emp_resigned", "Resigned User")

    vec_a = np.zeros(512, dtype=np.float32)
    vec_a[0] = 1.0
    vec_b = np.zeros(512, dtype=np.float32)
    vec_b[1] = 1.0

    db_client.add_embedding("emp_active", vec_a, model_version="arcface_v1")
    db_client.add_embedding("emp_resigned", vec_b, model_version="arcface_v1")

    # Deactivate emp_resigned
    res = db_client.deactivate_person("emp_resigned")
    assert res is True

    resigned_data = db_client.get_person("emp_resigned")
    assert resigned_data["status"] == "inactive"

    # Query matching emp_resigned
    query_b = vec_b.copy()

    # With active_only=True, emp_resigned MUST NOT be returned
    candidates_active = db_client.search_top_k(query_b, active_only=True)
    assert not any(c.person_id == "emp_resigned" for c in candidates_active)

    decision_active = db_client.recognize_with_margin(query_b, active_only=True)
    assert decision_active.person_id != "emp_resigned"

    # With active_only=False, emp_resigned is returned
    candidates_all = db_client.search_top_k(query_b, active_only=False)
    assert any(c.person_id == "emp_resigned" for c in candidates_all)
    assert candidates_all[0].person_id == "emp_resigned"

    # Reactivate emp_resigned
    db_client.activate_person("emp_resigned")
    candidates_after_reactivate = db_client.search_top_k(query_b, active_only=True)
    assert candidates_after_reactivate[0].person_id == "emp_resigned"


