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
