"""
Database client for PostgreSQL + pgvector vector search,
person identity management, embedding storage, and access logging.
Includes transparent in-memory fallback for testing and development environments.
"""

from dataclasses import dataclass
import json
import logging
from typing import Any, Optional, Union
import numpy as np

try:
    import psycopg
    from pgvector.psycopg import register_vector
except ImportError:
    psycopg = None
    register_vector = None

logger = logging.getLogger(__name__)


@dataclass
class SearchCandidate:
    person_id: str
    similarity: float
    name: Optional[str] = None


@dataclass
class RecognitionDecision:
    status: str                         # 'MATCHED', 'AMBIGUOUS_MATCH', 'UNKNOWN'
    person_id: Optional[str] = None
    similarity: float = 0.0
    second_similarity: float = 0.0
    margin: float = 0.0
    top_candidates: list[SearchCandidate] = None

    def __post_init__(self):
        if self.top_candidates is None:
            self.top_candidates = []


class DatabaseClient:
    """
    Client for PostgreSQL with pgvector extension.
    Falls back gracefully to high-performance in-memory vector storage if PostgreSQL is offline.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        dbname: str = "face_recognition",
        user: str = "face_admin",
        password: str = "face_secure_password_2026",
        fallback_to_memory: bool = True,
        force_memory: bool = False,
    ):
        self.conn_info = f"host={host} port={port} dbname={dbname} user={user} password={password}"
        self.fallback_to_memory = fallback_to_memory
        self.use_memory = force_memory
        self._conn = None

        # In-memory storage for fallback/testing
        self._mem_persons: dict[str, dict[str, Any]] = {}
        self._mem_embeddings: list[dict[str, Any]] = []
        self._mem_logs: list[dict[str, Any]] = []

        if not self.use_memory:
            self._connect()

    def _connect(self) -> None:
        if psycopg is None:
            if self.fallback_to_memory:
                self.use_memory = True
                logger.info("psycopg not installed. Falling back to in-memory mode.")
                return
            raise ImportError("psycopg is required to connect to PostgreSQL")

        try:
            self._conn = psycopg.connect(self.conn_info, autocommit=True)
            register_vector(self._conn)
            logger.info("Connected to PostgreSQL + pgvector successfully.")
        except Exception as e:
            if self.fallback_to_memory:
                logger.warning(f"Failed to connect to PostgreSQL ({e}). Operating in in-memory mode.")
                self.use_memory = True
            else:
                raise e

    def init_schema(self, sql_file_path: Optional[str] = None) -> None:
        """Runs DDL initialization scripts."""
        if self.use_memory:
            return

        if sql_file_path:
            with open(sql_file_path, "r", encoding="utf-8") as f:
                ddl = f.read()
            with self._conn.cursor() as cur:
                cur.execute(ddl)

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()

    # -------------------------------------------------------------
    # Identity Management (Persons)
    # -------------------------------------------------------------

    def create_person(
        self,
        person_id: str,
        name: str,
        department: str = "default",
        role: str = "user",
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Registers a new individual identity."""
        metadata = metadata or {}

        if self.use_memory:
            record = {
                "person_id": person_id,
                "name": name,
                "department": department,
                "role": role,
                "metadata": metadata,
            }
            self._mem_persons[person_id] = record
            return record

        sql = """
        INSERT INTO persons (person_id, name, department, role, metadata)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (person_id) DO UPDATE 
        SET name = EXCLUDED.name, department = EXCLUDED.department, role = EXCLUDED.role, metadata = EXCLUDED.metadata
        RETURNING person_id, name, department, role, metadata;
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (person_id, name, department, role, json.dumps(metadata)))
            row = cur.fetchone()
            return {
                "person_id": row[0],
                "name": row[1],
                "department": row[2],
                "role": row[3],
                "metadata": row[4],
            }

    def get_person(self, person_id: str) -> Optional[dict[str, Any]]:
        """Fetches person info by person_id."""
        if self.use_memory:
            return self._mem_persons.get(person_id)

        sql = "SELECT person_id, name, department, role, metadata FROM persons WHERE person_id = %s;"
        with self._conn.cursor() as cur:
            cur.execute(sql, (person_id,))
            row = cur.fetchone()
            if not row:
                return None
            return {
                "person_id": row[0],
                "name": row[1],
                "department": row[2],
                "role": row[3],
                "metadata": row[4],
            }

    def list_persons(self) -> list[dict[str, Any]]:
        """Lists all registered identities."""
        if self.use_memory:
            return list(self._mem_persons.values())

        sql = "SELECT person_id, name, department, role, metadata FROM persons ORDER BY created_at DESC;"
        with self._conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            return [
                {"person_id": r[0], "name": r[1], "department": r[2], "role": r[3], "metadata": r[4]}
                for r in rows
            ]

    def delete_person(self, person_id: str) -> bool:
        """Deletes person and cascading embeddings."""
        if self.use_memory:
            if person_id in self._mem_persons:
                del self._mem_persons[person_id]
                self._mem_embeddings = [e for e in self._mem_embeddings if e["person_id"] != person_id]
                return True
            return False

        sql = "DELETE FROM persons WHERE person_id = %s;"
        with self._conn.cursor() as cur:
            cur.execute(sql, (person_id,))
            return cur.rowcount > 0

    # -------------------------------------------------------------
    # Embedding Operations
    # -------------------------------------------------------------

    def add_embedding(
        self,
        person_id: str,
        embedding: np.ndarray,
        model_version: str = "arcface_v1",
        quality_score: float = 1.0,
        source_image_path: Optional[str] = None,
    ) -> int:
        """Stores a 512D normalized embedding for a person."""
        emb = np.asarray(embedding, dtype=np.float32).flatten()
        norm = np.linalg.norm(emb)
        if norm > 1e-6:
            emb = emb / norm

        if self.use_memory:
            new_id = len(self._mem_embeddings) + 1
            self._mem_embeddings.append({
                "id": new_id,
                "person_id": person_id,
                "embedding": emb,
                "model_version": model_version,
                "quality_score": quality_score,
                "source_image_path": source_image_path,
            })
            return new_id

        sql = """
        INSERT INTO face_embeddings (person_id, embedding, model_version, quality_score, source_image_path)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id;
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (person_id, emb, model_version, quality_score, source_image_path))
            return cur.fetchone()[0]

    def search_top_k(
        self,
        query_vector: np.ndarray,
        model_version: str = "arcface_v1",
        top_k: int = 2,
    ) -> list[SearchCandidate]:
        """
        Performs vector similarity search.
        Cosine similarity = 1.0 - (embedding <=> query_vector)
        """
        q = np.asarray(query_vector, dtype=np.float32).flatten()
        norm = np.linalg.norm(q)
        if norm > 1e-6:
            q = q / norm

        if self.use_memory:
            matches = []
            for item in self._mem_embeddings:
                if item["model_version"] == model_version:
                    sim = float(np.dot(q, item["embedding"]))
                    pid = item["person_id"]
                    p_name = self._mem_persons.get(pid, {}).get("name", pid)
                    matches.append(SearchCandidate(person_id=pid, similarity=sim, name=p_name))

            # Sort descending by similarity
            matches.sort(key=lambda x: x.similarity, reverse=True)
            # Group by person_id to return highest similarity per individual
            seen_pids = set()
            unique_candidates = []
            for m in matches:
                if m.person_id not in seen_pids:
                    seen_pids.add(m.person_id)
                    unique_candidates.append(m)
                if len(unique_candidates) >= top_k:
                    break
            return unique_candidates

        # PostgreSQL HNSW vector search
        sql = """
        SELECT 
            e.person_id,
            1.0 - (e.embedding <=> %s::vector) AS similarity,
            p.name
        FROM face_embeddings e
        LEFT JOIN persons p ON e.person_id = p.person_id
        WHERE e.model_version = %s
        ORDER BY e.embedding <=> %s::vector ASC
        LIMIT %s;
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (q, model_version, q, top_k * 3))
            rows = cur.fetchall()

        # Deduplicate candidates by person_id
        seen_pids = set()
        candidates = []
        for pid, sim, name in rows:
            if pid not in seen_pids:
                seen_pids.add(pid)
                candidates.append(SearchCandidate(person_id=pid, similarity=float(sim), name=name))
            if len(candidates) >= top_k:
                break

        return candidates

    def recognize_with_margin(
        self,
        query_vector: np.ndarray,
        model_version: str = "arcface_v1",
        threshold: float = 0.60,
        margin: float = 0.08,
    ) -> RecognitionDecision:
        """
        Executes Dual-Threshold anti-false-match decision:
        Match if: Top1 >= threshold AND (Top1 - Top2) >= margin.
        """
        candidates = self.search_top_k(query_vector, model_version=model_version, top_k=2)

        if not candidates:
            return RecognitionDecision(
                status="UNKNOWN",
                person_id=None,
                similarity=0.0,
                second_similarity=0.0,
                margin=0.0,
                top_candidates=[],
            )

        top1 = candidates[0]
        top2_sim = candidates[1].similarity if len(candidates) > 1 else -1.0
        calc_margin = top1.similarity - top2_sim

        if top1.similarity < threshold:
            status = "UNKNOWN"
            matched_id = None
        elif calc_margin < margin:
            status = "AMBIGUOUS_MATCH"
            matched_id = None
        else:
            status = "MATCHED"
            matched_id = top1.person_id

        return RecognitionDecision(
            status=status,
            person_id=matched_id,
            similarity=top1.similarity,
            second_similarity=top2_sim,
            margin=calc_margin,
            top_candidates=candidates,
        )

    def log_access(
        self,
        person_id: Optional[str],
        device_id: Optional[str],
        similarity: float,
        margin: float,
        status: str,
    ) -> None:
        """Logs verification/recognition event."""
        if self.use_memory:
            self._mem_logs.append({
                "person_id": person_id,
                "device_id": device_id,
                "similarity": similarity,
                "margin": margin,
                "status": status,
            })
            return

        sql = """
        INSERT INTO access_logs (person_id, device_id, similarity, margin, status)
        VALUES (%s, %s, %s, %s, %s);
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (person_id, device_id, similarity, margin, status))
