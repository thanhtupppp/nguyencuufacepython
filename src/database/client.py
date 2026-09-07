"""
Database client for PostgreSQL + pgvector vector search,
person identity management, embedding storage, and access logging.
Includes transparent in-memory fallback for testing and development environments.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sqlite3
from typing import Any, Optional, Union
import numpy as np

try:
    import psycopg  # type: ignore
    from pgvector.psycopg import register_vector  # type: ignore
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
    top_candidates: list[SearchCandidate] = field(default_factory=list)


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
        sqlite_path: Optional[Union[str, Path]] = None,
    ):
        self.sqlite_path = Path(sqlite_path) if sqlite_path else None
        self.conn_info = f"host={host} port={port} dbname={dbname} user={user} password={password}"
        self.fallback_to_memory = fallback_to_memory
        self.use_memory = force_memory
        self.use_sqlite = self.sqlite_path is not None
        self._conn: Any = None
        self._sqlite_conn: Optional[sqlite3.Connection] = None

        # In-memory storage / cache for fast vector search
        self._mem_persons: dict[str, dict[str, Any]] = {}
        self._mem_embeddings: list[dict[str, Any]] = []
        self._mem_logs: list[dict[str, Any]] = []
        self._mem_dispense_logs: list[dict[str, Any]] = []

        if self.use_sqlite:
            self._init_sqlite()
        elif not self.use_memory:
            self._connect()

    def _init_sqlite(self) -> None:
        """Initializes local SQLite database and preloads existing records into memory cache."""
        assert self.sqlite_path is not None
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._sqlite_conn = sqlite3.connect(str(self.sqlite_path), check_same_thread=False)
        self._sqlite_conn.execute("PRAGMA foreign_keys = ON;")

        # Create tables
        self._sqlite_conn.executescript("""
        CREATE TABLE IF NOT EXISTS persons (
            person_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT DEFAULT 'default',
            role TEXT DEFAULT 'user',
            metadata TEXT DEFAULT '{}',
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS face_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id TEXT NOT NULL,
            embedding BLOB NOT NULL,
            model_version TEXT DEFAULT 'arcface_v1',
            quality_score REAL DEFAULT 1.0,
            source_image_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (person_id) REFERENCES persons(person_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS access_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id TEXT,
            device_id TEXT,
            similarity REAL,
            margin REAL,
            status TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS dispense_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id TEXT NOT NULL,
            device_id TEXT DEFAULT 'dispenser_01',
            status TEXT NOT NULL,
            similarity REAL DEFAULT 0.0,
            cooldown_seconds_remaining INTEGER DEFAULT 0,
            message TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_dispense_person ON dispense_logs(person_id, timestamp DESC);
        """)
        self._sqlite_conn.commit()

        # Ensure schema migrations for existing SQLite databases
        cur = self._sqlite_conn.cursor()
        cur.execute("PRAGMA table_info(persons);")
        existing_cols = {row[1] for row in cur.fetchall()}
        if "status" not in existing_cols:
            self._sqlite_conn.execute("ALTER TABLE persons ADD COLUMN status TEXT DEFAULT 'active';")
            self._sqlite_conn.execute("UPDATE persons SET status = 'active' WHERE status IS NULL;")
        if "updated_at" not in existing_cols:
            self._sqlite_conn.execute("ALTER TABLE persons ADD COLUMN updated_at TIMESTAMP;")
            self._sqlite_conn.execute("UPDATE persons SET updated_at = CURRENT_TIMESTAMP WHERE updated_at IS NULL;")
        self._sqlite_conn.commit()

        # Load existing persons into cache
        cur.execute("SELECT person_id, name, department, role, metadata, status FROM persons;")
        for row in cur.fetchall():
            meta = json.loads(row[4]) if row[4] else {}
            self._mem_persons[row[0]] = {
                "person_id": row[0],
                "name": row[1],
                "department": row[2],
                "role": row[3],
                "metadata": meta,
                "status": row[5] if len(row) > 5 and row[5] else "active",
            }

        # Load existing embeddings into cache
        cur.execute("SELECT id, person_id, embedding, model_version, quality_score, source_image_path FROM face_embeddings;")
        for row in cur.fetchall():
            emb_arr = np.frombuffer(row[2], dtype=np.float32).copy()
            self._mem_embeddings.append({
                "id": row[0],
                "person_id": row[1],
                "embedding": emb_arr,
                "model_version": row[3],
                "quality_score": float(row[4]) if row[4] is not None else 1.0,
                "source_image_path": row[5],
            })
        logger.info(
            f"SQLite DB initialized at {self.sqlite_path}. "
            f"Loaded {len(self._mem_persons)} persons and {len(self._mem_embeddings)} embeddings into cache."
        )

    def _connect(self) -> None:
        if psycopg is None:
            if self.fallback_to_memory:
                self.use_memory = True
                logger.info("psycopg not installed. Falling back to in-memory mode.")
                return
            raise ImportError("psycopg is required to connect to PostgreSQL")

        try:
            self._conn = psycopg.connect(self.conn_info, autocommit=True)
            if register_vector is not None:
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
        if self.use_memory or self.use_sqlite:
            return

        if sql_file_path:
            with open(sql_file_path, "r", encoding="utf-8") as f:
                ddl = f.read()
            with self._conn.cursor() as cur:
                cur.execute(ddl)

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()
        if self._sqlite_conn:
            self._sqlite_conn.close()

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
        status: str = "active",
    ) -> dict[str, Any]:
        """Registers a new individual identity."""
        metadata = metadata or {}

        if self.use_sqlite and self._sqlite_conn:
            with self._sqlite_conn:
                self._sqlite_conn.execute(
                    """
                    INSERT INTO persons (person_id, name, department, role, metadata, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT (person_id) DO UPDATE SET
                        name = excluded.name,
                        department = excluded.department,
                        role = excluded.role,
                        metadata = excluded.metadata,
                        status = excluded.status,
                        updated_at = CURRENT_TIMESTAMP;
                    """,
                    (person_id, name, department, role, json.dumps(metadata), status),
                )
            record = {
                "person_id": person_id,
                "name": name,
                "department": department,
                "role": role,
                "metadata": metadata,
                "status": status,
            }
            self._mem_persons[person_id] = record
            return record

        if self.use_memory:
            record = {
                "person_id": person_id,
                "name": name,
                "department": department,
                "role": role,
                "metadata": metadata,
                "status": status,
            }
            self._mem_persons[person_id] = record
            return record

        sql = """
        INSERT INTO persons (person_id, name, department, role, metadata, status)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (person_id) DO UPDATE 
        SET name = EXCLUDED.name, department = EXCLUDED.department, role = EXCLUDED.role,
            metadata = EXCLUDED.metadata, status = EXCLUDED.status, updated_at = CURRENT_TIMESTAMP
        RETURNING person_id, name, department, role, metadata, status;
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (person_id, name, department, role, json.dumps(metadata), status))
            row = cur.fetchone()
            return {
                "person_id": row[0],
                "name": row[1],
                "department": row[2],
                "role": row[3],
                "metadata": row[4],
                "status": row[5] if len(row) > 5 else status,
            }

    def update_person_status(self, person_id: str, status: str) -> bool:
        """Updates lifecycle status ('active', 'inactive', 'suspended') of an individual."""
        if self.use_sqlite and self._sqlite_conn:
            with self._sqlite_conn:
                cur = self._sqlite_conn.execute(
                    "UPDATE persons SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE person_id = ?;",
                    (status, person_id),
                )
                if cur.rowcount > 0:
                    if person_id in self._mem_persons:
                        self._mem_persons[person_id]["status"] = status
                    return True
            return False

        if self.use_memory:
            if person_id in self._mem_persons:
                self._mem_persons[person_id]["status"] = status
                return True
            return False

        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE persons SET status = %s, updated_at = CURRENT_TIMESTAMP WHERE person_id = %s;",
                (status, person_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def deactivate_person(self, person_id: str) -> bool:
        """Soft-deactivates an enrolled person so vector search will ignore them."""
        return self.update_person_status(person_id, "inactive")

    def activate_person(self, person_id: str) -> bool:
        """Re-activates a person for recognition."""
        return self.update_person_status(person_id, "active")

    def get_person(self, person_id: str) -> Optional[dict[str, Any]]:
        """Fetches person info by person_id."""
        if self.use_sqlite or self.use_memory:
            return self._mem_persons.get(person_id)

        sql = "SELECT person_id, name, department, role, metadata, status FROM persons WHERE person_id = %s;"
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
                "status": row[5] if len(row) > 5 else "active",
            }

    def list_persons(self) -> list[dict[str, Any]]:
        """Lists all registered identities."""
        if self.use_sqlite or self.use_memory:
            return list(self._mem_persons.values())

        sql = "SELECT person_id, name, department, role, metadata, status FROM persons ORDER BY created_at DESC;"
        with self._conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            return [
                {
                    "person_id": r[0],
                    "name": r[1],
                    "department": r[2],
                    "role": r[3],
                    "metadata": r[4],
                    "status": r[5] if len(r) > 5 else "active",
                }
                for r in rows
            ]

    def delete_person(self, person_id: str) -> bool:
        """Deletes person and cascading embeddings."""
        if self.use_sqlite and self._sqlite_conn:
            with self._sqlite_conn:
                cur = self._sqlite_conn.execute("DELETE FROM persons WHERE person_id = ?;", (person_id,))
                deleted = cur.rowcount > 0
            if person_id in self._mem_persons:
                del self._mem_persons[person_id]
            self._mem_embeddings = [e for e in self._mem_embeddings if e["person_id"] != person_id]
            return deleted

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

        if self.use_sqlite and self._sqlite_conn:
            emb_blob = emb.tobytes()
            with self._sqlite_conn:
                cur = self._sqlite_conn.execute(
                    """
                    INSERT INTO face_embeddings (person_id, embedding, model_version, quality_score, source_image_path)
                    VALUES (?, ?, ?, ?, ?);
                    """,
                    (person_id, emb_blob, model_version, quality_score, source_image_path),
                )
                new_id = cur.lastrowid or (len(self._mem_embeddings) + 1)
            self._mem_embeddings.append({
                "id": new_id,
                "person_id": person_id,
                "embedding": emb,
                "model_version": model_version,
                "quality_score": quality_score,
                "source_image_path": source_image_path,
            })
            return new_id

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
        active_only: bool = True,
    ) -> list[SearchCandidate]:
        """
        Performs vector similarity search.
        Cosine similarity = 1.0 - (embedding <=> query_vector)
        """
        q = np.asarray(query_vector, dtype=np.float32).flatten()
        norm = np.linalg.norm(q)
        if norm > 1e-6:
            q = q / norm

        if self.use_sqlite or self.use_memory:
            matches = []
            for item in self._mem_embeddings:
                if item["model_version"] == model_version:
                    pid = item["person_id"]
                    person_info = self._mem_persons.get(pid, {})
                    if active_only and person_info.get("status", "active") != "active":
                        continue
                    sim = float(np.dot(q, item["embedding"]))
                    p_name = person_info.get("name", pid)
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
        status_filter = "AND p.status = 'active'" if active_only else ""
        sql = f"""
        SELECT 
            e.person_id,
            1.0 - (e.embedding <=> %s::vector) AS similarity,
            p.name
        FROM face_embeddings e
        LEFT JOIN persons p ON e.person_id = p.person_id
        WHERE e.model_version = %s {status_filter}
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

    # Alias for search_top_k
    search_candidates = search_top_k

    def recognize_with_margin(
        self,
        query_vector: np.ndarray,
        model_version: str = "arcface_v1",
        threshold: float = 0.60,
        margin: float = 0.08,
        active_only: bool = True,
    ) -> RecognitionDecision:
        """
        Executes Dual-Threshold anti-false-match decision:
        Match if: Top1 >= threshold AND (Top1 - Top2) >= margin.
        """
        candidates = self.search_top_k(
            query_vector, model_version=model_version, top_k=2, active_only=active_only
        )

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
        if self.use_sqlite and self._sqlite_conn:
            with self._sqlite_conn:
                self._sqlite_conn.execute(
                    """
                    INSERT INTO access_logs (person_id, device_id, similarity, margin, status)
                    VALUES (?, ?, ?, ?, ?);
                    """,
                    (person_id, device_id, similarity, margin, status),
                )
            return

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

    # -------------------------------------------------------------
    # Smart Toilet Paper Dispenser Methods & Anti-Abuse Cooldown
    # -------------------------------------------------------------

    def get_last_successful_dispense(self, person_id: str) -> Optional[datetime]:
        """
        Retrieves the most recent successful dispense timestamp for a person.
        Considers status 'GRANTED' and 'NEW_USER_GRANTED'.
        """
        if self.use_sqlite and self._sqlite_conn:
            cur = self._sqlite_conn.cursor()
            cur.execute(
                """
                SELECT timestamp FROM dispense_logs
                WHERE person_id = ? AND status IN ('GRANTED', 'NEW_USER_GRANTED')
                ORDER BY id DESC LIMIT 1;
                """,
                (person_id,),
            )
            row = cur.fetchone()
            if row and row[0]:
                ts_val = row[0]
                if isinstance(ts_val, datetime):
                    return ts_val
                try:
                    return datetime.fromisoformat(str(ts_val).replace(" ", "T"))
                except Exception:
                    return datetime.strptime(str(ts_val), "%Y-%m-%d %H:%M:%S")
            return None

        if self.use_memory:
            for item in reversed(self._mem_dispense_logs):
                if item["person_id"] == person_id and item["status"] in ("GRANTED", "NEW_USER_GRANTED"):
                    ts = item["timestamp"]
                    if isinstance(ts, datetime):
                        return ts
                    return datetime.fromisoformat(str(ts).replace(" ", "T"))
            return None

        # PostgreSQL
        sql = """
        SELECT timestamp FROM dispense_logs
        WHERE person_id = %s AND status IN ('GRANTED', 'NEW_USER_GRANTED')
        ORDER BY id DESC LIMIT 1;
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (person_id,))
            row = cur.fetchone()
            if row and row[0]:
                return row[0]
        return None

    def log_dispense(
        self,
        person_id: str,
        status: str,
        device_id: str = "dispenser_01",
        similarity: float = 0.0,
        cooldown_seconds_remaining: int = 0,
        message: str = "",
        timestamp: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """
        Records a toilet paper dispense attempt (granted or blocked by cooldown).
        """
        now = timestamp or datetime.now(timezone.utc)
        now_str = now.isoformat()

        record = {
            "person_id": person_id,
            "device_id": device_id,
            "status": status,
            "similarity": similarity,
            "cooldown_seconds_remaining": cooldown_seconds_remaining,
            "message": message,
            "timestamp": now_str,
        }

        if self.use_sqlite and self._sqlite_conn:
            with self._sqlite_conn:
                self._sqlite_conn.execute(
                    """
                    INSERT INTO dispense_logs
                    (person_id, device_id, status, similarity, cooldown_seconds_remaining, message, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    (person_id, device_id, status, similarity, cooldown_seconds_remaining, message, now_str),
                )
            return record

        if self.use_memory:
            self._mem_dispense_logs.append(record)
            return record

        # PostgreSQL
        sql = """
        INSERT INTO dispense_logs
        (person_id, device_id, status, similarity, cooldown_seconds_remaining, message, timestamp)
        VALUES (%s, %s, %s, %s, %s, %s, %s);
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (person_id, device_id, status, similarity, cooldown_seconds_remaining, message, now))
        return record

    def get_dispense_stats(self, device_id: Optional[str] = None) -> dict[str, Any]:
        """
        Returns summary statistics for the smart toilet paper dispenser.
        """
        if self.use_sqlite and self._sqlite_conn:
            cur = self._sqlite_conn.cursor()
            cur.execute("SELECT COUNT(*) FROM dispense_logs WHERE status IN ('GRANTED', 'NEW_USER_GRANTED');")
            total_granted = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM dispense_logs WHERE status = 'COOLDOWN_BLOCKED';")
            total_blocked = cur.fetchone()[0]

            cur.execute("SELECT COUNT(DISTINCT person_id) FROM dispense_logs WHERE status IN ('GRANTED', 'NEW_USER_GRANTED');")
            unique_users = cur.fetchone()[0]

            cur.execute("SELECT timestamp FROM dispense_logs WHERE status IN ('GRANTED', 'NEW_USER_GRANTED') ORDER BY id DESC LIMIT 1;")
            last_row = cur.fetchone()
            last_dispensed_at = last_row[0] if last_row else None

            return {
                "total_granted": total_granted,
                "total_blocked": total_blocked,
                "unique_users": unique_users,
                "last_dispensed_at": last_dispensed_at,
            }

        if self.use_memory:
            granted = [l for l in self._mem_dispense_logs if l["status"] in ("GRANTED", "NEW_USER_GRANTED")]
            blocked = [l for l in self._mem_dispense_logs if l["status"] == "COOLDOWN_BLOCKED"]
            unique = len({l["person_id"] for l in granted})
            return {
                "total_granted": len(granted),
                "total_blocked": len(blocked),
                "unique_users": unique,
                "last_dispensed_at": granted[-1]["timestamp"] if granted else None,
            }

        return {
            "total_granted": 0,
            "total_blocked": 0,
            "unique_users": 0,
            "last_dispensed_at": None,
        }

    def get_dispense_logs(self, limit: int = 50) -> list[dict[str, Any]]:
        """
        Returns the latest dispense logs.
        """
        if self.use_sqlite and self._sqlite_conn:
            cur = self._sqlite_conn.cursor()
            cur.execute(
                """
                SELECT l.id, l.person_id, p.name, l.device_id, l.status, l.similarity,
                       l.cooldown_seconds_remaining, l.message, l.timestamp
                FROM dispense_logs l
                LEFT JOIN persons p ON l.person_id = p.person_id
                ORDER BY l.id DESC LIMIT ?;
                """,
                (limit,),
            )
            rows = cur.fetchall()
            return [
                {
                    "id": r[0],
                    "person_id": r[1],
                    "name": r[2] or r[1],
                    "device_id": r[3],
                    "status": r[4],
                    "similarity": r[5],
                    "cooldown_seconds_remaining": r[6],
                    "message": r[7],
                    "timestamp": r[8],
                }
                for r in rows
            ]

        if self.use_memory:
            return list(reversed(self._mem_dispense_logs[-limit:]))

        return []

    def clear_dispense_logs(self, clear_auto_enrolled_test_users: bool = False) -> int:
        """
        Clears all dispense logs for test reset.
        If clear_auto_enrolled_test_users is True, also deletes auto-enrolled USER_* test accounts.
        """
        deleted_count = 0
        if self.use_sqlite and self._sqlite_conn:
            cur = self._sqlite_conn.cursor()
            cur.execute("SELECT COUNT(*) FROM dispense_logs;")
            deleted_count = cur.fetchone()[0]
            cur.execute("DELETE FROM dispense_logs;")
            if clear_auto_enrolled_test_users:
                cur.execute("DELETE FROM face_embeddings WHERE person_id LIKE 'USER_%';")
                cur.execute("DELETE FROM persons WHERE person_id LIKE 'USER_%';")
            self._sqlite_conn.commit()
            return deleted_count

        if self.use_memory:
            deleted_count = len(self._mem_dispense_logs)
            self._mem_dispense_logs.clear()
            if clear_auto_enrolled_test_users:
                self._mem_persons = {k: v for k, v in self._mem_persons.items() if not k.startswith("USER_")}
                self._mem_embeddings = [e for e in self._mem_embeddings if not e["person_id"].startswith("USER_")]
            return deleted_count

        return 0


