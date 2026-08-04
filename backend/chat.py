"""FE-7.1 session history — SQLite-backed chat memory.

Stores per-session messages so the rag chain can pass conversation context
into the prompt, and persists 'Propose Action' recommendations (FE-7.5) for
the Module 5/6 approval workflow.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path

from backend import config


class SessionStore:
    """Thin, dependency-free SQLite store for chat sessions and actions."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = str(db_path or config.DB_FILE)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    db_filter TEXT,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );
                CREATE TABLE IF NOT EXISTS actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    action TEXT,
                    risk_level TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages(session_id, id);
                CREATE INDEX IF NOT EXISTS idx_actions_status
                    ON actions(status);
                """
            )

    # ----- sessions ------------------------------------------------------
    def create_session(self, db_filter: str | None = None) -> str:
        session_id = uuid.uuid4().hex
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, db_filter, created_at) VALUES (?, ?, ?)",
                (session_id, db_filter, time.time()),
            )
        return session_id

    def get_session(self, session_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_sessions(self, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT s.*, COUNT(m.id) AS message_count
                FROM sessions s
                LEFT JOIN messages m ON m.session_id = s.session_id
                GROUP BY s.session_id
                ORDER BY s.created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def add_message(self, session_id: str, role: str, content: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, time.time()),
            )

    def get_messages(self, session_id: str, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT role, content FROM messages
                WHERE session_id = ?
                ORDER BY id DESC LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    # ----- actions (FE-7.5) ----------------------------------------------
    def create_action(
        self,
        session_id: str,
        query: str,
        action: str,
        risk_level: str = "medium",
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO actions (session_id, query, action, risk_level, status, created_at)
                VALUES (?, ?, ?, ?, 'pending', ?)
                """,
                (session_id, query, action, risk_level, time.time()),
            )
            return int(cur.lastrowid)

    def list_actions(self, session_id: str | None = None, status: str | None = None) -> list[dict]:
        sql = "SELECT * FROM actions WHERE 1=1"
        params: list[str] = []
        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)
        if status:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY created_at DESC"
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]