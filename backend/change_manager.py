"""Human-in-the-Loop Change Manager.

Lifecycle of a ChangeProposal:
  pending  →  approved  →  applied
           →  rejected
           →  modified (user edits SQL)  →  approved  →  applied

Storage: SQLite (same ``dbrain.db`` file used by SessionStore / MetricsStore).
The change_manager never applies a change unless the proposal status is
exactly "approved" — every other path is read-only.

Safe changes only:
  • CREATE INDEX CONCURRENTLY
  • ANALYZE / VACUUM ANALYZE (PostgreSQL)
  • ALTER TABLE SET (...) — tablespace params, fill factor
  • db.collection.createIndex(...)  — MongoDB
  • SET GLOBAL / SET SESSION — MySQL session params
  Anything that looks like DROP / DELETE / TRUNCATE is blocked at proposal
  validation time.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from backend import config
from backend.diagnostics.models import ChangeProposal

LOG = logging.getLogger("dbrain.change_manager")

# Allowlist, not blocklist: LLM-written SQL runs with schema-owner rights, so
# only these single-statement shapes may ever be applied.
_ALLOWED_SQL = re.compile(
    r"""(?ix) ^\s*(
        CREATE\s+(UNIQUE\s+)?INDEX\b
      | ANALYZE\b
      | VACUUM\b
      | OPTIMIZE\s+TABLE\b
      | ALTER\s+TABLE\s+\S+\s+SET\s*\(
      | ALTER\s+TABLE\s+\S+\s+ADD\s+(UNIQUE\s+)?(INDEX|KEY)\b
      | SET\s+(GLOBAL|SESSION|PERSIST)\s
    )""",
)
_FORBIDDEN_SQL = re.compile(r"(?i)\b(DROP|DELETE|TRUNCATE|UPDATE|INSERT|GRANT|REVOKE|COPY|PROGRAM)\b")


class ChangeManager:
    """Persist, approve, apply and audit ChangeProposals."""

    def __init__(self) -> None:
        self._db_path = config.DB_FILE
        self._init_db()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_proposal(self, proposal: ChangeProposal) -> str:
        """Persist a new proposal (status=pending). Returns the proposal id."""
        if not proposal.id:
            proposal.id = uuid.uuid4().hex[:12]
        if not self._validate_safety(proposal.sql_or_command, proposal.engine):
            raise ValueError(
                f"Proposed change contains a blocked operation: {proposal.sql_or_command[:100]}"
            )
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO change_proposals
                   (id, engine, sql_or_command, rationale, risk_level,
                    rollback_sql, estimated_improvement, status, session_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    proposal.id, proposal.engine, proposal.sql_or_command,
                    proposal.rationale, proposal.risk_level,
                    proposal.rollback_sql, proposal.estimated_improvement,
                    proposal.status, proposal.session_id,
                    int(time.time()),
                ),
            )
        LOG.info("Change proposal saved: %s | %s", proposal.id, proposal.sql_or_command[:60])
        return proposal.id

    def get_proposal(self, proposal_id: str) -> ChangeProposal | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM change_proposals WHERE id = ?", (proposal_id,)
            ).fetchone()
        return self._row_to_proposal(row) if row else None

    def list_proposals(
        self,
        status: str | None = None,
        session_id: str | None = None,
    ) -> list[ChangeProposal]:
        sql = "SELECT * FROM change_proposals WHERE 1=1"
        params: list = []
        if status:
            sql += " AND status = ?"; params.append(status)
        if session_id:
            sql += " AND session_id = ?"; params.append(session_id)
        sql += " ORDER BY created_at DESC LIMIT 50"
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_proposal(r) for r in rows]

    def approve(self, proposal_id: str) -> ChangeProposal:
        """Mark a proposal as approved (does NOT apply it)."""
        return self._set_status(proposal_id, "approved")

    def reject(self, proposal_id: str, reason: str = "") -> ChangeProposal:
        """Mark a proposal as rejected."""
        proposal = self._set_status(proposal_id, "rejected")
        self._audit(proposal_id, "rejected", reason)
        return proposal

    def apply(self, proposal_id: str) -> dict:
        """Apply an approved change to the live database.

        Only works when proposal.status == "approved".
        Returns {"ok": bool, "message": str, "duration_ms": float}.
        """
        proposal = self.get_proposal(proposal_id)
        if not proposal:
            return {"ok": False, "message": "Proposal not found"}
        if proposal.status != "approved":
            return {
                "ok": False,
                "message": f"Cannot apply: status is '{proposal.status}' (must be 'approved')",
            }

        t0 = time.perf_counter()
        try:
            self._execute_change(proposal)
            duration_ms = round((time.perf_counter() - t0) * 1000, 1)
            self._set_status(proposal_id, "applied")
            self._audit(proposal_id, "applied", f"duration={duration_ms}ms")
            LOG.info("Change applied: %s in %.1fms", proposal_id, duration_ms)
            return {"ok": True, "message": "Change applied successfully", "duration_ms": duration_ms}
        except Exception as exc:  # noqa: BLE001
            duration_ms = round((time.perf_counter() - t0) * 1000, 1)
            self._audit(proposal_id, "apply_failed", str(exc))
            LOG.error("Change apply failed for %s: %s", proposal_id, exc)
            return {"ok": False, "message": str(exc), "duration_ms": duration_ms}

    # ------------------------------------------------------------------
    # Internal: database
    # ------------------------------------------------------------------

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS change_proposals (
                    id                   TEXT PRIMARY KEY,
                    engine               TEXT NOT NULL,
                    sql_or_command       TEXT NOT NULL,
                    rationale            TEXT,
                    risk_level           TEXT DEFAULT 'medium',
                    rollback_sql         TEXT,
                    estimated_improvement TEXT,
                    status               TEXT DEFAULT 'pending',
                    session_id           TEXT,
                    created_at           INTEGER,
                    applied_at           INTEGER
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS change_audit_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    proposal_id TEXT NOT NULL,
                    action      TEXT NOT NULL,
                    detail      TEXT,
                    ts          INTEGER
                )
            """)

    def _set_status(self, proposal_id: str, status: str) -> ChangeProposal:
        extra = {}
        if status == "applied":
            extra["applied_at"] = int(time.time())
        with self._conn() as conn:
            if extra:
                conn.execute(
                    f"UPDATE change_proposals SET status=?, applied_at=? WHERE id=?",
                    (status, extra.get("applied_at"), proposal_id),
                )
            else:
                conn.execute(
                    "UPDATE change_proposals SET status=? WHERE id=?",
                    (status, proposal_id),
                )
        proposal = self.get_proposal(proposal_id)
        if not proposal:
            raise RuntimeError(f"Proposal {proposal_id} not found after status update")
        return proposal

    def _audit(self, proposal_id: str, action: str, detail: str = "") -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO change_audit_log (proposal_id, action, detail, ts) VALUES (?,?,?,?)",
                (proposal_id, action, detail[:500], int(time.time())),
            )

    # ------------------------------------------------------------------
    # Internal: safety + execution
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_safety(sql: str, engine: str = "postgresql") -> bool:
        """True only for one allowlisted statement (or a Mongo createIndex spec)."""
        if engine == "mongodb":
            try:
                cmd = json.loads(sql)
            except ValueError:
                return False
            return (
                isinstance(cmd, dict)
                and set(cmd) <= {"collection", "keys"}
                and isinstance(cmd.get("collection"), str)
                and isinstance(cmd.get("keys"), dict) and bool(cmd["keys"])
            )
        stmt = sql.strip().removesuffix(";")
        if ";" in stmt or "--" in stmt or "/*" in stmt:
            return False  # one statement, no comment tricks
        return bool(_ALLOWED_SQL.match(stmt)) and not _FORBIDDEN_SQL.search(stmt)

    def _execute_change(self, proposal: ChangeProposal) -> None:
        """Execute the change against the appropriate live database."""
        engine = proposal.engine
        sql = proposal.sql_or_command.strip()
        # Re-check at execution time: rows could have been edited in SQLite.
        if not self._validate_safety(sql, engine):
            raise ValueError("Change is outside the safe-operation allowlist")

        if engine == "postgresql":
            if not config.PG_APPLY_DSN:
                raise RuntimeError("PG_APPLY_DSN not configured — cannot apply PostgreSQL changes")
            import psycopg2
            conn = psycopg2.connect(config.PG_APPLY_DSN, connect_timeout=10)
            try:
                # CONCURRENTLY and VACUUM cannot run inside a transaction block
                conn.autocommit = True
                with conn.cursor() as cur:
                    cur.execute(sql)
            finally:
                conn.close()

        elif engine == "mysql":
            if not config.MYSQL_APPLY_DSN:
                raise RuntimeError("MYSQL_APPLY_DSN not configured — cannot apply MySQL changes")
            import pymysql
            m = re.match(
                r"mysql\+pymysql://(?P<user>[^:]+):(?P<pwd>[^@]+)@(?P<host>[^:/]+):?(?P<port>\d+)?/(?P<db>\w+)",
                config.MYSQL_APPLY_DSN,
            )
            if not m:
                raise ValueError("Cannot parse MYSQL_APPLY_DSN")
            conn = pymysql.connect(
                host=m["host"], port=int(m["port"] or 3306),
                user=m["user"], password=m["pwd"], database=m["db"], autocommit=True,
            )
            try:
                with conn.cursor() as cur:
                    cur.execute(sql)
            finally:
                conn.close()

        elif engine == "mongodb":
            if not config.MONGO_APPLY_URI:
                raise RuntimeError("MONGO_APPLY_URI not configured — cannot apply MongoDB changes")
            import pymongo
            # For MongoDB, sql_or_command is a JSON dict: {"collection": "...", "keys": {...}}
            cmd = json.loads(sql) if sql.startswith("{") else {"raw": sql}
            client = pymongo.MongoClient(config.MONGO_APPLY_URI, serverSelectionTimeoutMS=8000)
            try:
                db = client[config.MONGO_DIAG_DB]
                col_name = cmd.get("collection")
                if col_name and cmd.get("keys"):
                    db[col_name].create_index(list(cmd["keys"].items()))
                    LOG.info("MongoDB index created on %s: %s", col_name, cmd["keys"])
                else:
                    raise ValueError(f"Unrecognised MongoDB command structure: {sql[:100]}")
            finally:
                client.close()
        else:
            raise ValueError(f"Unknown engine: {engine}")

    @staticmethod
    def _row_to_proposal(row) -> ChangeProposal:
        return ChangeProposal(
            id=row["id"],
            engine=row["engine"],
            sql_or_command=row["sql_or_command"],
            rationale=row["rationale"] or "",
            risk_level=row["risk_level"] or "medium",
            rollback_sql=row["rollback_sql"],
            estimated_improvement=row["estimated_improvement"] or "",
            status=row["status"],
            session_id=row["session_id"],
        )
