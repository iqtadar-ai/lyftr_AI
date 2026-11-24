import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional


def _resolve_sqlite_path() -> str:
    url = os.getenv("DATABASE_URL", "sqlite:///./app.db").strip()
    if url.startswith("sqlite:///"):
        return url[len("sqlite:///") :]
    return url


def _get_connection() -> sqlite3.Connection:
    db_path = _resolve_sqlite_path()
    parent = os.path.dirname(os.path.abspath(db_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db() -> None:
    """Creates the sessions and messages tables if they do not exist."""
    conn = _get_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                system_prompt TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);"
        )
        conn.commit()
    finally:
        conn.close()


def check_db_health() -> bool:
    """Verifies SQLite read/write availability."""
    try:
        conn = _get_connection()
        try:
            conn.execute("SELECT 1;").fetchone()
            return True
        finally:
            conn.close()
    except Exception:
        return False


def get_session_system_prompt(session_id: str) -> Optional[str]:
    """Returns the stored system_prompt for a session, or None."""
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT system_prompt FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row["system_prompt"] if row else None
    finally:
        conn.close()


def get_session_messages(session_id: str) -> list[dict[str, Any]]:
    """
    Returns chronological messages for a known session.
    Raises KeyError if the session does not exist.
    """
    conn = _get_connection()
    try:
        session_row = conn.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        msg_rows = conn.execute(
            """
            SELECT role, content, metadata, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()

        if session_row is None and not msg_rows:
            raise KeyError("session not found")

        results: list[dict[str, Any]] = []
        for row in msg_rows:
            meta = json.loads(row["metadata"]) if row["metadata"] else None
            results.append(
                {
                    "role": row["role"],
                    "content": row["content"],
                    "metadata": meta,
                    "created_at": row["created_at"],
                }
            )
        return results
    finally:
        conn.close()


def save_turn(
    session_id: str,
    user_message: str,
    assistant_reply: str,
    metadata: dict[str, Any],
    system_prompt: Optional[str] = None,
) -> None:
    """Atomically persists session metadata and a completed user + assistant turn."""
    now = datetime.now(timezone.utc).isoformat()
    conn = _get_connection()
    try:
        conn.execute(
            """
            INSERT INTO sessions (session_id, system_prompt, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                system_prompt = COALESCE(excluded.system_prompt, sessions.system_prompt)
            """,
            (session_id, system_prompt, now),
        )
        conn.execute(
            """
            INSERT INTO messages (session_id, role, content, metadata, created_at)
            VALUES (?, 'user', ?, NULL, ?)
            """,
            (session_id, user_message, now),
        )
        conn.execute(
            """
            INSERT INTO messages (session_id, role, content, metadata, created_at)
            VALUES (?, 'assistant', ?, ?, ?)
            """,
            (session_id, assistant_reply, json.dumps(metadata), now),
        )
        conn.commit()
    finally:
        conn.close()