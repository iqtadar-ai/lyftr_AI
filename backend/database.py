import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def resolve_db_path(database_url: str | None = None) -> str:
    """
    Converts a DATABASE_URL string into a local filesystem path for sqlite3.
    Examples:
      - 'sqlite:///./app.db'    -> './app.db'      (relative path, local dev)
      - 'sqlite:////data/app.db' -> '/data/app.db'  (absolute path, Docker volume)
    """
    url = database_url or os.getenv("DATABASE_URL", "sqlite:///./app.db")

    if url.startswith("sqlite:///"):
        # Strip the exact 10-char prefix 'sqlite:///'
        # If URL had 4 slashes ('sqlite:////data/app.db'), stripping 3 leaves '/data/app.db'.
        path = url[len("sqlite:///"):]
    else:
        path = url

    if path != ":memory:":
        parent = Path(path).parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)

    return path


def get_connection(database_url: str | None = None) -> sqlite3.Connection:
    """Opens a SQLite connection configured with dict-like rows and WAL mode."""
    db_path = resolve_db_path(database_url)
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db(database_url: str | None = None) -> None:
    """Creates the messages table and session index on startup if they don't exist."""
    with get_connection(database_url) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_session_id
            ON messages (session_id, id);
            """
        )


def check_db_health(database_url: str | None = None) -> bool:
    """Verifies SQLite is reachable (used by GET /health)."""
    try:
        with get_connection(database_url) as conn:
            conn.execute("SELECT 1;").fetchone()
        return True
    except sqlite3.Error:
        return False


def get_session_messages(
    session_id: str, database_url: str | None = None
) -> list[dict[str, Any]]:
    """
    Fetches all messages for a session in chronological order (oldest first).
    Returns an empty list if the session does not exist.
    """
    with get_connection(database_url) as conn:
        rows = conn.execute(
            """
            SELECT role, content, metadata
            FROM messages
            WHERE session_id = ?
            ORDER BY id ASC;
            """,
            (session_id,),
        ).fetchall()

    messages: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {
            "role": row["role"],
            "content": row["content"],
        }
        if row["role"] == "assistant" and row["metadata"]:
            item["metadata"] = json.loads(row["metadata"])
        messages.append(item)

    return messages
def _resolve_sqlite_path() -> str:
    url = os.getenv("DATABASE_URL", "sqlite:///./app.db").strip()
    if url.startswith("sqlite:///"):
        return url[len("sqlite:///") :]
    return url


def list_sessions(limit: int = 50) -> list[dict[str, Any]]:
    """Returns distinct chat sessions ordered by most recent message."""
    conn = sqlite3.connect(_resolve_sqlite_path())
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT
                m.session_id,
                COALESCE(
                    (
                        SELECT content
                        FROM messages u
                        WHERE u.session_id = m.session_id AND u.role = 'user'
                        ORDER BY u.id ASC
                        LIMIT 1
                    ),
                    'New chat'
                ) AS title,
                MAX(m.created_at) AS updated_at,
                MAX(m.id) AS last_id
            FROM messages m
            GROUP BY m.session_id
            ORDER BY last_id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [
            {
                "session_id": row["session_id"],
                "title": row["title"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]
    finally:
        conn.close()


def delete_session(session_id: str) -> None:
    """Deletes all messages belonging to a session."""
    conn = sqlite3.connect(_resolve_sqlite_path())
    try:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.commit()
    finally:
        conn.close()

def save_turn(
    session_id: str,
    user_message: str,
    assistant_reply: str,
    metadata: dict[str, Any],
    database_url: str | None = None,
) -> None:
    """
    Atomically persists one completed conversation turn (user + assistant).
    Called ONLY after a successful LLM or cache response so failed requests
    (422/503) never pollute session history.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    metadata_json = json.dumps(metadata)

    with get_connection(database_url) as conn:
        conn.execute(
            """
            INSERT INTO messages (session_id, role, content, metadata, created_at)
            VALUES (?, 'user', ?, NULL, ?);
            """,
            (session_id, user_message, now_iso),
        )
        conn.execute(
            """
            INSERT INTO messages (session_id, role, content, metadata, created_at)
            VALUES (?, 'assistant', ?, ?, ?);
            """,
            (session_id, assistant_reply, metadata_json, now_iso),
        )