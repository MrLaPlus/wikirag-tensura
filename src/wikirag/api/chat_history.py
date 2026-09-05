import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


class ChatHistoryStore:
    """SQLite persistence for conversations, messages, and non-secret settings."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY, project TEXT NOT NULL, title TEXT NOT NULL,
                    llm_provider TEXT, llm_model TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL, citations_json TEXT NOT NULL DEFAULT '[]',
                    retrieval_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at);
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT NOT NULL, scope TEXT NOT NULL DEFAULT 'global', project TEXT NOT NULL DEFAULT '',
                    value_json TEXT NOT NULL, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(key, scope, project)
                );
                """
            )

    def list_conversations(self, project: str = "tensura", limit: int = 100) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT c.*, COUNT(m.id) AS message_count FROM conversations c
                   LEFT JOIN messages m ON m.conversation_id = c.id
                   WHERE c.project = ? GROUP BY c.id ORDER BY c.updated_at DESC LIMIT ?""",
                (project, max(1, min(limit, 500))),
            ).fetchall()
            return [dict(row) for row in rows]

    def create_conversation(self, project: str, title: str = "New chat", provider: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
        conversation_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute("INSERT INTO conversations(id, project, title, llm_provider, llm_model) VALUES (?, ?, ?, ?, ?)", (conversation_id, project, title[:200] or "New chat", provider, model))
        return self.get_conversation(conversation_id)  # type: ignore[return-value]

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("""SELECT c.*, COUNT(m.id) AS message_count FROM conversations c
                LEFT JOIN messages m ON m.conversation_id = c.id WHERE c.id = ? GROUP BY c.id""", (conversation_id,)).fetchone()
            return dict(row) if row else None

    def get_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at, rowid", (conversation_id,)).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["citations"] = json.loads(item.pop("citations_json") or "[]")
                item["retrieval"] = json.loads(item.pop("retrieval_json") or "[]")
                result.append(item)
            return result

    def add_message(self, conversation_id: str, role: str, content: str, citations: Optional[List[Dict[str, Any]]] = None, retrieval: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        message_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute("INSERT INTO messages(id, conversation_id, role, content, citations_json, retrieval_json) VALUES (?, ?, ?, ?, ?, ?)", (message_id, conversation_id, role, content, json.dumps(citations or [], ensure_ascii=False), json.dumps(retrieval or [], ensure_ascii=False)))
            conn.execute("UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (conversation_id,))
        return {"id": message_id, "conversation_id": conversation_id, "role": role, "content": content, "citations": citations or [], "retrieval": retrieval or []}

    def update_conversation(self, conversation_id: str, title: Optional[str] = None, provider: Optional[str] = None, model: Optional[str] = None) -> Optional[Dict[str, Any]]:
        updates, values = [], []
        for column, value in (("title", title), ("llm_provider", provider), ("llm_model", model)):
            if value is not None:
                updates.append(f"{column} = ?")
                values.append(value[:200] if column == "title" else value)
        if updates:
            values.append(conversation_id)
            with self._connect() as conn:
                conn.execute(f"UPDATE conversations SET {', '.join(updates)}, updated_at = CURRENT_TIMESTAMP WHERE id = ?", values)
        return self.get_conversation(conversation_id)

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._connect() as conn:
            return conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,)).rowcount > 0

    def get_settings(self, project: Optional[str] = None) -> Dict[str, Any]:
        with self._connect() as conn:
            rows = conn.execute("SELECT key, value_json FROM settings WHERE scope = 'global' OR (scope = 'project' AND project = ?)", (project or "",)).fetchall()
            return {row["key"]: json.loads(row["value_json"]) for row in rows}

    def save_settings(self, values: Dict[str, Any], project: Optional[str] = None) -> Dict[str, Any]:
        scope = "project" if project else "global"
        stored_project = project or ""
        with self._connect() as conn:
            for key, value in values.items():
                conn.execute("""INSERT INTO settings(key, scope, project, value_json) VALUES (?, ?, ?, ?)
                    ON CONFLICT(key, scope, project) DO UPDATE SET value_json=excluded.value_json, updated_at=CURRENT_TIMESTAMP""", (key, scope, stored_project, json.dumps(value, ensure_ascii=False)))
        return self.get_settings(project)
