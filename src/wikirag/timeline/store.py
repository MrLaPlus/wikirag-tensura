import hashlib
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from wikirag.utils.logging import get_logger

logger = get_logger(__name__)


class TimelineStore:
    """Normalized, evidence-linked timeline index derived from existing chunks."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    entity TEXT NOT NULL,
                    event_text TEXT NOT NULL,
                    section_path TEXT NOT NULL DEFAULT '',
                    volume INTEGER,
                    chapter INTEGER,
                    temporal_relation TEXT NOT NULL DEFAULT '',
                    canonical_url TEXT NOT NULL DEFAULT '',
                    source_chunk_id TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL DEFAULT 0.5,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_timeline_entity ON events(entity);
                CREATE INDEX IF NOT EXISTS idx_timeline_volume_chapter ON events(volume, chapter);
                CREATE INDEX IF NOT EXISTS idx_timeline_relation ON events(temporal_relation);
                """
            )

    @staticmethod
    def _number(patterns: List[str], text: str) -> Optional[int]:
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except (TypeError, ValueError):
                    continue
        return None

    @classmethod
    def _normalize_row(cls, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        text = str(row.get("chunk_text") or row.get("raw_text") or "").strip()
        if not text:
            return None
        section = str(row.get("section_path") or "").strip()
        combined = f"{section} {text}"
        volume = cls._number([r"\b(?:volume|vol\.?|เล่ม)\s*([0-9]+)"], combined)
        chapter = cls._number([r"\b(?:chapter|ch\.?|ตอนที่)\s*([0-9]+)"], combined)
        temporal = ""
        if re.search(r"\b(before|prior to|earlier|ก่อน|ก่อนหน้า)\b", combined, re.IGNORECASE):
            temporal = "before"
        elif re.search(r"\b(after|later|following|หลัง|ภายหลัง|ต่อมา)\b", combined, re.IGNORECASE):
            temporal = "after"
        # A timeline row must have a temporal signal; ordinary prose is not an event.
        if volume is None and chapter is None and not temporal:
            return None
        entity = str(row.get("entity") or "").strip()
        if not entity:
            return None
        event_text = re.sub(r"\s+", " ", text)[:500]
        identity = "|".join([entity, section, str(volume), str(chapter), event_text])
        return {
            "event_id": hashlib.sha1(identity.encode("utf-8")).hexdigest(),
            "entity": entity,
            "event_text": event_text,
            "section_path": section,
            "volume": volume,
            "chapter": chapter,
            "temporal_relation": temporal,
            "canonical_url": str(row.get("canonical_url") or ""),
            "source_chunk_id": str(row.get("chunk_id") or ""),
            "confidence": 0.85 if volume is not None or chapter is not None else 0.6,
        }

    def rebuild_from_rows(self, rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
        events: Dict[str, Dict[str, Any]] = {}
        source_rows = 0
        for row in rows:
            source_rows += 1
            item = self._normalize_row(row)
            if item:
                events[item["event_id"]] = item
        with self._connect() as conn:
            conn.execute("DELETE FROM events")
            conn.executemany(
                """INSERT INTO events
                (event_id, entity, event_text, section_path, volume, chapter,
                 temporal_relation, canonical_url, source_chunk_id, confidence)
                VALUES (:event_id, :entity, :event_text, :section_path, :volume, :chapter,
                        :temporal_relation, :canonical_url, :source_chunk_id, :confidence)""",
                list(events.values()),
            )
        return {"source_rows": source_rows, "events": len(events)}

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM events").fetchone()[0])

    def search(self, query: str = "", entity: Optional[str] = None, volume: Optional[int] = None, chapter: Optional[int] = None, relation: Optional[str] = None, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        clauses, params = [], []
        if entity:
            clauses.append("LOWER(entity) LIKE LOWER(?)"); params.append(f"%{entity}%")
        if volume is not None:
            clauses.append("volume = ?"); params.append(volume)
        if chapter is not None:
            clauses.append("chapter = ?"); params.append(chapter)
        if relation:
            clauses.append("temporal_relation = ?"); params.append(relation)
        for term in [t for t in query.split() if t.strip()]:
            like = f"%{term.lower()}%"
            clauses.append("(LOWER(entity) LIKE ? OR LOWER(event_text) LIKE ? OR LOWER(section_path) LIKE ?)")
            params.extend([like, like, like])
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(f"SELECT * FROM events{where} ORDER BY volume IS NULL, volume, chapter IS NULL, chapter, event_id LIMIT ? OFFSET ?", (*params, max(1, min(limit, 500)), max(0, offset))).fetchall()
        return [dict(row) for row in rows]
