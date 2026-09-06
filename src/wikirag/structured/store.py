import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from wikirag.utils.logging import get_logger

logger = get_logger(__name__)


class StructuredKnowledgeStore:
    """Small normalized SQLite index derived from existing LanceDB metadata.

    This index deliberately stores facts separately from vectors. Rebuilding it
    never changes embeddings and does not require a model to be loaded.
    """

    FIELDS = ("species", "rank", "status", "affiliation", "skills", "equipment", "evolution")

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
                CREATE TABLE IF NOT EXISTS entities (
                    entity TEXT PRIMARY KEY,
                    canonical_url TEXT NOT NULL DEFAULT '',
                    entity_type TEXT NOT NULL DEFAULT 'character',
                    aliases_json TEXT NOT NULL DEFAULT '[]',
                    categories_json TEXT NOT NULL DEFAULT '[]',
                    species TEXT NOT NULL DEFAULT '',
                    rank TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT '',
                    affiliation_json TEXT NOT NULL DEFAULT '[]',
                    skills_json TEXT NOT NULL DEFAULT '[]',
                    equipment_json TEXT NOT NULL DEFAULT '[]',
                    evolution TEXT NOT NULL DEFAULT '',
                    source_count INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_structured_type ON entities(entity_type);
                CREATE INDEX IF NOT EXISTS idx_structured_species ON entities(species);
                CREATE INDEX IF NOT EXISTS idx_structured_rank ON entities(rank);
                CREATE INDEX IF NOT EXISTS idx_structured_status ON entities(status);
                """
            )

    @staticmethod
    def _values(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except json.JSONDecodeError:
                pass
        return [part.strip() for part in re.split(r"[,;\n•|]", text) if part.strip()]

    @classmethod
    def _field(cls, raw: Dict[str, Any], *names: str) -> str:
        lowered = {str(k).lower().replace(" ", "_"): v for k, v in raw.items()}
        for name in names:
            value = lowered.get(name)
            if value:
                return str(value).strip()
        return ""

    @classmethod
    def _field_list(cls, raw: Dict[str, Any], *names: str) -> List[str]:
        lowered = {str(k).lower().replace(" ", "_"): v for k, v in raw.items()}
        for name in names:
            if lowered.get(name):
                return cls._values(lowered[name])
        return []

    @staticmethod
    def _entity_type(categories: Iterable[str], raw: Dict[str, Any]) -> str:
        category_values = [str(c).strip().lower() for c in categories]
        text = " ".join([*category_values, *(str(k) for k in raw)]).lower()
        # Character pages can also carry race/skill categories. Prefer the
        # explicit Characters category so the primary entity type is stable.
        if any("character" in category for category in category_values):
            return "character"
        if any(word in text for word in ("skill", "magic", "ability", "เวทมนตร์", "สกิล")):
            return "skill"
        if any(word in text for word in ("weapon", "equipment", "item", "อาวุธ", "อุปกรณ์")):
            return "equipment"
        if any(word in text for word in ("race", "species", "เผ่าพันธุ์")):
            return "race"
        if any(word in text for word in ("location", "สถานที่")):
            return "location"
        if any(word in text for word in ("organization", "group", "องค์กร")):
            return "organization"
        return "character"

    @classmethod
    def _normalize_row(cls, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        entity = str(row.get("entity") or "").strip()
        if not entity:
            return None
        try:
            infobox = json.loads(row.get("infobox_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            infobox = {}
        if not isinstance(infobox, dict):
            infobox = {}
        categories = cls._values(row.get("categories_json"))
        aliases = cls._values(row.get("aliases_json"))
        skills = cls._field_list(infobox, "skills", "skill", "abilities", "ability", "magic", "spells")
        equipment = cls._field_list(infobox, "weapons", "weapon", "equipment", "items", "armaments")
        affiliation = cls._field_list(infobox, "affiliation", "faffiliation", "organization", "group")
        return {
            "entity": entity,
            "canonical_url": str(row.get("canonical_url") or ""),
            "entity_type": cls._entity_type(categories, infobox),
            "aliases": aliases,
            "categories": categories,
            "species": cls._field(infobox, "species", "race"),
            "rank": cls._field(infobox, "rank", "drank", "arank"),
            "status": cls._field(infobox, "status"),
            "affiliation": affiliation,
            "skills": skills,
            "equipment": equipment,
            "evolution": cls._field(infobox, "evolution", "evolves_to", "evolved_form"),
        }

    def rebuild_from_rows(self, rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
        """Rebuild from metadata rows, atomically replacing only this index."""
        normalized: Dict[str, Dict[str, Any]] = {}
        source_rows = 0
        for row in rows:
            source_rows += 1
            item = self._normalize_row(row)
            if item:
                # Prefer the row with the richest metadata for each entity.
                old = normalized.get(item["entity"])
                if old is None or sum(bool(item[key]) for key in self.FIELDS) > sum(bool(old[key]) for key in self.FIELDS):
                    normalized[item["entity"]] = item

        with self._connect() as conn:
            conn.execute("DELETE FROM entities")
            for item in normalized.values():
                conn.execute(
                    """INSERT INTO entities
                    (entity, canonical_url, entity_type, aliases_json, categories_json,
                     species, rank, status, affiliation_json, skills_json, equipment_json,
                     evolution, source_count, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                    (item["entity"], item["canonical_url"], item["entity_type"],
                     json.dumps(item["aliases"], ensure_ascii=False), json.dumps(item["categories"], ensure_ascii=False),
                     item["species"], item["rank"], item["status"],
                     json.dumps(item["affiliation"], ensure_ascii=False), json.dumps(item["skills"], ensure_ascii=False),
                     json.dumps(item["equipment"], ensure_ascii=False), item["evolution"], 1),
                )
        return {"source_rows": source_rows, "entities": len(normalized)}

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0])

    def get(self, entity: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM entities WHERE entity = ?", (entity,)).fetchone()
        if not row:
            return None
        item = dict(row)
        for key in ("aliases_json", "categories_json", "affiliation_json", "skills_json", "equipment_json"):
            item[key[:-5]] = json.loads(item.pop(key) or "[]")
        return item

    def search(self, query: str = "", limit: int = 50, offset: int = 0, entity_type: Optional[str] = None, rank: Optional[str] = None) -> List[Dict[str, Any]]:
        terms = [term.lower() for term in query.split() if term.strip()]
        clauses, params = [], []
        if entity_type:
            clauses.append("entity_type = ?"); params.append(entity_type)
        if rank:
            clauses.append("LOWER(rank) = LOWER(?)"); params.append(rank)
        for term in terms:
            like = f"%{term}%"
            clauses.append("(LOWER(entity) LIKE ? OR LOWER(species) LIKE ? OR LOWER(rank) LIKE ? OR LOWER(status) LIKE ? OR LOWER(affiliation_json) LIKE ? OR LOWER(skills_json) LIKE ? OR LOWER(equipment_json) LIKE ? OR LOWER(evolution) LIKE ?)")
            params.extend([like] * 8)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(f"SELECT * FROM entities{where} ORDER BY entity COLLATE NOCASE LIMIT ? OFFSET ?", (*params, max(1, min(limit, 500)), max(0, offset))).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            for key in ("aliases_json", "categories_json", "affiliation_json", "skills_json", "equipment_json"):
                item[key[:-5]] = json.loads(item.pop(key) or "[]")
            result.append(item)
        return result
