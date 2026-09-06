import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import lancedb
import numpy as np
import pyarrow as pa
from wikirag.utils.logging import get_logger
from wikirag.vectorstore.base import BaseVectorStore

logger = get_logger(__name__)


class LanceDBStore(BaseVectorStore):
    """Production local vector store powered by LanceDB (columnar, disk-based, zero-copy).
    
    Provides:
    - In-process storage without requiring external Docker/daemon
    - Idempotent upsert using unique content_hash keys
    - Native support for filtering by entity, chunk_type, category
    - Full-text search (BM25 / FTS) support for hybrid retrieval
    """

    def __init__(self, db_path: str, table_name: str = "wiki_chunks", dimension: int = 1024):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.table_name = table_name
        self.dimension = dimension
        self.db = lancedb.connect(str(self.db_path))
        self._table = None
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Opens existing table or creates an empty one with PyArrow schema."""
        existing_tables = self.db.table_names() if hasattr(self.db, "table_names") else []
        if self.table_name in existing_tables:
            self._table = self.db.open_table(self.table_name)
        else:
            schema = pa.schema(
                [
                    pa.field("chunk_id", pa.string()),
                    pa.field("content_hash", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), self.dimension)),
                    pa.field("entity", pa.string()),
                    pa.field("canonical_url", pa.string()),
                    pa.field("section_path", pa.string()),
                    pa.field("chunk_type", pa.string()),
                    pa.field("chunk_text", pa.string()),
                    pa.field("raw_text", pa.string()),
                    pa.field("infobox_json", pa.string()),
                    pa.field("categories_json", pa.string()),
                    pa.field("aliases_json", pa.string()),
                    pa.field("source_project", pa.string()),
                    pa.field("fetched_at", pa.float64()),
                ]
            )
            self._table = self.db.create_table(self.table_name, schema=schema)
            logger.info(f"Created LanceDB table '{self.table_name}' with dim={self.dimension}")

    def create_fts_index(self) -> None:
        """Builds native Tantivy full-text search index on chunk_text."""
        try:
            if len(self._table) > 0:
                self._table.create_fts_index("chunk_text", replace=True)
                logger.info("Created FTS index on 'chunk_text'")
        except Exception as e:
            logger.warning(f"Note on FTS index creation: {e}")

    def upsert_chunks(self, chunks: List[Dict[str, Any]], vectors: np.ndarray) -> int:
        """Upserts records into LanceDB using content_hash deduplication."""
        if not chunks:
            return 0

        # Retrieve existing content_hashes to skip duplicates
        existing_hashes = set()
        if len(self._table) > 0:
            try:
                df_existing = self._table.search().select(["content_hash"]).limit(1_000_000).to_arrow()
                existing_hashes = set(df_existing["content_hash"].to_pylist())
            except Exception as e:
                logger.warning(f"Could not read existing hashes for dedup: {e}")

        rows = []
        new_count = 0

        for i, chunk in enumerate(chunks):
            ch_hash = chunk.get("content_hash")
            if ch_hash in existing_hashes:
                continue

            rows.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "content_hash": ch_hash,
                    "vector": vectors[i].tolist(),
                    "entity": chunk.get("entity", ""),
                    "canonical_url": chunk.get("canonical_url", ""),
                    "section_path": chunk.get("section_path", ""),
                    "chunk_type": chunk.get("chunk_type", "text"),
                    "chunk_text": chunk.get("chunk_text", ""),
                    "raw_text": chunk.get("raw_text", ""),
                    "infobox_json": chunk.get("infobox_json") or "",
                    "categories_json": json.dumps(chunk.get("categories", []), ensure_ascii=False),
                    "aliases_json": json.dumps(chunk.get("aliases", []), ensure_ascii=False),
                    "source_project": chunk.get("source_project", ""),
                    "fetched_at": float(chunk.get("fetched_at", 0.0)),
                }
            )
            existing_hashes.add(ch_hash)
            new_count += 1

        if rows:
            self._table.add(rows)
            logger.info(f"Successfully indexed {new_count} new chunks into LanceDB.")

        return new_count

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Executes dense vector nearest-neighbor search with optional SQL-like WHERE filter."""
        if len(self._table) == 0:
            return []

        query = self._table.search(query_vector.tolist()).limit(top_k)

        if filters:
            conditions = []
            for k, v in filters.items():
                if isinstance(v, str):
                    conditions.append(f"{k} = '{v}'")
                elif isinstance(v, (int, float)):
                    conditions.append(f"{k} = {v}")
            if conditions:
                query = query.where(" AND ".join(conditions))

        results_df = query.to_arrow()
        records = []
        for i in range(len(results_df)):
            row = {col: results_df[col][i].as_py() for col in results_df.column_names}
            row["categories"] = json.loads(row.get("categories_json", "[]"))
            row["aliases"] = json.loads(row.get("aliases_json", "[]"))
            distance = row.get("_distance", 1.0)
            row["score"] = float(1.0 / (1.0 + distance))
            records.append(row)

        return records

    def search_fts(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Executes full-text keyword search (BM25 lexical leg)."""
        if len(self._table) == 0 or not query_text.strip():
            return []

        try:
            query = self._table.search(query_text, query_type="fts").limit(top_k)
            if filters:
                conditions = []
                for k, v in filters.items():
                    if isinstance(v, str):
                        conditions.append(f"{k} = '{v}'")
                    elif isinstance(v, (int, float)):
                        conditions.append(f"{k} = {v}")
                if conditions:
                    query = query.where(" AND ".join(conditions))

            results_df = query.to_arrow()
            records = []
            for i in range(len(results_df)):
                row = {col: results_df[col][i].as_py() for col in results_df.column_names}
                row["categories"] = json.loads(row.get("categories_json", "[]"))
                row["aliases"] = json.loads(row.get("aliases_json", "[]"))
                row["score"] = float(row.get("_score", 1.0))
                records.append(row)
            return records
        except Exception as e:
            logger.debug(f"FTS search fallback (no FTS index yet or query syntax): {e}")
            return []

    def search_structured(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search infobox/category/alias fields directly without re-embedding rows."""
        if len(self._table) == 0 or not query_text.strip():
            return []

    def metadata_rows(self) -> List[Dict[str, Any]]:
        """Returns metadata needed to build the separate structured index."""
        if len(self._table) == 0:
            return []
        cols = ["chunk_id", "entity", "canonical_url", "section_path", "chunk_text", "infobox_json", "categories_json", "aliases_json"]
        table = self._table.search().select(cols).limit(1_000_000).to_arrow()
        return [{col: table[col][i].as_py() for col in cols} for i in range(len(table))]

    def quality_rows(self) -> List[Dict[str, Any]]:
        """Return lightweight fields used by the data-quality audit."""
        if len(self._table) == 0:
            return []
        cols = ["chunk_id", "content_hash", "entity", "chunk_text", "fetched_at"]
        table = self._table.search().select(cols).limit(1_000_000).to_arrow()
        return [{col: table[col][i].as_py() for col in cols} for i in range(len(table))]
        terms = [t.lower() for t in query_text.split() if len(t) >= 2]
        try:
            cols = ["chunk_id", "entity", "canonical_url", "section_path", "chunk_type", "chunk_text", "infobox_json", "categories_json", "aliases_json"]
            tbl = self._table.search().select(cols).limit(1_000_000).to_arrow()
            results, seen = [], set()
            for i in range(len(tbl)):
                haystack = " ".join(str(tbl[col][i].as_py() or "") for col in cols[1:]).lower()
                score = sum(term in haystack for term in terms)
                key = tbl["chunk_id"][i].as_py()
                if score and key not in seen:
                    seen.add(key)
                    row = {col: tbl[col][i].as_py() for col in cols}
                    row["categories"] = json.loads(row.get("categories_json") or "[]")
                    row["aliases"] = json.loads(row.get("aliases_json") or "[]")
                    row["score"] = float(score)
                    results.append(row)
            return sorted(results, key=lambda r: r["score"], reverse=True)[:top_k]
        except Exception as e:
            logger.debug(f"Structured search unavailable: {e}")
            return []

    def get_entities(self, limit: int = 50, offset: int = 0, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves paginated unique entity cards with their infobox and category metadata."""
        if len(self._table) == 0:
            return []

        try:
            # Query infobox or lead chunks to build entity summaries
            q = self._table.search().select(["entity", "canonical_url", "infobox_json", "categories_json", "chunk_type"])
            if category:
                q = q.where(f"categories_json LIKE '%{category}%'")

            # Scan the complete table before deduplicating by entity. A 1000-row
            # cap silently omitted entities when each page had multiple chunks.
            arrow_tbl = q.limit(1_000_000).to_arrow()
            seen_entities = {}
            for i in range(len(arrow_tbl)):
                ent = arrow_tbl["entity"][i].as_py()
                if ent not in seen_entities:
                    infobox_raw = arrow_tbl["infobox_json"][i].as_py()
                    cats_raw = arrow_tbl["categories_json"][i].as_py()
                    seen_entities[ent] = {
                        "entity": ent,
                        "canonical_url": arrow_tbl["canonical_url"][i].as_py(),
                        "infobox": json.loads(infobox_raw) if infobox_raw else None,
                        "categories": json.loads(cats_raw) if cats_raw else [],
                    }

            all_list = list(seen_entities.values())
            return all_list[offset : offset + limit]
        except Exception as e:
            logger.error(f"Error fetching entities: {e}")
            return []

    def get_entity_detail(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """Fetches all chunks and structured infobox for a specific entity."""
        if len(self._table) == 0:
            return None

        try:
            safe_entity = entity_name.replace("'", "''")
            arrow_tbl = self._table.search().where(f"entity = '{safe_entity}'").limit(100).to_arrow()
            if len(arrow_tbl) == 0:
                return None

            sections = []
            infobox = None
            canonical_url = ""
            categories = []
            aliases = []

            for i in range(len(arrow_tbl)):
                canonical_url = arrow_tbl["canonical_url"][i].as_py()
                ch_type = arrow_tbl["chunk_type"][i].as_py()
                sec_path = arrow_tbl["section_path"][i].as_py()
                chunk_txt = arrow_tbl["chunk_text"][i].as_py()
                cats = json.loads(arrow_tbl["categories_json"][i].as_py())
                al = json.loads(arrow_tbl["aliases_json"][i].as_py())

                if cats and not categories:
                    categories = cats
                if al and not aliases:
                    aliases = al

                if ch_type == "infobox":
                    ib_str = arrow_tbl["infobox_json"][i].as_py()
                    if ib_str:
                        infobox = json.loads(ib_str)

                sections.append({
                    "section_path": sec_path,
                    "chunk_type": ch_type,
                    "text": chunk_txt,
                })

            return {
                "entity": entity_name,
                "canonical_url": canonical_url,
                "infobox": infobox,
                "categories": categories,
                "aliases": aliases,
                "sections": sections,
            }
        except Exception as e:
            logger.error(f"Error fetching entity detail for '{entity_name}': {e}")
            return None

    def count(self) -> int:
        return len(self._table)
