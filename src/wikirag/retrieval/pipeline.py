import os
import json
import re
import sqlite3
from typing import Any, Dict, List, Optional
from wikirag.chunking.embedder import BaseEmbedder
from wikirag.config import WikiRagProjectConfig
from wikirag.retrieval.fusion import reciprocal_rank_fusion
from wikirag.retrieval.preprocessing import QueryPreprocessor
from wikirag.retrieval.query_router import QueryArchetype, QueryRouter
from wikirag.retrieval.reranker import BaseReranker, LocalCrossEncoderReranker
from wikirag.utils.logging import get_logger
from wikirag.vectorstore.lancedb_store import LanceDBStore

logger = get_logger(__name__)


class RetrievalPipeline:
    """Production Hybrid Retrieval Pipeline.
    
    Combines:
    - Language detection & Alias expansion
    - Query Archetype Routing (Factual, Relational, etc.)
    - Dense vector similarity search
    - Sparse BM25 / FTS lexical keyword search
    - Reciprocal Rank Fusion (RRF)
    - Optional Cross-Encoder reranking (BAAI/bge-reranker-v2-m3)
    """

    def __init__(
        self,
        config: WikiRagProjectConfig,
        embedder: BaseEmbedder,
        vectorstore: LanceDBStore,
        reranker: Optional[BaseReranker] = None,
    ):
        self.config = config
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.preprocessor = QueryPreprocessor(config.storage.alias_map_path)
        self.router = QueryRouter()
        self.reranker = reranker

    def _related_entities(self, entity_names: List[str], max_names: int = 24) -> List[str]:
        """Expand explicit query entities through the existing SQLite graph."""
        graph_db = os.path.join(str(self.config.storage.data_dir), "graph.db")
        if not entity_names or not os.path.exists(graph_db):
            return []
        names = []
        try:
            with sqlite3.connect(graph_db) as conn:
                for name in entity_names[:8]:
                    rows = conn.execute(
                        """SELECT DISTINCT CASE WHEN source = ? THEN target ELSE source END
                           FROM relationships WHERE source = ? OR target = ? LIMIT 12""",
                        (name, name, name),
                    ).fetchall()
                    names.extend(row[0] for row in rows if row[0])
        except Exception as exc:
            logger.debug("Graph expansion unavailable: %s", exc)
        return list(dict.fromkeys(names))[:max_names]

    @staticmethod
    def _temporal_metadata(text: str) -> Dict[str, Any]:
        """Extract lightweight timeline hints without changing the existing vector schema."""
        value = text or ""
        volumes = re.findall(r"\b(?:volume|vol\.?|เล่ม)\s*([0-9]+)", value, re.I)
        chapters = re.findall(r"\b(?:chapter|ch\.?|ตอนที่)\s*([0-9]+)", value, re.I)
        return {
            "volumes": [int(v) for v in volumes],
            "chapters": [int(c) for c in chapters],
            "has_before_after": bool(re.search(r"\b(before|after|ก่อน|หลัง|ภายหลัง|ต่อมา)\b", value, re.I)),
        }

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        enable_reranking: Optional[bool] = None,
        enable_bm25: Optional[bool] = None,
        reranker_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Executes full hybrid retrieval with RRF and optional reranking."""
        k = top_k or self.config.retrieval.top_k
        lang = self.preprocessor.detect_language(query)
        archetype = self.router.classify(query)
        hints = self.router.get_strategy_hints(archetype)

        logger.debug(f"Query: '{query}' | Lang: {lang} | Archetype: {archetype}")

        # 1. Alias expansion
        expanded_query, matched_entities = self.preprocessor.expand_query(query)
        search_text = expanded_query if self.config.retrieval.query_expansion else query

        # Candidate pool size: retrieve 2-3x for fusion and reranking
        candidate_k = max(k * 3, 15)

        # 2. Dense Vector Leg
        query_vec = self.embedder.embed_query(search_text)
        dense_results = self.vectorstore.search(query_vec, top_k=candidate_k, filters=filters)

        # 3. Sparse BM25 / FTS Leg
        use_bm25 = enable_bm25 if enable_bm25 is not None else self.config.retrieval.enable_bm25_hybrid
        fts_results: List[Dict[str, Any]] = []
        if use_bm25:
            fts_results = self.vectorstore.search_fts(search_text, top_k=candidate_k, filters=filters)

        # 3b. Entity/relationship leg: use the existing graph to pull lexical
        # evidence for connected entities, without changing or re-embedding rows.
        related_names = self._related_entities(matched_entities)
        graph_results: List[Dict[str, Any]] = []
        if related_names:
            graph_results = self.vectorstore.search_fts(" ".join(related_names), top_k=candidate_k, filters=filters)
        structured_results = self.vectorstore.search_structured(search_text, top_k=candidate_k)

        # 4. Fusion Leg (RRF)
        if fts_results or graph_results or structured_results:
            candidates = reciprocal_rank_fusion(
                [dense_results, fts_results, graph_results, structured_results],
                k=60,
                top_n=candidate_k,
            )
        else:
            candidates = dense_results

        # 5. Factual Router Prioritization: if factual question, bump infobox chunk if present
        if hints.get("prioritize_infobox"):
            infobox_chunks = [c for c in candidates if c.get("chunk_type") == "infobox"]
            other_chunks = [c for c in candidates if c.get("chunk_type") != "infobox"]
            candidates = infobox_chunks + other_chunks

        # 6. Cross-Encoder Reranker Leg (Optional)
        do_rerank = enable_reranking if enable_reranking is not None else self.config.retrieval.enable_reranking
        if do_rerank and os.getenv("WIKIRAG_ALLOW_LARGE_RERANKER", "0") != "1":
            logger.warning("Reranking disabled: large cross-encoder is blocked. Set WIKIRAG_ALLOW_LARGE_RERANKER=1 to opt in.")
            do_rerank = False
        if do_rerank:
            try:
                import psutil
                available_gb = psutil.virtual_memory().available / (1024 ** 3)
                if available_gb < 3.0:
                    logger.warning("Reranking disabled: only %.1f GB RAM available", available_gb)
                    do_rerank = False
            except ImportError:
                pass
        if do_rerank:
            try:
                if not self.reranker:
                    self.reranker = LocalCrossEncoderReranker(
                        model_name=reranker_model or self.config.retrieval.reranker_model,
                    )
                final_results = self.reranker.rerank(query, candidates[:candidate_k], top_k=k)
            except Exception as exc:
                # Reranking is an optional quality layer. A model download,
                # OOM, or runtime failure must not take down normal retrieval.
                logger.warning("Reranker unavailable; using fused results: %s", str(exc)[:300])
                self.reranker = None
                final_results = candidates[:k]
        else:
            final_results = candidates[:k]

        # Attach metadata
        for r in final_results:
            r["query_language"] = lang
            r["matched_entities"] = matched_entities
            r["related_entities"] = related_names
            r["temporal"] = self._temporal_metadata(r.get("chunk_text", ""))
            r["archetype"] = archetype.value

        return final_results
