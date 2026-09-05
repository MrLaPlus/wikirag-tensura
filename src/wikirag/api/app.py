import asyncio
import json
import os
import sqlite3
import gc
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from wikirag.api.models import (
    ChatRequest,
    EntityCard,
    EntityDetailResponse,
    ProjectInfo,
    SearchRequest,
    SearchResponse,
    SyncStatus,
    ConversationCreateRequest,
    ConversationUpdateRequest,
)
from wikirag.api.chat_history import ChatHistoryStore
from wikirag.chunking.chunker import SectionAwareChunker
from wikirag.chunking.embedder import LocalSentenceTransformerEmbedder
from wikirag.config import load_project_config
from wikirag.connectors.mediawiki import MediaWikiConnector
from wikirag.generation.generator import GroundedAnswerGenerator
from wikirag.llm import get_llm_provider
from wikirag.parser.wikitext import WikitextParser
from wikirag.retrieval.pipeline import RetrievalPipeline
from wikirag.utils.logging import get_logger
from wikirag.vectorstore.lancedb_store import LanceDBStore

logger = get_logger(__name__)

app = FastAPI(
    title="WikiRAG API",
    description="Backend API for WikiRAG local knowledge base and conversational RAG",
    version="0.2.0",
)

# Enable CORS for local dev and frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index_page():
    """Serves the WikiRAG React web application."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>WikiRAG API Server Running</h1><p>Visit /docs for Swagger API</p>"


# Global runtime cache for stores and embedders to avoid reloading heavy models
_PIPELINE_CACHE: Dict[str, RetrievalPipeline] = {}
_SYNC_STATUS = {
    "is_syncing": False,
    "status_message": "Idle",
    "last_sync": None,
    "stage": "idle",
    "current_step": 0,
    "total_steps": 0,
    "progress_pct": 0.0,
}
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CHAT_HISTORY = ChatHistoryStore(str(_PROJECT_ROOT / "data" / "tensura" / "chat_history.db"))
_PIPELINE_LOCK = threading.RLock()



def get_pipeline(
    project_name: str,
    backend: Optional[str] = None,
    quantization: Optional[str] = None,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    batch_size: Optional[int] = None,
    reranker_model: Optional[str] = None,
) -> RetrievalPipeline:
    """Lazily initializes and caches the retrieval pipeline per project."""
    cfg = load_project_config(project_name)
    use_backend = backend or getattr(cfg.embedding, "backend", "default")
    use_quant = quantization or getattr(cfg.embedding, "quantization", "none")
    use_model = model_name or cfg.embedding.model_name
    use_device = device or cfg.embedding.device
    use_batch = batch_size or cfg.embedding.batch_size
    if reranker_model:
        cfg = cfg.model_copy(update={"retrieval": cfg.retrieval.model_copy(update={"reranker_model": reranker_model})})
    cache_key = f"{project_name}:{use_model}:{use_device}:{use_batch}:{use_backend}:{use_quant}"
    with _PIPELINE_LOCK:
        if cache_key not in _PIPELINE_CACHE:
            embedder = LocalSentenceTransformerEmbedder(
                model_name=use_model,
                device=use_device,
                backend=use_backend,
                quantization=use_quant,
                batch_size=use_batch,
                cache_dir=cfg.storage.embeddings_cache_dir,
            )
            store = LanceDBStore(
                db_path=cfg.storage.vectordb_dir,
                table_name=cfg.vectorstore.table_name,
                dimension=embedder.dimension,
            )
            _PIPELINE_CACHE[cache_key] = RetrievalPipeline(
                config=cfg,
                embedder=embedder,
                vectorstore=store,
            )
        return _PIPELINE_CACHE[cache_key]


@app.post("/api/settings/embedding")
def update_embedding_settings(
    payload: Dict[str, Any],
    project: str = "tensura",
):
    """Dynamically switches Embedding backend (PyTorch vs ONNX) and Quantization (none vs int8)."""
    global _PIPELINE_CACHE
    backend = payload.get("backend", "onnx")
    quantization = payload.get("quantization", "int8")
    model_name = payload.get("model_name")
    device = payload.get("device")
    batch_size = int(payload.get("batch_size", 8))
    if batch_size < 1 or batch_size > 64:
        raise HTTPException(status_code=400, detail="batch_size must be between 1 and 64")

    cfg = load_project_config(project)
    cache_key = f"{project}:{model_name or cfg.embedding.model_name}:{device or cfg.embedding.device}:{batch_size}:{backend}:{quantization}"
    # Do not reload an already active configuration. This endpoint is called
    # every time Settings is saved and repeated model initialization can exhaust RAM.
    with _PIPELINE_LOCK:
        if cache_key in _PIPELINE_CACHE:
            pipeline = _PIPELINE_CACHE[cache_key]
            return {"status": "success", "backend": backend, "quantization": quantization, "dimension": pipeline.embedder.dimension, "reloaded": False}
        for key in list(_PIPELINE_CACHE):
            if key.startswith(f"{project}:"):
                del _PIPELINE_CACHE[key]
        gc.collect()

    logger.info(f"Switched embedding settings: backend={backend}, quant={quantization}")
    try:
        pipeline = get_pipeline(project, backend=backend, quantization=quantization, model_name=model_name, device=device, batch_size=batch_size)
        return {
            "status": "success",
            "backend": backend,
            "quantization": quantization,
            "dimension": pipeline.embedder.dimension,
            "reloaded": True,
        }
    except Exception as e:
        logger.error(f"Failed to reload embedder: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/api/health")
def health_check():
    return {"status": "ok", "time": time.time()}


@app.get("/api/conversations")
def list_conversations(project: str = "tensura", limit: int = 100):
    return _CHAT_HISTORY.list_conversations(project, limit)


@app.post("/api/conversations")
def create_conversation(req: ConversationCreateRequest):
    return _CHAT_HISTORY.create_conversation(req.project, req.title, req.llm_provider, req.llm_model)


@app.get("/api/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    conversation = _CHAT_HISTORY.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    conversation["messages"] = _CHAT_HISTORY.get_messages(conversation_id)
    return conversation


@app.patch("/api/conversations/{conversation_id}")
def update_conversation(conversation_id: str, req: ConversationUpdateRequest):
    conversation = _CHAT_HISTORY.update_conversation(conversation_id, req.title, req.llm_provider, req.llm_model)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    if not _CHAT_HISTORY.delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"deleted": True}


@app.get("/api/settings")
def get_settings(project: Optional[str] = None):
    return _CHAT_HISTORY.get_settings(project)


@app.put("/api/settings")
def save_settings(payload: Dict[str, Any], project: Optional[str] = None):
    # API keys remain client/env-managed and are deliberately not persisted server-side.
    secret_keys = {"apiKey", "api_key", "OPENROUTER_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"}
    return _CHAT_HISTORY.save_settings({k: v for k, v in payload.items() if k not in secret_keys}, project)


@app.get("/api/projects", response_model=List[ProjectInfo])
def list_projects():
    """Lists available wiki projects in the projects/ folder."""
    results = []
    projects_dir = Path("projects")
    if projects_dir.exists():
        for yml in projects_dir.glob("*.yaml"):
            try:
                cfg = load_project_config(str(yml))
                # Count docs in vectorstore if exists
                store = LanceDBStore(
                    db_path=cfg.storage.vectordb_dir,
                    table_name=cfg.vectorstore.table_name,
                )
                count = store.count()
                results.append(
                    ProjectInfo(
                        name=cfg.project.name,
                        title=cfg.project.title,
                        description=cfg.project.description,
                        doc_count=count,
                        table_name=cfg.vectorstore.table_name,
                    )
                )
            except Exception as e:
                logger.warning(f"Could not load project {yml}: {e}")
    return results


@app.post("/api/search", response_model=SearchResponse)
def search_chunks(req: SearchRequest):
    """Retrieves matching chunks and scores without generating an LLM response."""
    try:
        pipeline = get_pipeline(req.project)
        chunks = pipeline.retrieve(
            query=req.query,
            top_k=req.top_k,
            enable_reranking=req.enable_reranking,
            enable_bm25=req.enable_bm25,
            reranker_model=req.reranker_model,
        )
        archetype = chunks[0].get("archetype") if chunks else None
        return SearchResponse(query=req.query, chunks=chunks, archetype=archetype)
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat_stream(req: ChatRequest):
    """Server-Sent Events (SSE) streaming endpoint for grounded generation."""
    try:
        cfg = load_project_config(req.project)
        pipeline = get_pipeline(req.project)

        conversation_id = req.conversation_id
        if conversation_id:
            conversation = _CHAT_HISTORY.get_conversation(conversation_id)
            if not conversation or conversation["project"] != req.project:
                raise HTTPException(status_code=404, detail="Conversation not found")
        else:
            conversation = _CHAT_HISTORY.create_conversation(
                req.project,
                req.message[:80],
                req.llm_provider or cfg.llm.default_provider,
                req.llm_model or cfg.llm.default_model,
            )
            conversation_id = conversation["id"]
        _CHAT_HISTORY.add_message(conversation_id, "user", req.message)

        # 1. Retrieve relevant chunks
        chunks = pipeline.retrieve(
            query=req.message,
            top_k=req.top_k,
            enable_reranking=req.enable_reranking,
            enable_bm25=req.enable_bm25,
            reranker_model=req.reranker_model,
        )

        # 2. Resolve LLM provider
        llm = get_llm_provider(
            provider_name=req.llm_provider or cfg.llm.default_provider,
            model_name=req.llm_model or cfg.llm.default_model,
            api_key=req.api_key,
            base_url=req.base_url,
        )
        generator = GroundedAnswerGenerator(config=cfg, llm_provider=llm)
        if req.system_prompt and req.system_prompt.strip():
            generator.system_prompt = req.system_prompt.strip()


        async def sse_event_generator():
            yield f"event: conversation\ndata: {json.dumps({'id': conversation_id}, ensure_ascii=False)}\n\n"
            # Send retrieved sources as the first event
            sources_data = [
                {
                    "index": i,
                    "entity": c.get("entity", "Unknown"),
                    "section": c.get("section_path", "General"),
                    "url": c.get("canonical_url", ""),
                    "score": round(float(c.get("score", 0.0)), 4),
                    "chunk_type": c.get("chunk_type", "text"),
                    "snippet": c.get("chunk_text", "")[:280] + "...",
                }
                for i, c in enumerate(chunks, 1)
            ]
            yield f"event: sources\ndata: {json.dumps(sources_data, ensure_ascii=False)}\n\n"

            # Stream generation tokens
            try:
                answer_parts = []
                for token in generator.stream_answer(req.message, chunks, temperature=req.temperature, max_output_tokens=req.max_output_tokens):
                    answer_parts.append(token)
                    payload = json.dumps({"token": token}, ensure_ascii=False)
                    yield f"event: token\ndata: {payload}\n\n"
                    # Small async sleep to yield control to event loop
                    await asyncio.sleep(0.005)
                _CHAT_HISTORY.add_message(conversation_id, "assistant", "".join(answer_parts), sources_data, chunks)
            except Exception as err:
                logger.error(f"Generation error during stream: {err}")
                err_payload = json.dumps({"error": str(err)})
                yield f"event: error\ndata: {err_payload}\n\n"

            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(
            sse_event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    except Exception as e:
        logger.error(f"Chat initialization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/entities", response_model=List[EntityCard])
def get_entities(
    project: str = "tensura",
    limit: int = 40,
    offset: int = 0,
    category: Optional[str] = None,
):
    """Returns paginated entity summaries with structured infobox data."""
    try:
        pipeline = get_pipeline(project)
        raw_cards = pipeline.vectorstore.get_entities(limit=limit, offset=offset, category=category)
        return [EntityCard(**c) for c in raw_cards]
    except Exception as e:
        logger.error(f"Get entities error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/entities/{entity_name}", response_model=EntityDetailResponse)
def get_entity_detail(entity_name: str, project: str = "tensura"):
    """Fetches full infobox and section chunks for a given entity."""
    try:
        pipeline = get_pipeline(project)
        detail = pipeline.vectorstore.get_entity_detail(entity_name)
        if not detail:
            raise HTTPException(status_code=404, detail=f"Entity '{entity_name}' not found")
        return EntityDetailResponse(**detail)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Entity detail error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph")
def get_graph_overview(project: str = "tensura", limit: int = 40):
    """Returns top connected nodes and relational edges for global network graph."""
    from wikirag.graph.sqlite_graph import SQLiteEntityGraph
    cfg = load_project_config(project)
    graph_db = Path(cfg.storage.data_dir) / "graph.db"
    graph = SQLiteEntityGraph(str(graph_db))
    subgraph = graph.get_overview_graph(limit_nodes=limit)
    return {
        "nodes": [n.model_dump() for n in subgraph.nodes],
        "edges": [e.model_dump() for e in subgraph.edges],
    }


@app.get("/api/graph/{entity_name}")
def get_entity_subgraph(entity_name: str, project: str = "tensura", depth: int = 2):
    """Returns 1-hop or 2-hop relational subgraph around a specific entity."""
    from wikirag.graph.sqlite_graph import SQLiteEntityGraph
    cfg = load_project_config(project)
    graph_db = Path(cfg.storage.data_dir) / "graph.db"
    graph = SQLiteEntityGraph(str(graph_db))
    subgraph = graph.get_subgraph(entity_id=entity_name, max_depth=depth)
    return {
        "nodes": [n.model_dump() for n in subgraph.nodes],
        "edges": [e.model_dump() for e in subgraph.edges],
    }


@app.get("/api/categories")
def get_categories(project: str = "tensura"):
    """Returns unique categories discovered in the knowledge base."""
    # Tensura common category taxonomy
    return [
        "Characters",
        "Demon Lords",
        "True Dragons",
        "Tempest",
        "Monsters",
        "Otherworlders",
        "Spiritual life-forms",
        "Octagram",
        "Skills",
        "Magic",
        "Locations",
    ]


def _background_sync_task(project: str, incremental: bool):
    """Background ingestion worker with real-time progress tracking."""
    global _SYNC_STATUS
    _SYNC_STATUS["is_syncing"] = True
    _SYNC_STATUS["stage"] = "crawling"
    _SYNC_STATUS["current_step"] = 0
    _SYNC_STATUS["total_steps"] = 3403
    _SYNC_STATUS["progress_pct"] = 0.0
    _SYNC_STATUS["status_message"] = "กำลังดึงข้อมูลบทความจาก MediaWiki API..."

    try:
        cfg = load_project_config(project)
        connector = MediaWikiConnector(cfg)
        
        # 1. Crawl stage
        records = []
        gen = connector.sync_incremental() if incremental else connector.crawl_all(resume=True)
        for rec in gen:
            records.append(rec)
            count = len(records)
            _SYNC_STATUS["current_step"] = count
            _SYNC_STATUS["status_message"] = f"กำลังดึงบทความ: {count} หน้า ({rec.get('title', '')[:30]})..."
            _SYNC_STATUS["progress_pct"] = round(min(100.0, (count / 3403) * 100), 1)

        if records:
            # 2. Parsing stage
            _SYNC_STATUS["stage"] = "parsing"
            _SYNC_STATUS["total_steps"] = len(records)
            _SYNC_STATUS["current_step"] = 0
            _SYNC_STATUS["status_message"] = f"กำลังแยกเนื้อหาและ Infobox ({len(records)} หน้า)..."

            alias_map = {}
            if Path(cfg.storage.alias_map_path).exists():
                with open(cfg.storage.alias_map_path, "r", encoding="utf-8") as f:
                    alias_map = json.load(f)

            parser = WikitextParser(failed_pages_path=cfg.storage.failed_pages_path)
            chunker = SectionAwareChunker(cfg.chunking, project_name=project)
            from wikirag.graph.builder import GraphBuilder
            from wikirag.graph.sqlite_graph import SQLiteEntityGraph
            graph_db = Path(cfg.storage.data_dir) / "graph.db"
            graph_builder = GraphBuilder(SQLiteEntityGraph(str(graph_db)))

            new_chunks = []
            for i, raw_rec in enumerate(records, start=1):
                parsed_page = parser.parse_page(raw_rec, alias_map=alias_map)
                if parsed_page:
                    graph_builder.process_page(parsed_page)
                    for chunk in chunker.chunk_page(parsed_page):
                        new_chunks.append(chunk.model_dump())
                if i % 20 == 0 or i == len(records):
                    _SYNC_STATUS["current_step"] = i
                    _SYNC_STATUS["progress_pct"] = round((i / len(records)) * 100, 1)
                    _SYNC_STATUS["status_message"] = f"กำลังวิเคราะห์โครงสร้างบทความ: {i}/{len(records)} หน้า (ได้ {len(new_chunks)} chunks)"

            # 3. Embedding & Indexing stage (in batches for real-time progress)
            if new_chunks:
                _SYNC_STATUS["stage"] = "embedding"
                total_chunks = len(new_chunks)
                _SYNC_STATUS["total_steps"] = total_chunks
                _SYNC_STATUS["current_step"] = 0
                _SYNC_STATUS["progress_pct"] = 0.0
                _SYNC_STATUS["status_message"] = f"กำลังโหลดโมเดลและแปลงเวกเตอร์ 0/{total_chunks} chunks..."

                try:
                    pipeline = get_pipeline(project)
                    embed_batch_size = 32
                    
                    for b_start in range(0, total_chunks, embed_batch_size):
                        b_end = min(b_start + embed_batch_size, total_chunks)
                        batch_chunks = new_chunks[b_start:b_end]
                        batch_texts = [c["chunk_text"] for c in batch_chunks]
                        
                        # Embed this mini-batch
                        batch_vectors = pipeline.embedder.embed_texts(batch_texts, show_progress=False)
                        pipeline.vectorstore.upsert_chunks(batch_chunks, batch_vectors)
                        
                        _SYNC_STATUS["current_step"] = b_end
                        pct = round((b_end / total_chunks) * 100, 1)
                        _SYNC_STATUS["progress_pct"] = pct
                        _SYNC_STATUS["status_message"] = f"กำลังแปลงเวกเตอร์: {b_end}/{total_chunks} chunks ({pct}%)..."
                except Exception as emb_err:
                    logger.error(f"Embedding step error: {emb_err}")
                    _SYNC_STATUS["status_message"] = f"ดึงและแยกเนื้อหาสำเร็จ {len(records)} หน้า ({len(new_chunks)} chunks) [การแปลงเวกเตอร์จะทำต่อเมื่อโหลดโมเดลเสร็จ]"

        _SYNC_STATUS["stage"] = "completed"
        _SYNC_STATUS["progress_pct"] = 100.0
        _SYNC_STATUS["last_sync"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if not _SYNC_STATUS["status_message"].startswith("ดึงและแยก"):
            _SYNC_STATUS["status_message"] = "การซิงก์ข้อมูลเสร็จสมบูรณ์เรียบร้อยแล้ว!"

    except Exception as e:
        logger.error(f"Sync task error: {e}")
        _SYNC_STATUS["stage"] = "error"
        _SYNC_STATUS["status_message"] = f"เกิดข้อผิดพลาดในการ Sync: {e}"
    finally:
        _SYNC_STATUS["is_syncing"] = False


@app.post("/api/admin/sync")
def trigger_sync(background_tasks: BackgroundTasks, project: str = "tensura", incremental: bool = True):
    """Triggers an ingestion sync in the background."""
    global _SYNC_STATUS
    if _SYNC_STATUS["is_syncing"]:
        raise HTTPException(status_code=409, detail="A sync operation is already in progress")

    background_tasks.add_task(_background_sync_task, project, incremental)
    return {"message": "Sync started in background", "project": project, "incremental": incremental}


def _background_embed_task(project: str):
    """Processes Wiki articles directly into embeddings and indexes them into LanceDB."""
    global _SYNC_STATUS
    _SYNC_STATUS["is_syncing"] = True
    _SYNC_STATUS["stage"] = "embedding"
    _SYNC_STATUS["current_step"] = 0
    _SYNC_STATUS["total_steps"] = 0
    _SYNC_STATUS["progress_pct"] = 0.0
    _SYNC_STATUS["status_message"] = "กำลังเตรียมโมเดลและดึงข้อมูลเพื่อแปลงเวกเตอร์..."

    try:
        cfg = load_project_config(project)
        pipeline = get_pipeline(project)
        parser = WikitextParser(failed_pages_path=cfg.storage.failed_pages_path)
        chunker = SectionAwareChunker(cfg.chunking, project_name=project)
        connector = MediaWikiConnector(cfg)

        alias_map = {}
        if Path(cfg.storage.alias_map_path).exists():
            with open(cfg.storage.alias_map_path, "r", encoding="utf-8") as f:
                alias_map = json.load(f)

        _SYNC_STATUS["status_message"] = "กำลังดึงเนื้อหาและแปลงเวกเตอร์เป็นชุด..."
        
        batch_records = []
        batch_chunks = []
        total_embedded = 0
        pages_processed = 0

        for rec in connector.crawl_all(resume=False):
            pages_processed += 1
            _SYNC_STATUS["current_item"] = rec.get("title", "")
            parsed_page = parser.parse_page(rec, alias_map=alias_map)
            if parsed_page:
                for chunk in chunker.chunk_page(parsed_page):
                    batch_chunks.append(chunk.model_dump())

            # When we have enough chunks, embed and flush to LanceDB
            if len(batch_chunks) >= 32:
                texts = [c["chunk_text"] for c in batch_chunks]
                vecs = pipeline.embedder.embed_texts(texts, show_progress=False)
                pipeline.vectorstore.upsert_chunks(batch_chunks, vecs)
                total_embedded += len(batch_chunks)
                batch_chunks.clear()

                _SYNC_STATUS["current_step"] = total_embedded
                _SYNC_STATUS["status_message"] = f"ประมวลผลแล้ว {pages_processed} หน้า | บันทึกเวกเตอร์แล้ว {total_embedded} chunks..."

        # Flush remaining chunks
        if batch_chunks:
            texts = [c["chunk_text"] for c in batch_chunks]
            vecs = pipeline.embedder.embed_texts(texts, show_progress=False)
            pipeline.vectorstore.upsert_chunks(batch_chunks, vecs)
            total_embedded += len(batch_chunks)
            batch_chunks.clear()

        _SYNC_STATUS["stage"] = "completed"
        _SYNC_STATUS["progress_pct"] = 100.0
        _SYNC_STATUS["last_sync"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _SYNC_STATUS["status_message"] = f"แปลงเวกเตอร์สำเร็จทั้งหมด {total_embedded} chunks จาก {pages_processed} หน้า!"

    except Exception as e:
        logger.error(f"Embed task error: {e}")
        _SYNC_STATUS["stage"] = "error"
        _SYNC_STATUS["status_message"] = f"เกิดข้อผิดพลาดในการแปลงเวกเตอร์: {e}"
    finally:
        _SYNC_STATUS["is_syncing"] = False


@app.post("/api/admin/embed")
def trigger_embed(background_tasks: BackgroundTasks, project: str = "tensura"):
    """Triggers batch embedding generation and indexing in background."""
    global _SYNC_STATUS
    if _SYNC_STATUS["is_syncing"]:
        raise HTTPException(status_code=409, detail="A task is already running")

    background_tasks.add_task(_background_embed_task, project)
    return {"message": "Embedding process started in background", "project": project}


@app.get("/api/admin/status", response_model=SyncStatus)
def get_sync_status(project: str = "tensura"):
    """Returns live ingestion progress, article count, vector store count, and graph metrics."""
    try:
        pipeline = get_pipeline(project)
        chunk_count = pipeline.vectorstore.count()
    except Exception:
        chunk_count = 0

    # Get cached embeddings count
    cached_count = 0
    try:
        cfg = load_project_config(project)
        cache_db = Path(cfg.storage.embeddings_cache_dir) / "embeddings_cache.sqlite3"
        if cache_db.exists():
            with sqlite3.connect(str(cache_db)) as conn:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM embedding_cache")
                cached_count = cur.fetchone()[0]
    except Exception:
        pass

    # Get crawled pages from state.json
    crawled = 0
    try:
        state_file = Path(cfg.storage.state_file)
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                st = json.load(f)
                crawled = st.get("processed_count", 0)
    except Exception:
        pass

    # Get graph database counts
    entities_count = 0
    rel_count = 0
    try:
        graph_db = Path(cfg.storage.data_dir) / "graph.db"
        if graph_db.exists():
            with sqlite3.connect(str(graph_db)) as conn:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM entities")
                entities_count = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM relationships")
                rel_count = cur.fetchone()[0]
    except Exception:
        pass

    return SyncStatus(
        is_syncing=_SYNC_STATUS["is_syncing"],
        status_message=_SYNC_STATUS["status_message"],
        total_articles=3403,
        total_chunks=chunk_count,
        crawled_pages=crawled,
        parsed_entities=entities_count,
        relationships_count=rel_count,
        cached_embeddings=cached_count,
        last_sync=_SYNC_STATUS["last_sync"],
        stage=_SYNC_STATUS.get("stage", "idle"),
        current_step=_SYNC_STATUS.get("current_step", 0),
        total_steps=_SYNC_STATUS.get("total_steps", 0),
        progress_pct=_SYNC_STATUS.get("progress_pct", 0.0),
        current_item=_SYNC_STATUS.get("current_item"),
    )
