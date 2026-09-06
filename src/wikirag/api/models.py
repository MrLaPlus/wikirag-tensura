from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    project: str = "tensura"
    llm_provider: Optional[str] = None # e.g. "ollama", "openrouter", "gemini", "openai"
    llm_model: Optional[str] = None
    fallback_model: Optional[str] = None
    enable_retry: bool = True
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_output_tokens: int = Field(default=1024, ge=1, le=16384)
    system_prompt: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=96)
    enable_reranking: bool = False
    enable_bm25: bool = False
    reranker_model: Optional[str] = None
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    verification_mode: str = "off"  # off | verify_only | suggest | auto
    verification_strictness: str = "balanced"  # fast | balanced | detailed
    verify_numbers: bool = True
    verify_names: bool = True
    verify_skill_ranks: bool = True
    verify_relationships: bool = True
    verify_citations: bool = True
    verify_unsupported: bool = True


class VerificationRequest(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    project: str = "tensura"
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    verification_strictness: str = "balanced"
    verify_numbers: bool = True
    verify_names: bool = True
    verify_skill_ranks: bool = True
    verify_relationships: bool = True
    verify_citations: bool = True
    verify_unsupported: bool = True


class ConversationCreateRequest(BaseModel):
    project: str = "tensura"
    title: str = "New chat"
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None


class ConversationUpdateRequest(BaseModel):
    title: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None



class SearchRequest(BaseModel):
    query: str
    project: str = "tensura"
    top_k: int = Field(default=5, ge=1, le=96)
    enable_reranking: bool = False
    enable_bm25: bool = False
    reranker_model: Optional[str] = None


class CitationResponse(BaseModel):
    index: int
    entity: str
    section: str
    url: str
    score: float


class SearchResponse(BaseModel):
    query: str
    chunks: List[Dict[str, Any]]
    archetype: Optional[str] = None


class EntityCard(BaseModel):
    entity: str
    canonical_url: str
    infobox: Optional[Dict[str, Any]] = None
    categories: List[str] = Field(default_factory=list)
    entity_type: Optional[str] = None
    species: Optional[str] = None
    rank: Optional[str] = None
    status: Optional[str] = None
    affiliation: List[str] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    equipment: List[str] = Field(default_factory=list)
    evolution: Optional[str] = None


class EntityDetailResponse(BaseModel):
    entity: str
    canonical_url: str
    infobox: Optional[Dict[str, Any]] = None
    categories: List[str] = Field(default_factory=list)
    aliases: List[str] = Field(default_factory=list)
    sections: List[Dict[str, Any]] = Field(default_factory=list)


class ProjectInfo(BaseModel):
    name: str
    title: str
    description: str
    doc_count: int
    table_name: str


class SyncStatus(BaseModel):
    is_syncing: bool
    status_message: str
    total_articles: int
    total_chunks: int
    crawled_pages: int = 0
    parsed_entities: int = 0
    relationships_count: int = 0
    cached_embeddings: int = 0
    last_sync: Optional[str] = None
    stage: str = "idle"  # "idle" | "crawling" | "parsing" | "embedding" | "completed" | "error"
    current_step: int = 0
    total_steps: int = 0
    progress_pct: float = 0.0
    current_item: Optional[str] = None
    cancel_requested: bool = False
    model_loaded: bool = False
    embedding_model: Optional[str] = None
    embedding_device: Optional[str] = None
    memory_mb: Optional[float] = None
