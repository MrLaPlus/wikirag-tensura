import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field


class ProjectMetadata(BaseModel):
    name: str
    title: str = "Wiki Knowledge Base"
    description: str = ""
    license: str = "CC BY-SA 3.0"
    attribution_required: bool = True


class MediaWikiSourceConfig(BaseModel):
    type: str = "mediawiki"
    api_url: str
    base_url: str = ""
    namespace: int = 0
    user_agent: str = "WikiRAG/0.1.0 (Localhost RAG Research)"
    request_delay_seconds: float = 0.5
    batch_size: int = 50
    checkpoint_interval: int = 100


class StorageConfig(BaseModel):
    data_dir: str = "./data"
    raw_dir: str = "./data/raw"
    parsed_dir: str = "./data/parsed"
    embeddings_cache_dir: str = "./data/embeddings"
    vectordb_dir: str = "./data/vectordb"
    alias_map_path: str = "./data/aliases.json"
    state_file: str = "./data/state.json"
    failed_pages_path: str = "./data/failed_pages.jsonl"


class ChunkingConfig(BaseModel):
    target_min_tokens: int = 400
    target_max_tokens: int = 800
    overlap_percentage: float = 0.15
    separate_infobox_chunk: bool = True
    prepend_contextual_header: bool = True


class EmbeddingConfig(BaseModel):
    provider: str = "sentence-transformers"
    model_name: str = "BAAI/bge-m3"
    fallback_model: str = "intfloat/multilingual-e5-large"
    device: str = "auto"
    backend: str = "onnx" # "default" or "onnx"
    quantization: str = "int8" # "none", "int8", "fp16"
    batch_size: int = 8
    normalize: bool = True


class VectorStoreConfig(BaseModel):
    type: str = "lancedb"
    table_name: str = "wiki_chunks"


class RetrievalConfig(BaseModel):
    top_k: int = 5
    enable_reranking: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    enable_bm25_hybrid: bool = False
    query_expansion: bool = True


class LLMConfig(BaseModel):
    default_provider: str = "ollama"
    default_model: str = "llama3.1:8b"
    temperature: float = 0.1
    max_output_tokens: int = 1024
    system_prompt: Optional[str] = None


class WikiRagProjectConfig(BaseModel):
    project: ProjectMetadata
    source: MediaWikiSourceConfig
    storage: StorageConfig
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)


def load_project_config(project_name_or_path: str) -> WikiRagProjectConfig:
    """Loads and validates a project YAML file.
    
    Accepts either:
    - A project name (e.g. 'tensura', looked up in projects/tensura.yaml)
    - A direct path to a YAML file.
    """
    path = Path(project_name_or_path)

    # Derive the project root: parent of the 'src' directory where this package lives
    # e.g.  .../<project_root>/src/wikirag/config.py  →  <project_root>
    _pkg_root = Path(__file__).resolve().parent.parent.parent  # src/wikirag → src → project_root

    if not path.exists():
        candidates = [
            # Relative to package root (most reliable — works regardless of CWD)
            _pkg_root / "projects" / f"{project_name_or_path}.yaml",
            # Relative to CWD (legacy fallback)
            Path("projects") / f"{project_name_or_path}.yaml",
        ]
        resolved = None
        for c in candidates:
            if c.exists():
                resolved = c
                break
        if resolved is None:
            raise FileNotFoundError(
                f"Project configuration not found. Tried:\n"
                + "\n".join(f"  - {c}" for c in candidates)
            )
        path = resolved

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    config = WikiRagProjectConfig.model_validate(data)

    # Resolve all relative storage paths to be absolute, anchored at the project root
    # so the data directory is always at <project_root>/data/ regardless of CWD.
    def _abs(p: str) -> str:
        pp = Path(p)
        if pp.is_absolute():
            return p
        return str(_pkg_root / pp)

    s = config.storage
    config.storage = s.model_copy(update={
        "data_dir": _abs(s.data_dir),
        "raw_dir": _abs(s.raw_dir),
        "parsed_dir": _abs(s.parsed_dir),
        "embeddings_cache_dir": _abs(s.embeddings_cache_dir),
        "vectordb_dir": _abs(s.vectordb_dir),
        "alias_map_path": _abs(s.alias_map_path),
        "state_file": _abs(s.state_file),
        "failed_pages_path": _abs(s.failed_pages_path),
    })

    # Ensure required storage directories exist

    for dir_path in [
        config.storage.data_dir,
        config.storage.raw_dir,
        config.storage.parsed_dir,
        config.storage.embeddings_cache_dir,
        config.storage.vectordb_dir,
    ]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    return config
