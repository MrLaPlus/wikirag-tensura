import json
import os
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
import numpy as np
from wikirag.utils.hashing import compute_sha256
from wikirag.utils.logging import get_logger

logger = get_logger(__name__)


class BaseEmbedder(ABC):
    """Abstract interface for all text embedding providers."""

    @abstractmethod
    def embed_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Generates embedding vectors for a list of strings."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Embeds a single search query string."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the vector dimensionality."""
        pass


class DiskEmbeddingCache:
    """Persistent SQLite disk cache for embeddings.
    
    Prevents re-embedding identical text across runs when re-chunking or resuming.
    Keyed by SHA-256 hash of the input text.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "embeddings_cache.sqlite3"
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    content_hash TEXT PRIMARY KEY,
                    embedding_blob BLOB
                )
                """
            )
            conn.commit()

    def get_many(self, hashes: List[str], expected_dim: Optional[int] = None) -> dict:
        """Retrieves cached embeddings for given hashes, ignoring mismatched dimensions."""
        res = {}
        with sqlite3.connect(self.db_path) as conn:
            # Batch in chunks of 500 to stay well under SQLite parameter limit
            for i in range(0, len(hashes), 500):
                batch = hashes[i : i + 500]
                placeholders = ",".join("?" for _ in batch)
                cursor = conn.execute(
                    f"SELECT content_hash, embedding_blob FROM embedding_cache WHERE content_hash IN ({placeholders})",
                    batch,
                )
                for h, blob in cursor.fetchall():
                    vec = np.frombuffer(blob, dtype=np.float32)
                    if expected_dim is None or len(vec) == expected_dim:
                        res[h] = vec
        return res


    def set_many(self, items: List[tuple]) -> None:
        """Caches a batch of (hash, numpy_vector) pairs."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embedding_cache (content_hash, embedding_blob) VALUES (?, ?)",
                [(h, vec.astype(np.float32).tobytes()) for h, vec in items],
            )
            conn.commit()


class LocalSentenceTransformerEmbedder(BaseEmbedder):
    """Production local embedder using sentence-transformers with PyTorch or ONNX Runtime.
    
    Features:
    - Supports native ONNX Runtime backend and INT8 Quantization (cuts RAM/disk by 4x, 2x faster on CPU)
    - Auto-detects CUDA / MPS / CPU
    - Normalizes embeddings for cosine similarity via dot product
    - Built-in SQLite disk cache by content hash
    - Batch encoding with progress tracking
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "auto",
        backend: str = "default",  # "default" (PyTorch) or "onnx" (ONNX Runtime)
        quantization: str = "none",  # "none", "int8", "fp16"
        batch_size: int = 8,
        cache_dir: Optional[str] = None,
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.backend = backend
        self.quantization = quantization
        self.batch_size = batch_size
        self.normalize = normalize
        self.cache = DiskEmbeddingCache(cache_dir) if cache_dir else None

        logger.info(
            f"Loading local embedding model: {model_name} "
            f"(backend={backend}, quant={quantization}, device={device})..."
        )
        from sentence_transformers import SentenceTransformer
        import torch

        # Device detection
        if device == "auto":
            if torch.cuda.is_available():
                actual_device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                actual_device = "mps"
            else:
                actual_device = "cpu"
        else:
            actual_device = device

        logger.info(f"Using device '{actual_device}' for embedding.")

        self._ort_session = None
        self._tokenizer = None

        # Check for local ONNX model file
        local_onnx_path = Path("models/bge-m3-onnx/model_int8.onnx")
        if not local_onnx_path.exists():
            # Also check user Downloads folder
            dl_path = Path(os.path.expanduser("~")) / "Downloads" / "model BAAI-bge-m3-int8.onnx"
            if dl_path.exists():
                local_onnx_path = dl_path

        if backend == "onnx" and local_onnx_path.exists():
            try:
                import onnxruntime as ort
                from transformers import XLMRobertaTokenizerFast
                logger.info(f"Loading direct local ONNX model from: {local_onnx_path}")
                self._ort_session = ort.InferenceSession(str(local_onnx_path), providers=["CPUExecutionProvider"])
                try:
                    self._tokenizer = XLMRobertaTokenizerFast.from_pretrained("BAAI/bge-m3", local_files_only=True)
                except Exception:
                    # Resolve an already cached tokenizer without ever reaching
                    # Hugging Face. The ONNX weights are local, so startup must
                    # remain fully offline as well.
                    cache_roots = [
                        Path(os.getenv("HF_HOME", "")) / "hub" if os.getenv("HF_HOME") else None,
                        Path(os.path.expanduser("~")) / ".cache" / "huggingface" / "hub",
                    ]
                    snapshots = []
                    for root in cache_roots:
                        if root:
                            snapshots.extend((root / "models--BAAI--bge-m3" / "snapshots").glob("*"))
                    local_snapshot = next(
                        (p for p in sorted(snapshots, reverse=True)
                         if (p / "tokenizer_config.json").exists() and (p / "sentencepiece.bpe.model").exists()),
                        None,
                    )
                    if not local_snapshot:
                        raise RuntimeError(
                            "Local tokenizer files for BAAI/bge-m3 were not found. "
                            "Place tokenizer_config.json, tokenizer.json and sentencepiece.bpe.model "
                            "in the Hugging Face cache; network download is disabled."
                        )
                    self._tokenizer = XLMRobertaTokenizerFast.from_pretrained(str(local_snapshot), local_files_only=True)
                self._dim = 1024
                logger.info(f"Direct ONNX model loaded successfully! Dimension: {self._dim}")
                return
            except Exception as e:
                # Never silently fall back to the multi-GB PyTorch BGE-M3 model
                # when the user explicitly selected the lightweight local ONNX model.
                raise RuntimeError(
                    f"Could not load the local 570MB ONNX embedding model at {local_onnx_path}: {e}. "
                    "The full BAAI/bge-m3 model fallback is disabled to protect RAM."
                ) from e

        # Fallback to SentenceTransformer
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=actual_device)
        self._dim = self.model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dim


    def embed_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        # 1. Check disk cache if enabled
        hashes = [compute_sha256(t) for t in texts]
        cached_map = self.cache.get_many(hashes, expected_dim=self.dimension) if self.cache else {}


        uncached_indices = [i for i, h in enumerate(hashes) if h not in cached_map]
        uncached_texts = [texts[i] for i in uncached_indices]

        # 2. Compute embeddings for missing texts
        if uncached_texts:
            logger.info(f"Embedding {len(uncached_texts)} new texts (cached: {len(cached_map)})...")
            
            if self._ort_session is not None:
                # Direct ONNX inference
                batches = [uncached_texts[i:i + self.batch_size] for i in range(0, len(uncached_texts), self.batch_size)]
                new_embs_list = []
                for b in batches:
                    tok_out = self._tokenizer(b, padding=True, truncation=True, max_length=512, return_tensors="np")
                    ort_inputs = dict(tok_out)
                    if "token_type_ids" not in ort_inputs:
                        ort_inputs["token_type_ids"] = np.zeros_like(ort_inputs["input_ids"])
                    out = self._ort_session.run(None, ort_inputs)
                    # CLS token embedding: out[0][:, 0]
                    cls_vecs = out[0][:, 0].astype(np.float32)
                    if self.normalize:
                        norms = np.linalg.norm(cls_vecs, axis=1, keepdims=True)
                        cls_vecs = cls_vecs / np.clip(norms, a_min=1e-12, a_max=None)
                    new_embs_list.append(cls_vecs)
                new_embeddings = np.vstack(new_embs_list) if new_embs_list else np.empty((0, self.dimension), dtype=np.float32)
            else:
                new_embeddings = self.model.encode(
                    sentences=uncached_texts,
                    batch_size=self.batch_size,
                    show_progress_bar=show_progress,
                    normalize_embeddings=self.normalize,
                    convert_to_numpy=True,
                )

            # Store to disk cache
            if self.cache:
                to_cache = [
                    (hashes[idx], new_embeddings[i])
                    for i, idx in enumerate(uncached_indices)
                ]
                self.cache.set_many(to_cache)

            # Assemble full result
            results = np.zeros((len(texts), self.dimension), dtype=np.float32)
            for i, h in enumerate(hashes):
                if h in cached_map:
                    results[i] = cached_map[h]
                else:
                    new_idx = uncached_indices.index(i)
                    results[i] = new_embeddings[new_idx]
            return results

        # All were cached
        results = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, h in enumerate(hashes):
            results[i] = cached_map[h]
        return results

    def embed_query(self, text: str) -> np.ndarray:
        """Embeds a single query string without disk caching."""
        if self._ort_session is not None:
            tok_out = self._tokenizer([text], padding=True, truncation=True, max_length=512, return_tensors="np")
            ort_inputs = dict(tok_out)
            if "token_type_ids" not in ort_inputs:
                ort_inputs["token_type_ids"] = np.zeros_like(ort_inputs["input_ids"])
            out = self._ort_session.run(None, ort_inputs)
            cls_vec = out[0][0, 0].astype(np.float32)
            if self.normalize:
                norm = np.linalg.norm(cls_vec)
                cls_vec = cls_vec / max(norm, 1e-12)
            return cls_vec

        vec = self.model.encode(
            sentences=[text],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return vec[0]



class OllamaEmbedder(BaseEmbedder):
    """Embedder using local Ollama instance (e.g. nomic-embed-text or bge-m3 via Ollama)."""

    def __init__(self, model_name: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self._dim = 768 # Default fallback, dynamically fetched on first call

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        import httpx
        from tqdm import tqdm

        embeddings = []
        iterator = tqdm(texts, desc="Ollama Embedding") if show_progress else texts

        with httpx.Client(base_url=self.base_url, timeout=60.0) as client:
            for t in iterator:
                resp = client.post("/api/embeddings", json={"model": self.model_name, "prompt": t})
                resp.raise_for_status()
                vec = resp.json().get("embedding", [])
                self._dim = len(vec)
                embeddings.append(vec)

        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text], show_progress=False)[0]
