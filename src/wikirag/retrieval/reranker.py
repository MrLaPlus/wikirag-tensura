from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from wikirag.utils.logging import get_logger

logger = get_logger(__name__)


class BaseReranker(ABC):
    """Abstract interface for Cross-Encoder rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Reranks candidate documents given the query string."""
        pass


class LocalCrossEncoderReranker(BaseReranker):
    """Production Cross-Encoder reranker using sentence-transformers or FlagEmbedding.
    
    Default model: BAAI/bge-reranker-v2-m3 (Multilingual, covers Thai + English).
    Lightweight fallback: cross-encoder/ms-marco-MiniLM-L-6-v2 (English-only).
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "auto",
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self._model = None
        self.device = device

    def _load_model(self) -> None:
        if self._model is not None:
            return

        logger.info(f"Loading Cross-Encoder Reranker: {self.model_name}...")
        try:
            from sentence_transformers import CrossEncoder
            import torch

            actual_device = "cpu"
            if self.device == "auto":
                if torch.cuda.is_available():
                    actual_device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    actual_device = "mps"
            else:
                actual_device = self.device

            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=actual_device,
            )
            logger.info(f"Reranker loaded successfully on device '{actual_device}'.")
        except Exception as e:
            logger.error(f"Failed to load reranker {self.model_name}: {e}")
            raise

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        if not documents:
            return []

        self._load_model()

        # Build pair tuples: (query, candidate_text)
        pairs = [(query, doc.get("chunk_text", "")) for doc in documents]

        # Predict cross-entropy scores
        scores = self._model.predict(pairs)

        # Attach rerank score and sort
        scored_docs = []
        for i, doc in enumerate(documents):
            item = dict(doc)
            score_val = float(scores[i])
            item["rerank_score"] = score_val
            item["score"] = score_val
            scored_docs.append(item)

        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:top_k]
