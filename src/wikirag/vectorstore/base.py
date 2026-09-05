from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np


class BaseVectorStore(ABC):
    """Abstract interface for all Vector Stores (LanceDB, Qdrant, Chroma).
    
    Guarantees that swapping the underlying vector engine never requires changes
    to the retrieval pipeline or ingestion jobs.
    """

    @abstractmethod
    def upsert_chunks(self, chunks: List[Dict[str, Any]], vectors: np.ndarray) -> int:
        """Inserts or updates chunks and their corresponding dense vectors idempotently.
        
        Returns the count of inserted records.
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Performs nearest-neighbor vector search returning matched chunk records with scores."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Returns the total number of indexed vectors."""
        pass
