from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional


class BaseConnector(ABC):
    """Abstract Base Class for all WikiRAG data source connectors.
    
    Any new data source (MediaWiki, Generic Web, Local PDF/Markdown) must implement
    this interface. Ingestion pipelines interact solely with BaseConnector, ensuring
    zero core changes when adding new sources.
    """

    @abstractmethod
    def crawl_all(self, resume: bool = True) -> Iterator[Dict[str, Any]]:
        """Enumerates and fetches all documents from the source.
        
        Yields raw page records containing title, page_id, raw_content, and metadata.
        """
        pass

    @abstractmethod
    def sync_incremental(self, watermark: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        """Fetches only pages updated after the given watermark timestamp."""
        pass

    @abstractmethod
    def build_alias_map(self) -> Dict[str, List[str]]:
        """Constructs a mapping of canonical_title -> [aliases, redirects]."""
        pass
