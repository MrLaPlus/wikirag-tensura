import os
import time
from pathlib import Path
from typing import Dict, Iterator, List
from wikirag.config import WikiRagProjectConfig
from wikirag.connectors.base import BaseConnector
from wikirag.utils.logging import get_logger

logger = get_logger(__name__)


class LocalFilesConnector(BaseConnector):
    """Ingests local documents (.md, .txt, .json, .pdf) from a local folder."""

    def __init__(self, config: WikiRagProjectConfig):
        self.config = config
        self.root_dir = Path(getattr(config.source, "base_url", "./docs"))

    def crawl_all(self, resume: bool = True) -> Iterator[Dict[str, str]]:
        if not self.root_dir.exists():
            logger.warning(f"Local files path does not exist: {self.root_dir}")
            return

        supported_extensions = {".md", ".txt", ".markdown", ".json"}

        for p in self.root_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in supported_extensions:
                try:
                    with open(p, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                    title = p.stem.replace("_", " ").replace("-", " ").title()
                    yield {
                        "page_id": hash(str(p.resolve())) % 1_000_000,
                        "title": title,
                        "canonical_url": f"file:///{p.resolve()}",
                        "wikitext": content,
                        "fetched_at": time.time(),
                    }
                except Exception as e:
                    logger.warning(f"Could not read local file {p}: {e}")

    def sync_incremental(self) -> Iterator[Dict[str, str]]:
        yield from self.crawl_all(resume=True)

    def build_alias_map(self) -> Dict[str, List[str]]:
        return {}
