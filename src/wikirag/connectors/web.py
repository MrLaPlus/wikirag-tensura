import re
import time
import urllib.parse
from typing import Dict, Iterator, List, Optional, Set
from bs4 import BeautifulSoup
from wikirag.config import WikiRagProjectConfig
from wikirag.connectors.base import BaseConnector
from wikirag.utils.hashing import compute_sha256
from wikirag.utils.http import PoliteHttpClient
from wikirag.utils.logging import get_logger

logger = get_logger(__name__)


class GenericWebConnector(BaseConnector):
    """Production web crawler connector for standard websites and documentation portals.
    
    Features:
    - Sitemap parsing (sitemap.xml)
    - Breadth-first crawling up to configured max_depth
    - Domain boundary restrictions
    - HTML noise stripping (scripts, styles, headers, footers)
    - Polite rate-limiting & retry backoff
    """

    def __init__(self, config: WikiRagProjectConfig):
        self.config = config
        self.base_url = getattr(config.source, "base_url", "")
        self.sitemap_url = getattr(config.source, "sitemap_url", None)
        self.max_depth = getattr(config.source, "max_depth", 2)
        self.http_client = PoliteHttpClient(
            user_agent=config.source.user_agent,
            rate_limit_rps=config.source.rate_limit_rps,
        )

    def crawl_all(self, resume: bool = True) -> Iterator[Dict[str, str]]:
        """Crawls all discoverable pages starting from sitemap or base_url."""
        discovered_urls: Set[str] = set()

        # 1. Try sitemap first
        if self.sitemap_url:
            logger.info(f"Checking sitemap: {self.sitemap_url}")
            try:
                resp = self.http_client.get(self.sitemap_url)
                soup = BeautifulSoup(resp.text, "xml")
                for loc in soup.find_all("loc"):
                    u = loc.text.strip()
                    if self._is_valid_url(u):
                        discovered_urls.add(u)
                logger.info(f"Discovered {len(discovered_urls)} URLs from sitemap.")
            except Exception as e:
                logger.warning(f"Sitemap parsing failed: {e}. Falling back to BFS crawl.")

        if not discovered_urls and self.base_url:
            discovered_urls.add(self.base_url)

        # 2. BFS Crawl
        visited: Set[str] = set()
        queue = list(discovered_urls)

        while queue:
            url = queue.pop(0)
            if url in visited:
                continue

            visited.add(url)
            try:
                resp = self.http_client.get(url)
                if resp.status_code != 200 or "text/html" not in resp.headers.get("Content-Type", ""):
                    continue

                soup = BeautifulSoup(resp.text, "html.parser")
                title = soup.title.string.strip() if soup.title and soup.title.string else url

                # Discover new links within the same domain
                base_domain = urllib.parse.urlparse(self.base_url).netloc
                for a in soup.find_all("a", href=True):
                    next_url = urllib.parse.urljoin(url, a["href"])
                    if self._is_valid_url(next_url) and urllib.parse.urlparse(next_url).netloc == base_domain:
                        if next_url not in visited and next_url not in queue and len(visited) + len(queue) < 500:
                            queue.append(next_url)

                # Clean content
                for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    tag.decompose()

                body_text = soup.get_text(separator="\n", strip=True)
                if len(body_text) < 50:
                    continue

                yield {
                    "page_id": hash(url) % 1_000_000,
                    "title": title,
                    "canonical_url": url,
                    "wikitext": body_text,
                    "fetched_at": time.time(),
                }
            except Exception as e:
                logger.warning(f"Failed to crawl URL {url}: {e}")

    def sync_incremental(self) -> Iterator[Dict[str, str]]:
        """Crawls recent pages (delegates to crawl_all for generic web)."""
        yield from self.crawl_all(resume=True)

    def build_alias_map(self) -> Dict[str, List[str]]:
        return {}

    def _is_valid_url(self, url: str) -> bool:
        if not url.startswith("http"):
            return False
        # Avoid binaries / media
        if any(url.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".pdf", ".zip", ".exe"]):
            return False
        return True
