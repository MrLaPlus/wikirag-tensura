import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from wikirag.config import WikiRagProjectConfig
from wikirag.connectors.base import BaseConnector
from wikirag.utils.http import PoliteHttpClient
from wikirag.utils.logging import get_logger

logger = get_logger(__name__)


class MediaWikiConnector(BaseConnector):
    """Production-grade MediaWiki connector supporting any MediaWiki instance (Wikipedia, Fandom, Wikia).
    
    Implements:
    - Page enumeration with continue tokens
    - Batched content retrieval (up to 50 titles/batch with auto GET -> POST)
    - Checkpointed resumability (never restarts from page 0 on crash)
    - Full alias/redirect graph map extraction
    - Incremental sync via recentchanges watermark
    - Rate-limiting & polite retry mechanism
    """

    def __init__(self, config: WikiRagProjectConfig):
        self.config = config
        self.src = config.source
        self.storage = config.storage
        self.client = PoliteHttpClient(
            user_agent=self.src.user_agent,
            delay_seconds=self.src.request_delay_seconds,
        )
        self.api_url = self.src.api_url
        self.base_url = self.src.base_url.rstrip("/") if self.src.base_url else ""
        self.checkpoint_file = Path(self.storage.state_file)

    def _load_state(self) -> Dict[str, Any]:
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read state file: {e}. Starting fresh.")
        return {"last_page_token": None, "watermark": None, "processed_count": 0}

    def _save_state(self, state: Dict[str, Any]) -> None:
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def crawl_all(self, resume: bool = True) -> Iterator[Dict[str, Any]]:
        """Enumerates and batches all articles in the configured namespace with resumability."""
        state = self._load_state() if resume else {"last_page_token": None, "processed_count": 0}
        apcontinue = state.get("last_page_token")

        if apcontinue:
            logger.info(f"Resuming MediaWiki crawl from token: {apcontinue} (Processed so far: {state.get('processed_count', 0)})")
        else:
            logger.info("Starting fresh crawl of all MediaWiki articles...")

        page_buffer: List[str] = []
        batch_size = self.src.batch_size
        checkpoint_interval = self.src.checkpoint_interval
        total_fetched = state.get("processed_count", 0)

        while True:
            params: Dict[str, Any] = {
                "action": "query",
                "list": "allpages",
                "apnamespace": self.src.namespace,
                "apfilterredir": "nonredirects",
                "aplimit": "max",
                "format": "json",
            }
            if apcontinue:
                params["apcontinue"] = apcontinue

            resp = self.client.get(self.api_url, params=params).json()
            allpages = resp.get("query", {}).get("allpages", [])

            for page in allpages:
                page_buffer.append(page["title"])
                if len(page_buffer) >= batch_size:
                    yield from self._fetch_batch_content(page_buffer)
                    total_fetched += len(page_buffer)
                    page_buffer.clear()

                    # Save checkpoint every checkpoint_interval pages
                    if total_fetched % checkpoint_interval < batch_size:
                        state["last_page_token"] = apcontinue
                        state["processed_count"] = total_fetched
                        self._save_state(state)
                        logger.info(f"Checkpoint saved at {total_fetched} pages.")

            # Check for MediaWiki continue token
            if "continue" in resp and "apcontinue" in resp["continue"]:
                apcontinue = resp["continue"]["apcontinue"]
            else:
                break

        # Flush remaining pages in buffer
        if page_buffer:
            yield from self._fetch_batch_content(page_buffer)
            total_fetched += len(page_buffer)
            page_buffer.clear()

        # Mark crawl completed
        state["last_page_token"] = None
        state["processed_count"] = total_fetched
        state["watermark"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._save_state(state)
        logger.info(f"Crawl completed! Total pages fetched: {total_fetched}")

    def _fetch_batch_content(self, titles: List[str]) -> Iterator[Dict[str, Any]]:
        """Fetches revisions and categories for up to 50 titles in a single batch call.
        
        Automatically toggles between GET and POST when query string exceeds ~2000 chars.
        """
        titles_param = "|".join(titles)
        params: Dict[str, Any] = {
            "action": "query",
            "prop": "revisions|categories",
            "rvprop": "content|timestamp|ids",
            "rvslots": "main",
            "cllimit": "max",
            "formatversion": "2",
            "format": "json",
        }

        # URL length safety check
        encoded_query = "&".join(f"{k}={v}" for k, v in params.items()) + f"&titles={titles_param}"
        if len(encoded_query) > 2000:
            post_data = {**params, "titles": titles_param}
            resp = self.client.post(self.api_url, data=post_data).json()
        else:
            get_params = {**params, "titles": titles_param}
            resp = self.client.get(self.api_url, params=get_params).json()

        pages = resp.get("query", {}).get("pages", [])
        for page in pages:
            # Skip missing pages or pages with no revision slots
            if page.get("missing", False):
                continue

            revisions = page.get("revisions", [])
            content = ""
            rev_id = 0
            rev_timestamp = ""
            if revisions:
                main_slot = revisions[0].get("slots", {}).get("main", {})
                content = main_slot.get("content", "")
                rev_id = revisions[0].get("revid", 0)
                rev_timestamp = revisions[0].get("timestamp", "")

            # Categories cleanup
            raw_cats = page.get("categories", [])
            categories = [c.get("title", "").replace("Category:", "") for c in raw_cats if c.get("title")]

            canonical_url = f"{self.base_url}/{page['title'].replace(' ', '_')}" if self.base_url else ""

            yield {
                "page_id": page["pageid"],
                "title": page["title"],
                "canonical_url": canonical_url,
                "revision_id": rev_id,
                "revision_timestamp": rev_timestamp,
                "categories": categories,
                "raw_wikitext": content,
                "fetched_at": time.time(),
            }

    def build_alias_map(self) -> Dict[str, List[str]]:
        """Scrapes all redirects to construct canonical_title -> [aliases] mapping.
        
        Example: 'Rimuru Tempest' -> ['Slime', 'Satoru Mikami', 'Demon Lord Rimuru']
        Saves result to config.storage.alias_map_path.
        """
        logger.info("Constructing Alias & Redirect Map from wiki...")
        alias_to_target: Dict[str, str] = {}
        apcontinue = None

        while True:
            params: Dict[str, Any] = {
                "action": "query",
                "list": "allpages",
                "apnamespace": self.src.namespace,
                "apfilterredir": "redirects",
                "aplimit": "max",
                "format": "json",
            }
            if apcontinue:
                params["apcontinue"] = apcontinue

            resp = self.client.get(self.api_url, params=params).json()
            pages = resp.get("query", {}).get("allpages", [])
            redirect_titles = [p["title"] for p in pages]

            # Resolve redirect targets in batches of 50
            for i in range(0, len(redirect_titles), 50):
                batch = redirect_titles[i : i + 50]
                target_params = {
                    "action": "query",
                    "titles": "|".join(batch),
                    "redirects": "1",
                    "format": "json",
                }
                t_resp = self.client.get(self.api_url, params=target_params).json()
                resolved = t_resp.get("query", {}).get("redirects", [])
                for r in resolved:
                    alias_to_target[r["from"]] = r["to"]

            if "continue" in resp and "apcontinue" in resp["continue"]:
                apcontinue = resp["continue"]["apcontinue"]
            else:
                break

        # Invert map to canonical -> [aliases]
        canonical_to_aliases: Dict[str, List[str]] = {}
        for alias, canonical in alias_to_target.items():
            canonical_to_aliases.setdefault(canonical, []).append(alias)

        out_path = Path(self.storage.alias_map_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(canonical_to_aliases, f, indent=2, ensure_ascii=False)

        logger.info(f"Alias map constructed! Saved {len(canonical_to_aliases)} canonical entities with aliases to {out_path}")
        return canonical_to_aliases

    def sync_incremental(self, watermark: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        """Fetches only pages modified after watermark using list=recentchanges."""
        state = self._load_state()
        since = watermark or state.get("watermark")
        if not since:
            logger.info("No prior watermark found. Falling back to full crawl.")
            yield from self.crawl_all(resume=True)
            return

        logger.info(f"Performing incremental sync for edits since: {since}")
        rccontinue = None
        changed_titles = set()

        while True:
            params: Dict[str, Any] = {
                "action": "query",
                "list": "recentchanges",
                "rcnamespace": self.src.namespace,
                "rcend": since,
                "rclimit": "max",
                "format": "json",
            }
            if rccontinue:
                params["rccontinue"] = rccontinue

            resp = self.client.get(self.api_url, params=params).json()
            rc_entries = resp.get("query", {}).get("recentchanges", [])

            for entry in rc_entries:
                if entry.get("type") in ("edit", "new"):
                    changed_titles.add(entry["title"])

            if "continue" in resp and "rccontinue" in resp["continue"]:
                rccontinue = resp["continue"]["rccontinue"]
            else:
                break

        logger.info(f"Found {len(changed_titles)} changed articles since {since}")
        titles_list = list(changed_titles)
        for i in range(0, len(titles_list), self.src.batch_size):
            batch = titles_list[i : i + self.src.batch_size]
            yield from self._fetch_batch_content(batch)

        # Update watermark to now
        state["watermark"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._save_state(state)
