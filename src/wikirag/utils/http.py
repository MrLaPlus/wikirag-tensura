import time
import httpx
from typing import Any, Dict, Optional
from wikirag.utils.logging import get_logger

logger = get_logger(__name__)


class PoliteHttpClient:
    """HTTP Client with polite rate-limiting, exponential backoff, and Retry-After handling.
    
    Fandom/MediaWiki APIs can return 429 Too Many Requests or 503 Service Unavailable
    under high concurrency. This wrapper ensures automatic backoff and respects rate limits.
    """

    def __init__(
        self,
        user_agent: str,
        delay_seconds: float = 0.5,
        max_retries: int = 5,
        timeout: float = 30.0,
    ):
        self.user_agent = user_agent
        self.delay_seconds = delay_seconds
        self.max_retries = max_retries
        self.timeout = timeout
        self._last_request_time: float = 0.0
        self.client = httpx.Client(
            headers={"User-Agent": self.user_agent},
            timeout=self.timeout,
            follow_redirects=True,
        )

    def _throttle(self) -> None:
        """Enforces minimum interval between consecutive outbound calls."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.delay_seconds:
            time.sleep(self.delay_seconds - elapsed)
        self._last_request_time = time.time()

    def request(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """Executes an HTTP request with exponential backoff on transient errors."""
        backoff = 1.0

        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                response = self.client.request(
                    method=method,
                    url=url,
                    params=params,
                    data=data,
                )

                # If rate-limited or transient server error, back off
                if response.status_code in (429, 503, 502, 504):
                    retry_after = response.headers.get("Retry-After")
                    sleep_time = float(retry_after) if retry_after and retry_after.isdigit() else backoff
                    logger.warning(
                        f"HTTP {response.status_code} received from {url}. Retrying in {sleep_time:.1f}s (Attempt {attempt}/{self.max_retries})"
                    )
                    time.sleep(sleep_time)
                    backoff *= 2.0
                    continue

                response.raise_for_status()
                return response

            except (httpx.RequestError, httpx.HTTPStatusError) as exc:
                if attempt == self.max_retries:
                    logger.error(f"HTTP request failed permanently after {self.max_retries} attempts: {exc}")
                    raise
                logger.warning(f"Request error: {exc}. Retrying in {backoff:.1f}s...")
                time.sleep(backoff)
                backoff *= 2.0

        raise RuntimeError(f"Failed to fetch {url} after {self.max_retries} retries")

    def get(self, url: str, params: Optional[Dict[str, Any]] = None) -> httpx.Response:
        return self.request("GET", url, params=params)

    def post(self, url: str, data: Optional[Dict[str, Any]] = None) -> httpx.Response:
        return self.request("POST", url, data=data)

    def close(self) -> None:
        self.client.close()
