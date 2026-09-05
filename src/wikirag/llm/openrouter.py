import json
import os
import time
from typing import Iterator, Optional
import httpx
from wikirag.llm.base import BaseLLMProvider, LLMResponse


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter Cloud Provider supporting hundreds of open and proprietary models.
    
    Uses standard HTTP/REST with streaming SSE via httpx (zero extra dependencies).
    Compatible with any model on OpenRouter:
      e.g. minimax/minimax-m3:free, meta-llama/llama-3.1-8b-instruct:free, etc.
    """

    def __init__(
        self,
        model: str = "minimax/minimax-m3:free",
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        site_url: str = "http://localhost:8000",
        site_name: str = "WikiRAG Tensura",
        retry_enabled: bool = True,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. Please provide your API Key in Settings or set the OPENROUTER_API_KEY environment variable."
            )
        self.base_url = base_url.rstrip("/")
        self.site_url = site_url
        self.site_name = site_name
        self.retry_enabled = retry_enabled

    def _get_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name,
            "Content-Type": "application/json",
        }

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After", "")
        try:
            return min(30.0, max(1.0, float(retry_after)))
        except ValueError:
            return min(30.0, 1.5 ** attempt)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        with httpx.Client(timeout=60.0) as client:
            data = None
            for attempt in range(3 if self.retry_enabled else 1):
                resp = client.post(f"{self.base_url}/chat/completions", headers=self._get_headers(), json=payload)
                if resp.status_code not in (429, 502, 503, 504) or attempt == (2 if self.retry_enabled else 0):
                    resp.raise_for_status()
                    data = resp.json()
                    break
                time.sleep(self._retry_delay(resp, attempt))

        choice = data.get("choices", [{}])[0]
        text = choice.get("message", {}).get("content", "")
        usage = data.get("usage", {})

        return LLMResponse(
            text=text,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            model=self.model,
        )

    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        with httpx.Client(timeout=60.0) as client:
            for attempt in range(3 if self.retry_enabled else 1):
                with client.stream("POST", f"{self.base_url}/chat/completions", headers=self._get_headers(), json=payload) as response:
                    if response.status_code in (429, 502, 503, 504) and self.retry_enabled and attempt < 2:
                        time.sleep(self._retry_delay(response, attempt))
                        continue
                    response.raise_for_status()
                    for line in response.iter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                return
                            try:
                                data = json.loads(data_str)
                                delta = data.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content")
                                if content:
                                    yield content
                            except Exception:
                                continue
                    return
