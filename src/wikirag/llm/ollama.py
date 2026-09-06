import json
import os
from typing import Iterator, Optional
import httpx
from wikirag.llm.base import BaseLLMProvider, LLMResponse
from wikirag.utils.logging import get_logger

logger = get_logger(__name__)


class OllamaProvider(BaseLLMProvider):
    """Local LLM provider interfacing with the official Ollama REST API.
    
    Zero configuration, zero cost, completely offline.
    """

    def __init__(self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt or "",
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        timeout = float(os.getenv("WIKIRAG_LLM_TIMEOUT_SECONDS", "180"))
        with httpx.Client(base_url=self.base_url, timeout=timeout) as client:
            resp = client.post("/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()

        return LLMResponse(
            text=data.get("response", ""),
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            model=self.model,
        )

    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt or "",
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        timeout = float(os.getenv("WIKIRAG_LLM_TIMEOUT_SECONDS", "180"))
        with httpx.Client(base_url=self.base_url, timeout=timeout) as client:
            with client.stream("POST", "/api/generate", json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        yield chunk.get("response", "")
