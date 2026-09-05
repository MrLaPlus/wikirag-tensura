import os
from typing import Iterator, Optional
from wikirag.llm.base import BaseLLMProvider, LLMResponse


class OpenAIProvider(BaseLLMProvider):
    """OpenAI Cloud Provider (e.g. gpt-4o-mini, gpt-4o)."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment or config.")

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=base_url.rstrip("/") if base_url else None)
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

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

        res = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return LLMResponse(
            text=res.choices[0].message.content or "",
            prompt_tokens=res.usage.prompt_tokens if res.usage else 0,
            completion_tokens=res.usage.completion_tokens if res.usage else 0,
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

        stream_res = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream_res:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
