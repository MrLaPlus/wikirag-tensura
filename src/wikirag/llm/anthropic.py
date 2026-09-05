import os
from typing import Iterator, Optional
from wikirag.llm.base import BaseLLMProvider, LLMResponse


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude Provider (e.g. claude-3-5-sonnet, claude-3-haiku)."""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set in environment or config.")

        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        params = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            params["system"] = system_prompt

        res = self.client.messages.create(**params)
        text = res.content[0].text if res.content else ""
        return LLMResponse(
            text=text,
            prompt_tokens=res.usage.input_tokens if res.usage else 0,
            completion_tokens=res.usage.output_tokens if res.usage else 0,
            model=self.model,
        )

    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        params = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            params["system"] = system_prompt

        with self.client.messages.stream(**params) as stream:
            for text in stream.text_stream:
                yield text
