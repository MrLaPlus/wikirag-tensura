import os
from typing import Optional
from wikirag.llm.anthropic import AnthropicProvider
from wikirag.llm.base import BaseLLMProvider, LLMResponse
from wikirag.llm.gemini import GeminiProvider
from wikirag.llm.llamacpp import LlamaCppProvider
from wikirag.llm.ollama import OllamaProvider
from wikirag.llm.openai import OpenAIProvider
from wikirag.llm.openrouter import OpenRouterProvider


def get_llm_provider(
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> BaseLLMProvider:
    """Factory function resolving the requested LLM provider dynamically at runtime.
    
    Can be configured via:
    - CLI parameter (e.g. --llm ollama:llama3.1 or --llm openrouter:meta-llama/llama-3.1-8b-instruct:free)
    - Environment variables (DEFAULT_LLM_PROVIDER, DEFAULT_LLM_MODEL, OPENROUTER_API_KEY)
    - Project YAML configuration
    """
    prov = (provider_name or os.getenv("DEFAULT_LLM_PROVIDER", "ollama")).lower()
    model = model_name or os.getenv("DEFAULT_LLM_MODEL")

    if prov == "ollama":
        return OllamaProvider(
            model=model or "llama3.1:8b",
            base_url=base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    elif prov == "openrouter":
        return OpenRouterProvider(
            model=model or "meta-llama/llama-3.1-8b-instruct:free",
            api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
            base_url=base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )
    elif prov == "gemini":
        return GeminiProvider(model=model or "gemini-2.5-flash", api_key=api_key)
    elif prov == "openai":
        return OpenAIProvider(model=model or "gpt-4o-mini", api_key=api_key, base_url=base_url)
    elif prov == "anthropic":
        return AnthropicProvider(model=model or "claude-3-5-sonnet-20241022", api_key=api_key)
    elif prov == "llamacpp":
        return LlamaCppProvider(model_path=model or "")
    else:
        raise ValueError(
            f"Unknown LLM provider: '{prov}'. Supported: ollama, openrouter, gemini, openai, anthropic, llamacpp"
        )


__all__ = [
    "BaseLLMProvider",
    "LLMResponse",
    "OllamaProvider",
    "OpenRouterProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "LlamaCppProvider",
    "get_llm_provider",
]
