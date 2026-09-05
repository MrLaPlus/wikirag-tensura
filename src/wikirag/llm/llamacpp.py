from typing import Iterator, Optional
from wikirag.llm.base import BaseLLMProvider, LLMResponse


class LlamaCppProvider(BaseLLMProvider):
    """LlamaCpp local provider using llama-cpp-python.
    
    (Note: As agreed during the Discovery Phase ADR, concrete runtime is scheduled for Phase 3,
    as Ollama wraps llama.cpp with superior operational tooling and zero extra build complexity).
    """

    def __init__(self, model_path: str):
        self.model_path = model_path

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        raise NotImplementedError(
            "LlamaCppProvider is scheduled for Phase 3. Please use OllamaProvider, "
            "which wraps llama.cpp locally without needing compiled C++ wheel setup."
        )

    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        raise NotImplementedError(
            "LlamaCppProvider is scheduled for Phase 3. Please use OllamaProvider."
        )
