from typing import Any, Dict, Iterator, List, Optional
from pydantic import BaseModel
from wikirag.config import WikiRagProjectConfig
from wikirag.generation.prompts import (
    CC_ATTRIBUTION_TEMPLATE,
    DEFAULT_SYSTEM_PROMPT,
    build_rag_prompt,
)
from wikirag.llm.base import BaseLLMProvider
from wikirag.utils.logging import get_logger

logger = get_logger(__name__)


class Citation(BaseModel):
    index: int
    entity: str
    section: str
    url: str
    score: float


class GenerationResult(BaseModel):
    answer: str
    citations: List[Citation]
    retrieved_chunks: List[Dict[str, Any]]
    confidence_passed: bool


class GroundedAnswerGenerator:
    """Answers user queries grounded strictly on retrieved chunks with inline citations
    and automatic license attribution.
    """

    def __init__(self, config: WikiRagProjectConfig, llm_provider: BaseLLMProvider):
        self.config = config
        self.llm = llm_provider
        self.system_prompt = (
            self.config.llm.system_prompt or DEFAULT_SYSTEM_PROMPT
        )
        self.min_confidence_score = 0.35 # Below this, mark low confidence

    def _build_attribution(self) -> str:
        p = self.config.project
        s = self.config.source
        return CC_ATTRIBUTION_TEMPLATE.format(
            wiki_title=p.title,
            wiki_url=s.base_url or s.api_url,
            license=p.license,
        )

    def _extract_citations(self, chunks: List[Dict[str, Any]]) -> List[Citation]:
        citations = []
        for i, c in enumerate(chunks, 1):
            citations.append(
                Citation(
                    index=i,
                    entity=c.get("entity", "Unknown"),
                    section=c.get("section_path", "General"),
                    url=c.get("canonical_url", ""),
                    score=c.get("score", 0.0),
                )
            )
        return citations

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> GenerationResult:
        """Executes non-streaming generation with grounding and attribution."""
        citations = self._extract_citations(retrieved_chunks)

        # Check confidence: if no chunks or highest score is below threshold
        max_score = max([c.score for c in citations], default=0.0)
        confidence_passed = max_score >= self.min_confidence_score and len(retrieved_chunks) > 0

        temp = temperature if temperature is not None else self.config.llm.temperature
        rag_prompt = build_rag_prompt(query, retrieved_chunks)

        llm_resp = self.llm.generate(
            prompt=rag_prompt,
            system_prompt=self.system_prompt,
            temperature=temp,
            max_tokens=max_output_tokens or self.config.llm.max_output_tokens,
        )

        full_answer = llm_resp.text.strip() + self._build_attribution()

        return GenerationResult(
            answer=full_answer,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            confidence_passed=confidence_passed,
        )

    def stream_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        """Streams generated tokens and yields attribution suffix at completion."""
        rag_prompt = build_rag_prompt(query, retrieved_chunks)
        temp = temperature if temperature is not None else self.config.llm.temperature

        for token in self.llm.stream(
            prompt=rag_prompt,
            system_prompt=self.system_prompt,
            temperature=temp,
            max_tokens=max_output_tokens or self.config.llm.max_output_tokens,
        ):
            yield token

        yield self._build_attribution()
