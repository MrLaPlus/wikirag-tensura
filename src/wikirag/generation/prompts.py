from typing import Any, Dict, List

DEFAULT_SYSTEM_PROMPT = """You are an accurate, encyclopedic assistant specialized in the wiki domain.
You answer user questions strictly based on the provided context passages.

CRITICAL INSTRUCTIONS:
1. Grounding: Answer ONLY using facts directly stated in the context. Never guess, assume, or hallucinate.
2. Low Confidence: If the context does not contain enough facts to answer the question, state clearly:
   "I do not have sufficient information in the knowledge base to answer this question accurately."
   (If the user asked in Thai: "ไม่มีข้อมูลที่เพียงพอในฐานความรู้ที่จะตอบคำถามนี้ได้อย่างถูกต้อง")
3. Citations: Add inline citations [1], [2] next to claims, referencing the sources provided.
4. Language Matching: Always reply in the exact language the user used (e.g. Thai if asked in Thai, English if asked in English).
5. License Attribution: The footer attribution is required by license and will be handled by the system.
"""

CC_ATTRIBUTION_TEMPLATE = "\n\n---\n*Content derived from {wiki_title} ({wiki_url}) under {license} license.*"


def build_rag_prompt(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Formats context chunks into a structured prompt with numbered citations."""
    context_blocks = []
    for i, c in enumerate(retrieved_chunks, 1):
        entity = c.get("entity", "Unknown")
        section = c.get("section_path", "General")
        url = c.get("canonical_url", "")
        text = c.get("chunk_text", "").strip()
        context_blocks.append(
            f"--- Source [{i}]: {entity} (Section: {section}) ---\nURL: {url}\n{text}\n"
        )

    all_context = "\n".join(context_blocks)
    return (
        f"KNOWLEDGE BASE CONTEXT:\n\n{all_context}\n\n"
        f"USER QUESTION: {query}\n\n"
        f"Please provide an accurate, grounded answer citing sources [1], [2], etc.:"
    )
