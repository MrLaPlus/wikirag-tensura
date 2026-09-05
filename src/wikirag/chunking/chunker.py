import json
import re
from typing import Iterator, List, Optional
from wikirag.config import ChunkingConfig
from wikirag.parser.models import ChunkRecord, ParsedPage
from wikirag.utils.hashing import compute_chunk_id, compute_sha256


class SectionAwareChunker:
    """Chunks structured wiki pages by respecting section boundaries, sentence integrity,
    and injecting contextual headers to dramatically boost retrieval recall on fandom queries.
    """

    def __init__(self, config: ChunkingConfig, project_name: str):
        self.config = config
        self.project_name = project_name

    def _split_into_sentences(self, text: str) -> List[str]:
        """Splits text on sentence boundaries without breaking mid-sentence."""
        # Split on period, exclamation, or question mark followed by space or newline
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _build_contextual_header(
        self,
        entity: str,
        aliases: List[str],
        section_path: str,
        categories: List[str],
    ) -> str:
        """Constructs the high-signal contextual header prepended before embedding.
        
        Example:
        Entity: Rimuru Tempest | Aliases: Slime, Satoru Mikami | Section: Appearance > Slime Form | Categories: Characters, Slimes
        """
        alias_str = ", ".join(aliases[:5]) if aliases else "None"
        cat_str = ", ".join(categories[:6]) if categories else "General"
        return (
            f"Entity: {entity} | Aliases: {alias_str} | "
            f"Section: {section_path} | Categories: {cat_str}\n"
        )

    def chunk_page(self, page: ParsedPage) -> Iterator[ChunkRecord]:
        """Processes a ParsedPage into an iterator of indexed ChunkRecords."""
        entity = page.title
        aliases = page.aliases
        categories = page.categories
        canonical_url = page.canonical_url
        fetched_at = page.fetched_at

        # 1. Dedicated Infobox Chunk (if infobox exists)
        if self.config.separate_infobox_chunk and page.infobox:
            infobox_dict = page.infobox.model_dump(exclude_none=True)
            # Create a structured factual summary text from the infobox
            facts = [f"{k.capitalize()}: {v}" for k, v in infobox_dict.items() if k != "raw_fields" and v]
            if "raw_fields" in infobox_dict:
                for rk, rv in infobox_dict["raw_fields"].items():
                    if rk not in infobox_dict:
                        facts.append(f"{rk.capitalize()}: {rv}")

            raw_infobox_text = "\n".join(facts)
            header = self._build_contextual_header(entity, aliases, "Infobox & Overview", categories)
            full_infobox_text = f"{header}\n[INFOBOX FACT SHEET]\n{raw_infobox_text}"
            content_hash = compute_sha256(full_infobox_text)

            yield ChunkRecord(
                chunk_id=compute_chunk_id(entity, "Infobox", 0, full_infobox_text),
                entity=entity,
                canonical_url=canonical_url,
                section_path="Infobox",
                chunk_type="infobox",
                chunk_text=full_infobox_text,
                raw_text=raw_infobox_text,
                infobox_json=json.dumps(infobox_dict, ensure_ascii=False),
                categories=categories,
                aliases=aliases,
                content_hash=content_hash,
                source_project=self.project_name,
                fetched_at=fetched_at,
            )

        # 2. Chunk Lead Section
        if page.lead_section:
            yield from self._chunk_text_block(
                text=page.lead_section,
                section_path="Introduction",
                entity=entity,
                aliases=aliases,
                categories=categories,
                canonical_url=canonical_url,
                fetched_at=fetched_at,
            )

        # 3. Chunk Body Sections
        for sec in page.sections:
            yield from self._chunk_text_block(
                text=sec.content,
                section_path=sec.path,
                entity=entity,
                aliases=aliases,
                categories=categories,
                canonical_url=canonical_url,
                fetched_at=fetched_at,
            )

    def _chunk_text_block(
        self,
        text: str,
        section_path: str,
        entity: str,
        aliases: List[str],
        categories: List[str],
        canonical_url: str,
        fetched_at: float,
    ) -> Iterator[ChunkRecord]:
        """Slices a section body into target token chunks with sentence preservation and overlap."""
        sentences = self._split_into_sentences(text)
        if not sentences:
            return

        # Simple whitespace token estimation (~1.3 words per token average)
        def token_len(s: str) -> int:
            return max(1, int(len(s.split()) * 1.3))

        max_tokens = self.config.target_max_tokens
        overlap_tokens = int(max_tokens * self.config.overlap_percentage)

        current_sentences: List[str] = []
        current_len = 0
        chunk_idx = 0

        header = self._build_contextual_header(entity, aliases, section_path, categories) if self.config.prepend_contextual_header else ""

        for sent in sentences:
            s_len = token_len(sent)
            if current_len + s_len > max_tokens and current_sentences:
                # Flush chunk
                raw_chunk = " ".join(current_sentences)
                full_chunk_text = f"{header}\n{raw_chunk}".strip()
                content_hash = compute_sha256(full_chunk_text)

                yield ChunkRecord(
                    chunk_id=compute_chunk_id(entity, section_path, chunk_idx, full_chunk_text),
                    entity=entity,
                    canonical_url=canonical_url,
                    section_path=section_path,
                    chunk_type="text",
                    chunk_text=full_chunk_text,
                    raw_text=raw_chunk,
                    categories=categories,
                    aliases=aliases,
                    content_hash=content_hash,
                    source_project=self.project_name,
                    fetched_at=fetched_at,
                )
                chunk_idx += 1

                # Keep trailing sentences for overlap window
                overlap_accum: List[str] = []
                overlap_accum_len = 0
                for prev_sent in reversed(current_sentences):
                    prev_len = token_len(prev_sent)
                    if overlap_accum_len + prev_len <= overlap_tokens:
                        overlap_accum.insert(0, prev_sent)
                        overlap_accum_len += prev_len
                    else:
                        break
                current_sentences = overlap_accum
                current_len = overlap_accum_len

            current_sentences.append(sent)
            current_len += s_len

        # Flush final remaining sentences
        if current_sentences:
            raw_chunk = " ".join(current_sentences)
            full_chunk_text = f"{header}\n{raw_chunk}".strip()
            content_hash = compute_sha256(full_chunk_text)

            yield ChunkRecord(
                chunk_id=compute_chunk_id(entity, section_path, chunk_idx, full_chunk_text),
                entity=entity,
                canonical_url=canonical_url,
                section_path=section_path,
                chunk_type="text",
                chunk_text=full_chunk_text,
                raw_text=raw_chunk,
                categories=categories,
                aliases=aliases,
                content_hash=content_hash,
                source_project=self.project_name,
                fetched_at=fetched_at,
            )
