import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import mwparserfromhell
from wikirag.parser.infobox import clean_wikitext_value, extract_infobox
from wikirag.parser.models import InfoboxData, ParsedPage, SectionData
from wikirag.utils.logging import get_logger

logger = get_logger(__name__)


class WikitextParser:
    """Production wikitext parser for MediaWiki and Fandom articles.
    
    Transforms noisy raw wikitext into structured clean text preserving:
    - Section tree hierarchy (H2 -> H3 -> H4)
    - Extracted Infobox key-value pairs
    - Outbound entity links for graph construction
    - Quarantine for malformed/failed articles
    """

    def __init__(self, failed_pages_path: Optional[str] = None):
        self.failed_pages_path = Path(failed_pages_path) if failed_pages_path else None
        if self.failed_pages_path:
            self.failed_pages_path.parent.mkdir(parents=True, exist_ok=True)

    def _quarantine(self, raw_record: Dict[str, Any], error: Exception) -> None:
        """Appends unparseable or erroring articles to failed_pages.jsonl for auditing."""
        if not self.failed_pages_path:
            return
        payload = {
            "title": raw_record.get("title"),
            "page_id": raw_record.get("page_id"),
            "error": str(error),
            "raw_wikitext": raw_record.get("raw_wikitext"),
        }
        with open(self.failed_pages_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def parse_page(self, raw_record: Dict[str, Any], alias_map: Optional[Dict[str, List[str]]] = None) -> Optional[ParsedPage]:
        """Parses a raw MediaWiki API response record into a clean ParsedPage."""
        title = raw_record.get("title", "")
        raw_text = raw_record.get("raw_wikitext", "")

        if not raw_text or not raw_text.strip():
            return None

        try:
            # Parse raw text using mwparserfromhell C-tokenizer
            wikicode = mwparserfromhell.parse(raw_text)

            # 1. Extract Infobox before stripping templates
            infobox = extract_infobox(wikicode)

            # 2. Extract outbound wiki links (for entity graph)
            wiki_links: List[str] = []
            for link in wikicode.filter_wikilinks():
                target = str(link.title).strip()
                # Exclude image/category/interwiki links
                if not any(target.startswith(prefix) for prefix in ("Category:", "File:", "Image:", "es:", "de:", "ru:", "zh:", "pt-br:", "vi:", "id:", "pl:", "tr:")):
                    wiki_links.append(target)

            # 3. Strip noise: comments, navboxes, references, galleries
            for tag in wikicode.filter_tags():
                tag_name = str(tag.tag).lower()
                if tag_name in ("ref", "gallery", "references"):
                    wikicode.remove(tag)

            for comment in wikicode.filter_comments():
                wikicode.remove(comment)

            # Strip noisy templates except meaningful content wrappers
            for template in list(wikicode.filter_templates()):
                try:
                    t_name = str(template.name).strip().lower()
                    # Remove infoboxes, quotes, clearlines, navboxes, tabs
                    if any(x in t_name for x in ("infobox", "navbox", "pagetabs", "clear", "gallerylink", "parallel link", "references")):
                        wikicode.remove(template)
                    elif t_name == "s" and template.params:
                        # {{S|note}} -> (note)
                        param_val = str(template.params[0].value).strip()
                        wikicode.replace(template, f" ({param_val})")
                    elif t_name == "nihongo" and template.params:
                        wikicode.replace(template, str(template.params[0].value).strip())
                except (ValueError, KeyError, IndexError):
                    pass


            # 4. Extract section tree and text chunks
            sections_data: List[SectionData] = []
            lead_text = ""

            # mwparserfromhell get_sections returns hierarchy
            mw_sections = wikicode.get_sections(include_lead=True, flat=True)

            # Maintain active hierarchy stack for path breadcrumbs (e.g. H2 > H3 > H4)
            current_stack: List[Tuple[int, str]] = []

            for s in mw_sections:
                headings = s.filter_headings()
                if not headings:
                    # Lead section
                    clean_lead = self._clean_section_text(s.strip_code())
                    if clean_lead:
                        lead_text = clean_lead
                    continue

                heading = headings[0]
                level = heading.level
                h_title = str(heading.title).strip()

                # Ignore references, trivia, and gallery sections
                if h_title.lower() in ("references", "gallery", "external links", "navigation"):
                    continue

                # Pop deeper or equal levels to maintain clean breadcrumb
                while current_stack and current_stack[-1][0] >= level:
                    current_stack.pop()
                current_stack.append((level, h_title))

                section_path = " > ".join(item[1] for item in current_stack)

                # Get body text excluding the heading itself
                body_code = s.strip_code()
                # Remove heading title from start if present
                clean_body = self._clean_section_text(body_code)

                if clean_body:
                    sections_data.append(
                        SectionData(
                            title=h_title,
                            level=level,
                            path=section_path,
                            content=clean_body,
                        )
                    )

            # Get known aliases from alias map if available
            aliases = []
            if alias_map and title in alias_map:
                aliases = alias_map[title]
            if infobox and infobox.aliases:
                aliases = list(set(aliases + infobox.aliases))

            return ParsedPage(
                page_id=raw_record.get("page_id", 0),
                title=title,
                canonical_url=raw_record.get("canonical_url", ""),
                categories=raw_record.get("categories", []),
                aliases=aliases,
                infobox=infobox,
                lead_section=lead_text,
                sections=sections_data,
                wiki_links=list(set(wiki_links)),
                fetched_at=raw_record.get("fetched_at", 0.0),
            )

        except Exception as exc:
            logger.warning(f"Failed to parse page '{title}': {exc}. Quarantining.")
            self._quarantine(raw_record, exc)
            return None

    def _clean_section_text(self, text: str) -> str:
        """Normalizes whitespace and removes residual artifacts."""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        cleaned = "\n".join(lines)
        # Collapse multiple blank spaces
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        return cleaned.strip()
