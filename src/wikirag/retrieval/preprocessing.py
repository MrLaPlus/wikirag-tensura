import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from wikirag.utils.logging import get_logger

logger = get_logger(__name__)


class QueryPreprocessor:
    """Preprocesses user queries for cross-lingual Thai/English fandom retrieval.
    
    Functions:
    - Detects language (Thai unicode range \u0E00-\u0E7F)
    - Expands aliases using the persistent wiki alias map
    - Resolves character aliases to canonical search terms
    """

    def __init__(self, alias_map_path: Optional[str] = None):
        self.alias_to_canonical: Dict[str, str] = {}
        self.canonical_to_aliases: Dict[str, List[str]] = {}
        if alias_map_path and Path(alias_map_path).exists():
            self._load_alias_map(Path(alias_map_path))

    def _load_alias_map(self, path: Path) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.canonical_to_aliases = data
            for canonical, aliases in data.items():
                for a in aliases:
                    self.alias_to_canonical[a.lower()] = canonical
                self.alias_to_canonical[canonical.lower()] = canonical
            logger.info(f"Loaded {len(self.alias_to_canonical)} alias terms into QueryPreprocessor.")
        except Exception as e:
            logger.warning(f"Could not load alias map from {path}: {e}")

    def detect_language(self, text: str) -> str:
        """Determines if the query contains Thai characters."""
        thai_pattern = re.compile(r"[\u0E00-\u0E7F]")
        if thai_pattern.search(text):
            return "th"
        return "en"

    def expand_query(self, query: str) -> Tuple[str, List[str]]:
        """Finds entity matches in the query and enriches with known aliases.
        
        Returns:
            (expanded_query_text, matched_entities)
        """
        matched_entities = []
        tokens = query.split()
        expanded_parts = [query]

        # Check full query and sub-phrases against alias map
        query_lower = query.lower()
        for alias, canonical in self.alias_to_canonical.items():
            if len(alias) >= 3 and alias in query_lower:
                if canonical not in matched_entities:
                    matched_entities.append(canonical)
                    # Add top aliases
                    aliases = self.canonical_to_aliases.get(canonical, [])
                    if aliases:
                        expanded_parts.append(f"(also known as: {', '.join(aliases[:3])})")

        return " ".join(expanded_parts), matched_entities
