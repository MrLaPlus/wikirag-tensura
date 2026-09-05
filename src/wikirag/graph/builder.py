from typing import List, Optional
from wikirag.graph.models import EntityNode, RelationshipEdge
from wikirag.graph.sqlite_graph import SQLiteEntityGraph
from wikirag.parser.models import ParsedPage
from wikirag.utils.logging import get_logger

logger = get_logger(__name__)


class GraphBuilder:
    """Extracts entities and relational triples from parsed wiki pages and builds the SQLite graph."""

    def __init__(self, graph: SQLiteEntityGraph):
        self.graph = graph

    def process_page(self, page: ParsedPage) -> None:
        """Extracts node attributes and edges from a parsed page."""
        entity_id = page.title

        # 1. Upsert Entity Node
        attributes = {}
        if page.infobox:
            attributes = page.infobox.model_dump(exclude_none=True)

        entity_type = "character"
        if any("location" in c.lower() for c in page.categories):
            entity_type = "location"
        elif any("skill" in c.lower() or "magic" in c.lower() for c in page.categories):
            entity_type = "skill"
        elif any("organization" in c.lower() or "group" in c.lower() for c in page.categories):
            entity_type = "organization"

        node = EntityNode(
            id=entity_id,
            name=entity_id,
            entity_type=entity_type,
            attributes=attributes,
            aliases=page.aliases,
        )
        self.graph.upsert_entity(node)

        # 2. Extract Triples from Infobox Attributes
        if page.infobox:
            ib = page.infobox
            # Species edge
            if ib.species:
                self.graph.add_relationship(
                    RelationshipEdge(
                        source=entity_id,
                        target=ib.species,
                        relation_type="species_of",
                        confidence=1.0,
                    )
                )

            # Titles edges (e.g. Demon Lord, True Dragon)
            for title in ib.titles:
                self.graph.add_relationship(
                    RelationshipEdge(
                        source=entity_id,
                        target=title,
                        relation_type="has_title",
                        confidence=1.0,
                    )
                )

            # Custom attributes (affiliation, country, relatives, master, etc.)
            for k, v in ib.raw_fields.items():
                k_lower = k.lower()
                if any(term in k_lower for term in ["master", "teacher", "mentor"]):

                    for val in self._split_values(v):
                        self.graph.add_relationship(
                            RelationshipEdge(
                                source=entity_id,
                                target=val,
                                relation_type="disciple_of",
                                confidence=0.9,
                            )
                        )
                elif any(term in k_lower for term in ["subordinate", "servant"]):
                    for val in self._split_values(v):
                        self.graph.add_relationship(
                            RelationshipEdge(
                                source=val,
                                target=entity_id,
                                relation_type="subordinate_of",
                                confidence=0.9,
                            )
                        )
                elif any(term in k_lower for term in ["country", "nation", "affiliation"]):
                    for val in self._split_values(v):
                        self.graph.add_relationship(
                            RelationshipEdge(
                                source=entity_id,
                                target=val,
                                relation_type="affiliated_with",
                                confidence=0.95,
                            )
                        )

        # 3. Extract Links from Lead Section Mentions
        if page.lead_section:
            # Connect entity to pages directly referenced in lead section
            for link in page.wiki_links[:15]:
                if link != entity_id and not link.startswith("File:") and not link.startswith("Category:"):
                    self.graph.add_relationship(
                        RelationshipEdge(
                            source=entity_id,
                            target=link,
                            relation_type="associated_with",
                            confidence=0.7,
                        )
                    )

    def _split_values(self, text: str) -> List[str]:
        """Splits comma/bullet separated lists of names."""
        import re
        parts = re.split(r"[,;\n•]|<br\s*/?>", text)
        clean = []
        for p in parts:
            s = p.strip(" []{}*\"'")
            if s and len(s) > 1 and not s.startswith("http"):
                clean.append(s)
        return clean
