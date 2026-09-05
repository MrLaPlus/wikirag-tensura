import re
from enum import Enum
from typing import Dict, List, Optional
from wikirag.utils.logging import get_logger

logger = get_logger(__name__)


class QueryArchetype(str, Enum):
    FACTUAL = "factual"          # Specific attribute lookup (species, age, gender, rank)
    RELATIONAL = "relational"    # Connection between 2+ entities (allies, enemy, master)
    COMPARATIVE = "comparative"  # Compare powers/skills between characters
    TIMELINE = "timeline"        # Chronological progression or specific volume/arc
    GENERAL = "general"          # Descriptive overview, lore, or backstory


class QueryRouter:
    """Classifies incoming search queries into archetypes to select optimal retrieval strategies."""

    def __init__(self):
        # Heuristic keywords for Thai and English
        self.factual_keywords = [
            # Thai
            "เผ่า", "อายุ", "เพศ", "สถานะ", "ฉายา", "ตำแหน่ง", "คันจิ", "ส่วนสูง", "แรงค์", "คือตัวอะไร", "ใครคือ",
            # English
            "species", "race", "gender", "age", "status", "title", "titles", "rank", "kanji", "height", "who is", "what is"
        ]

        self.relational_keywords = [
            # Thai
            "ความสัมพันธ์", "เป็นอะไรกับ", "เกี่ยวข้องกับ", "ลูกน้อง", "อาจารย์", "คู่หู", "เพื่อน",
            # English
            "relationship", "related to", "subordinate", "master", "partner", "friend", "ally", "connection"
        ]

        self.comparative_keywords = [
            # Thai
            "เปรียบเทียบ", "เก่งกว่า", "ใครชนะ", "แกร่งกว่า", "ต่างกันยังไง",
            # English
            "compare", "versus", " vs ", "stronger than", "who wins", "difference between"
        ]

        self.timeline_keywords = [
            # Thai
            "เล่มที่", "ตอนที่", "ช่วงไหน", "ภาค", "วิวัฒนาการตอน",
            # English
            "volume", "chapter", "episode", "season", "arc", "when did", "timeline"
        ]

    def classify(self, query: str) -> QueryArchetype:
        """Classifies the query string into one of 5 archetypes using fast heuristic analysis."""
        q_lower = query.lower()

        # 1. Check Comparative
        if any(kw in q_lower for kw in self.comparative_keywords):
            return QueryArchetype.COMPARATIVE

        # 2. Check Relational
        if any(kw in q_lower for kw in self.relational_keywords):
            return QueryArchetype.RELATIONAL

        # 3. Check Timeline
        if any(kw in q_lower for kw in self.timeline_keywords):
            return QueryArchetype.TIMELINE

        # 4. Check Factual
        if any(kw in q_lower for kw in self.factual_keywords):
            return QueryArchetype.FACTUAL

        return QueryArchetype.GENERAL

    def get_strategy_hints(self, archetype: QueryArchetype) -> Dict[str, bool]:
        """Provides retrieval hints based on classified archetype."""
        if archetype == QueryArchetype.FACTUAL:
            return {"prioritize_infobox": True, "top_k_boost": 2}
        elif archetype == QueryArchetype.RELATIONAL:
            return {"expand_all_entities": True, "top_k_boost": 5}
        elif archetype == QueryArchetype.COMPARATIVE:
            return {"split_entities": True, "top_k_boost": 6}
        else:
            return {"prioritize_infobox": False, "top_k_boost": 0}
