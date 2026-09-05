from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EntityNode(BaseModel):
    id: str  # Canonical name, e.g. "Rimuru Tempest"
    name: str
    entity_type: str = "character"  # character, location, organization, skill, item
    attributes: Dict[str, Any] = Field(default_factory=dict)
    aliases: List[str] = Field(default_factory=list)


class RelationshipEdge(BaseModel):
    id: Optional[int] = None
    source: str
    target: str
    relation_type: str  # ally, subordinate, master, enemy, creator, member_of, species
    confidence: float = 1.0
    source_chunk_id: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)


class GraphSubgraph(BaseModel):
    nodes: List[EntityNode]
    edges: List[RelationshipEdge]
