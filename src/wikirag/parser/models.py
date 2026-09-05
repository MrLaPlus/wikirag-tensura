from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class InfoboxData(BaseModel):
    """Structured, typed representation of a Wiki character/entity infobox."""
    name: Optional[str] = None
    kanji: Optional[str] = None
    romaji: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)
    species: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[str] = None
    status: Optional[str] = None
    country: Optional[str] = None
    affiliation: List[str] = Field(default_factory=list)
    occupation: List[str] = Field(default_factory=list)
    titles: List[str] = Field(default_factory=list)
    family: List[str] = Field(default_factory=list)
    rank: Optional[str] = None
    first_appearance: Optional[str] = None
    image_url: Optional[str] = None
    raw_fields: Dict[str, str] = Field(default_factory=dict)


class SectionData(BaseModel):
    """Represents a structured section preserving document hierarchy."""
    title: str
    level: int # 2 for H2, 3 for H3, 4 for H4
    path: str # e.g. "Appearance > Slime Form"
    content: str


class ParsedPage(BaseModel):
    """Clean, structured output of a parsed wiki page."""
    page_id: int
    title: str
    canonical_url: str
    categories: List[str] = Field(default_factory=list)
    aliases: List[str] = Field(default_factory=list)
    infobox: Optional[InfoboxData] = None
    lead_section: str = ""
    sections: List[SectionData] = Field(default_factory=list)
    wiki_links: List[str] = Field(default_factory=list)
    fetched_at: float


class ChunkRecord(BaseModel):
    """Final chunk record ready for embedding and indexing."""
    chunk_id: str
    entity: str
    canonical_url: str
    section_path: str
    chunk_type: str # "infobox" | "text"
    chunk_text: str # Text with contextual header prepended
    raw_text: str # Original body without header
    infobox_json: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    aliases: List[str] = Field(default_factory=list)
    content_hash: str
    source_project: str
    fetched_at: float
