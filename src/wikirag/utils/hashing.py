import hashlib
from typing import Any


def compute_sha256(text: str) -> str:
    """Computes a hex-encoded SHA-256 hash of the input string.
    
    This ensures deterministic chunk IDs and allows idempotent upsertion:
    re-indexing the exact same text produces the exact same hash, preventing duplicates.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_chunk_id(entity: str, section_path: str, chunk_index: int, content: str) -> str:
    """Creates a deterministic, human-readable chunk identifier with hash suffix.
    
    Example: Rimuru_Tempest__Appearance__chunk_0__a1b2c3d4
    """
    clean_entity = entity.replace(" ", "_").replace("/", "_")
    clean_section = section_path.replace(" > ", "__").replace(" ", "_")
    content_hash = compute_sha256(content)[:10]
    return f"{clean_entity}__{clean_section}__c{chunk_index}__{content_hash}"
