from collections import Counter
from typing import Any, Dict, Iterable


def audit_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Audit indexed chunks without mutating the vector store."""
    records = list(rows)
    hashes = [str(row.get("content_hash") or "") for row in records]
    duplicate_groups = sum(count > 1 for value, count in Counter(hashes).items() if value)
    missing = {
        "chunk_id": sum(not row.get("chunk_id") for row in records),
        "content_hash": sum(not row.get("content_hash") for row in records),
        "entity": sum(not str(row.get("entity") or "").strip() for row in records),
        "chunk_text": sum(not str(row.get("chunk_text") or "").strip() for row in records),
    }
    fetched = [float(row.get("fetched_at") or 0) for row in records if row.get("fetched_at")]
    return {
        "chunks": len(records),
        "unique_content_hashes": len(set(hashes)) - (1 if "" in hashes else 0),
        "duplicate_hash_groups": duplicate_groups,
        "missing_fields": missing,
        "latest_fetched_at": max(fetched) if fetched else None,
        "ok": not duplicate_groups and not any(missing.values()),
    }
