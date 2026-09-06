from wikirag.quality import audit_rows


def test_quality_audit_detects_duplicates_and_missing_fields():
    result = audit_rows([
        {"chunk_id": "1", "content_hash": "same", "entity": "A", "chunk_text": "text"},
        {"chunk_id": "2", "content_hash": "same", "entity": "", "chunk_text": ""},
    ])
    assert result["chunks"] == 2
    assert result["duplicate_hash_groups"] == 1
    assert result["missing_fields"]["entity"] == 1
    assert result["missing_fields"]["chunk_text"] == 1
    assert result["ok"] is False
