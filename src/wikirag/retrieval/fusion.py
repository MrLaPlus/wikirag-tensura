from typing import Any, Dict, List


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = 60,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """Merges multiple ranked lists of retrieved chunks using Reciprocal Rank Fusion (RRF).
    
    Formula:
        RRF_Score(d) = Sum_{m in M} (1 / (k + r_m(d)))
    where r_m(d) is the 1-based rank position of document d in list m, and k=60 is
    the standard smoothing constant.
    
    RRF is parameter-free, highly robust against mismatched score scales (e.g. dense dot product
    vs BM25 scores), and provides significantly better overall recall than individual retrieval legs.
    """
    scores: Dict[str, float] = {}
    doc_lookup: Dict[str, Dict[str, Any]] = {}

    for ranked_list in ranked_lists:
        for rank_idx, doc in enumerate(ranked_list, start=1):
            doc_id = doc.get("chunk_id") or doc.get("content_hash")
            if not doc_id:
                continue

            if doc_id not in doc_lookup:
                doc_lookup[doc_id] = doc

            # Accumulate RRF score
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank_idx))

    # Sort documents by accumulated RRF score in descending order
    sorted_doc_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    fused_results: List[Dict[str, Any]] = []
    for doc_id in sorted_doc_ids[:top_n]:
        item = dict(doc_lookup[doc_id])
        item["rrf_score"] = scores[doc_id]
        # Keep original score for debugging, but set score to RRF
        item["raw_vector_score"] = item.get("score", 0.0)
        item["score"] = scores[doc_id]
        fused_results.append(item)

    return fused_results
