from typing import List, Set


def compute_hit_rate_at_k(retrieved_entities: List[str], expected_entities: List[str], k: int = 5) -> float:
    """Returns 1.0 if at least one expected entity is found in the top-k retrieved list, else 0.0."""
    if not expected_entities or not retrieved_entities:
        return 0.0

    top_k_set = {e.lower() for e in retrieved_entities[:k]}
    for exp in expected_entities:
        if exp.lower() in top_k_set:
            return 1.0
    return 0.0


def compute_mrr(retrieved_entities: List[str], expected_entities: List[str]) -> float:
    """Computes Mean Reciprocal Rank (MRR) for the first matching expected entity.
    
    Formula:
        MRR = 1 / rank_position
    """
    if not expected_entities or not retrieved_entities:
        return 0.0

    exp_set = {e.lower() for e in expected_entities}
    for rank, item in enumerate(retrieved_entities, start=1):
        if item.lower() in exp_set:
            return 1.0 / rank

    return 0.0


def compute_keyword_recall(generated_answer: str, expected_keywords: List[str]) -> float:
    """Measures the proportion of expected keywords/facts present in the generated answer."""
    if not expected_keywords:
        return 1.0

    ans_lower = generated_answer.lower()
    matches = sum(1 for kw in expected_keywords if kw.lower() in ans_lower)
    return float(matches / len(expected_keywords))
