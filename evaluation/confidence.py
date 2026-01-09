def confidence_score(
        retrieval_score: float,
        overlap_score: float,
        llm_grounded: bool | None = None
) -> float:
    """
    Computes a final confidence score (0-1) combining:
    - retrieval relevance
    - answer faithfulness (context overlap)
    - optional LLm grounding judgement
    """

    # Base weighted score
    score = 0.6 * retrieval_score + 0.4 * overlap_score

    # Optional LLM judge adjustment
    if llm_grounded is not None:
        if llm_grounded:
            score += 0.1
        else:
            score -= 0.2

    # Clamp to [0,1]
    return max(0.0, min(score,1.0))