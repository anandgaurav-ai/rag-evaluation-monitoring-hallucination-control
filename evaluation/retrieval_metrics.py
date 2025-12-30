def recall_at_k(retrieved_sources, expected_source, k):
    """
    Returns 1 if expected_source is present in top-k retrieved sources, else 0
    """
    return int(expected_source in retrieved_sources[:k])