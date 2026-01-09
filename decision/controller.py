def decide_action(
        answer: str,
        confidence: float,
        threshold: float = 0.65
):
    """
    Decides whether to answer or refuse based on confidence.
    """

    if confidence >= threshold:
        return {
            "decision": "answer",
            "final_answer": answer

        }

    return {
        "decision": "refuse",
        "final_answer": "I don't have enough reliable information to answer this."
    }