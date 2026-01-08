def context_overlap(answer: str, context: str) -> float:
    """
    Measures how much of the answer is grounded in the retrieved context.
    Returns a score between 0 and 1.
    """

    if not answer or not context:
        return 0.0

    answer_tokens = set(answer.lower().split())
    context_tokens = set(context.lower().split())

    overlap = answer_tokens & context_tokens

    return len(overlap) / max(len(answer_tokens), 1)

def llm_faithfulness_check(answer: str, context: str, llm) -> bool:
    """
    Uses an LLM to judge whether the answer is fully supported by the context.
    Returns True if grounded, False otherwise.
    """

    if not answer or not context:
        return False


    prompt = f"""
You are a strict evaluator.

Answer:
{answer}

Context:
{context}

Question:
Is the answer fully supported by the context?

Reply with only one word: Yes or No.
""".strip()

    response = llm(prompt).strip().lower()

    return response == "yes"