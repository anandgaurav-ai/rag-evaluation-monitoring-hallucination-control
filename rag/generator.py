def generate_answer(query: str, context: str, llm) -> str:
    """
    Generate an answer using ONLY the provided context.
    If the answer is not present in the context, the model must say "I don't know".
    """

    prompt = f"""
You are a cautious assistant.

Answer the question using ONLY the information present in the context below.
If the answer cannot be found in the context, reply exactly with:
"I don't know."

Context:
{context}

Question:
{query}

Answer:
""".strip()

    response = llm(prompt)

    return response.strip()
