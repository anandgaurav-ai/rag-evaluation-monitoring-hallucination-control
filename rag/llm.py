import os
from dotenv import load_dotenv

from openai import OpenAI

load_dotenv()

def get_llm():
    """
    Returns a callable LLM function that takes a prompt (str)
    and returns a generated response (str).
    """

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found."
            "Set it in your environment or .env file."
        )

    client = OpenAI(api_key = api_key)

    def call_llm(prompt: str) -> str:
        response = client.chat.completions.create(
            model = "gpt-40-mini",
            messages = [
                {"role": "user", "content": prompt}
            ], temperature = 0.2
        )

        return response.choices[0].mesaage.content.strip()

    return call_llm
