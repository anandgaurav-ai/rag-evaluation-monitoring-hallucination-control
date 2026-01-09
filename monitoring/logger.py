import json
from datetime import datetime
from pathlib import path


LOG_FILE = path("logs.jsonl")

def log_event(
        query: str,
        answer: str,
        confidence: float,
        decision: str,
        sources: list[str] | None = None
):
    """
    Logs a single RAG interaction for monitoring and auditing.
    """

    event = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "answer": answer,
        "confidence": round(confidence,3),
        "decision": decision,
        "sources": sources or []
    }

    with LOG_FILE.open("a", encoding = "utf-8") as f:
        f.write(json.dumps(event) + "\n")