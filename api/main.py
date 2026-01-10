from fastapi import FastAPI
from pydantic import BaseModel

from rag.retriever import Retriever
from rag.generator import generate_answer

from evaluation.faithfulness import (
    context_overlap,
    llm_faithfulness_check
)
from evaluation.confidence import confidence_score
from decision.controller import decide_action
from monitoring.logger import log_event

from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title = "RAG Evaluation and Hallucination Control API")


# ---- Initialize core components once ----
retriever = Retriever()


class QueryRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(request: QueryRequest):
    query = request.question

    # 1. Retrieve context
    context, retrieval_score, sources = retriever.retrieve(query)


    # 2. Generate Answer
    from rag.llm import get_llm
    llm = get_llm()
    answer = generate_answer(query, context, llm)

    # 3. Rule based faithfulness
    overlap_score = context_overlap(answer, context)

    # 4. LLm-as-judge (borderline cases)
    llm_grounded = None

    if  0.2 <= overlap_score <= 0.5:
        llm_grounded = llm_faithfulness_check(answer, context, llm)

    # 5. Confidence Scoring
    confidence = confidence_score(
        retrieval_score = retrieval_score,
        overlap_score = overlap_score,
        llm_grounded = llm_grounded
    )

    # 6.Decision
    decision = decide_action(answer, confidence)

    # 7. Monitoring
    log_event(
        query = query,
        answer = decision["final_answer"],
        confidence = confidence,
        decision = decision["decision"],
        sources = sources
    )

    return {
        "question": query,
        "decision": decision["decision"],
        "answer": decision["final_answer"],
        "confidence": round(confidence, 3),
        "sources": sources
    }



