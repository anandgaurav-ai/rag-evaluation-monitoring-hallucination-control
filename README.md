🧠 RAG Evaluation, Monitoring & Hallucination Control System

This repository demonstrates a production-grade Retrieval-Augmented Generation (RAG) system with a strong focus on:

🔍 Retrieval evaluation

🚫 Hallucination detection

📊 Confidence-based decision making

📝 Monitoring & observability

Unlike typical RAG demos, this system does not always answer.
It explicitly refuses when confidence is low.

🚀 Key Features

FAISS-based retrieval with SentenceTransformers

Strictly grounded answer generation

Rule-based + LLM-based faithfulness checks

Confidence scoring combining retrieval & faithfulness

Decision controller (answer vs refuse)

Structured JSON logging for monitoring

Offline retrieval evaluation (Recall@K)

🏗️ Architecture Overview
Online (Inference / API)
User Query
   ↓
Retriever (FAISS)
   ↓
Context-only Generator
   ↓
Faithfulness Checks
   ↓
Confidence Scoring
   ↓
Decision (Answer / Refuse)
   ↓
Monitoring Logs

Offline (Evaluation)
eval_queries.jsonl
   ↓
Retriever
   ↓
Recall@K (R@1, R@3, R@5)


Important:
Offline evaluation and online inference are intentionally separated.

🧠 Design Philosophy

Never hallucinate confidently

Prefer refusal over misinformation

Separate evaluation from inference

Keep signals explainable

Optimize for production realism

