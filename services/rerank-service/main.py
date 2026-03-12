"""
OCRLLM Reranker Service
========================
Cross-encoder reranking: given a question and N candidate text chunks,  
scores each (question, chunk) pair jointly and returns them sorted by  
descending relevance.

Model: BAAI/bge-reranker-v2-m3
  - State-of-the-art multilingual cross-encoder
  - Handles Japanese, Chinese, English
  - ~570MB, can run on CPU (but benefits from GPU if available)

Usage in RAG pipeline:
  1. Qdrant retrieves top-20 candidates (high recall, lower precision)
  2. THIS service reranks → returns top-N (high precision)
  3. LLM sees only the top-N most relevant chunks

Port: 8004
"""

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from FlagEmbedding import FlagReranker

app = FastAPI(title="OCRLLM Reranker Service")

MODEL_NAME = "BAAI/bge-reranker-v2-m3"
reranker: FlagReranker | None = None


class RerankCandidate(BaseModel):
    chunk_id: str
    text: str
    page_num: int


class RerankRequest(BaseModel):
    question: str
    candidates: list[RerankCandidate]
    # How many to return after reranking (default: all, sorted)
    top_k: int | None = None


class RankedChunk(BaseModel):
    chunk_id: str
    text: str
    page_num: int
    score: float


class RerankResponse(BaseModel):
    ranked: list[RankedChunk]


@app.on_event("startup")
def load_model():
    global reranker
    print(f"Loading reranker model: {MODEL_NAME}")
    # use_fp16=True halves memory and doubles throughput on GPU; safe on CPU too
    reranker = FlagReranker(MODEL_NAME, use_fp16=True)
    print("Reranker model loaded.")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "ready": reranker is not None}


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest):
    assert reranker is not None, "Reranker not loaded yet"

    if not req.candidates:
        return RerankResponse(ranked=[])

    # Build (query, passage) pairs for the cross-encoder
    pairs = [[req.question, c.text] for c in req.candidates]
    scores: list[float] = reranker.compute_score(pairs, normalize=True)

    # Zip scores with candidates and sort descending
    ranked = sorted(
        zip(scores, req.candidates),
        key=lambda x: x[0],
        reverse=True,
    )

    top_k = req.top_k if req.top_k else len(ranked)
    result = [
        RankedChunk(
            chunk_id=c.chunk_id,
            text=c.text,
            page_num=c.page_num,
            score=round(float(score), 6),
        )
        for score, c in ranked[:top_k]
    ]

    return RerankResponse(ranked=result)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)
