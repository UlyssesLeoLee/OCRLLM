"""
OCRLLM Embedding Service
=========================
Converts text strings to dense float vectors for storage in Qdrant.

Model: intfloat/multilingual-e5-small
  - 384-dimensional vectors
  - Handles Japanese, Chinese, and English natively
  - ~120MB memory, fast on CPU
  - Best practice: prefix "query: " for queries, "passage: " for documents

Port: 8003
"""

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI(title="OCRLLM Embedding Service")

MODEL_NAME = "intfloat/multilingual-e5-small"
model: SentenceTransformer | None = None


class EmbedRequest(BaseModel):
    texts: list[str]
    # "passage" for chunks being stored, "query" for user questions
    mode: str = "passage"


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    dim: int


@app.on_event("startup")
def load_model():
    global model
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print(f"Embedding model loaded. Vector dimension: {model.get_sentence_embedding_dimension()}")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "ready": model is not None}


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    assert model is not None, "Model not loaded yet"

    # multilingual-e5 requires task prefixes for best performance
    prefix = "passage: " if req.mode == "passage" else "query: "
    prefixed = [prefix + t for t in req.texts]

    vectors = model.encode(prefixed, normalize_embeddings=True).tolist()
    dim = len(vectors[0]) if vectors else 0

    return EmbedResponse(embeddings=vectors, dim=dim)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
