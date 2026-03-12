"""
OCRLLM Model-Runner Service
============================
Acts as a thin proxy/gateway between the Rust controller and the vLLM inference
backend running inside Docker.

Architecture:
  Rust Gateway (port 7070)
       │
       ▼
  model-runner (this service, port 8001)   ← FastAPI proxy
       │
       ▼
  vLLM Docker Container (port 8001 on host → 8000 in container)

The vLLM container exposes an OpenAI-compatible API. This service:
  1. Accepts requests from the Rust controller.
  2. Forwards them to vLLM.
  3. Returns the response.

If the vLLM backend is not running, returns a clear error (not a silent stub).

Run the vLLM backend first:
  docker compose -f services/model-runner/docker-compose.yml up -d

Then run this service:
  python services/model-runner/main.py
"""

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="OCRLLM Model-Runner (vLLM Proxy)")

# vLLM Docker container exposes an OpenAI-compatible API on this URL.
# docker-compose.yml maps container port 8000 -> host port 8001.
# This proxy runs on 8002 so both can coexist without port conflict.
VLLM_BASE_URL = "http://localhost:8001"
VLLM_MODEL_NAME = "Qwen2.5-7B-Instruct"

# Timeout for inference requests (large models can be slow on first token)
INFERENCE_TIMEOUT_SECONDS = 120.0


class ChatRequest(BaseModel):
    messages: list[dict]
    max_tokens: int = 512
    temperature: float = 0.3


@app.get("/health")
async def health():
    """Check if the vLLM backend is reachable."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{VLLM_BASE_URL}/health")
            r.raise_for_status()
        return {"status": "ok", "backend": "vllm", "url": VLLM_BASE_URL}
    except Exception as e:
        return {"status": "backend_unavailable", "error": str(e), "url": VLLM_BASE_URL}


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatRequest):
    """
    Proxy chat completion requests to the vLLM OpenAI-compatible endpoint.
    Returns an error if vLLM is not running (no silent stub fallback).
    """
    payload = {
        "model": VLLM_MODEL_NAME,
        "messages": request.messages,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
    }

    try:
        async with httpx.AsyncClient(timeout=INFERENCE_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{VLLM_BASE_URL}/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=(
                "vLLM backend is not running. Start it with: "
                "docker compose -f services/model-runner/docker-compose.yml up -d"
            ),
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail=f"vLLM backend timed out after {INFERENCE_TIMEOUT_SECONDS}s.",
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"vLLM error: {e.response.text}",
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
