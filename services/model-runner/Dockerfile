# Use the official vLLM image with CUDA support
# This is the only supported way to run vLLM on Windows (via Docker with GPU passthrough)
FROM vllm/vllm-openai:latest

# The model will be mounted from the host to avoid re-downloading inside the container.
# Model weights should be placed at: e:\OCRLLM\storage\models\<model-name>\
# They will be mounted to /models inside the container.

# vLLM exposes an OpenAI-compatible API by default.
# The entrypoint is already set to run vllm serve.
# We override CMD via docker-compose or run command.
