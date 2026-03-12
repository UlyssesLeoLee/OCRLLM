"""
Model Download Script for OCRLLM
=================================
Downloads Qwen2.5-7B-Instruct weights from HuggingFace to local storage.
Run this ONCE before starting the vLLM Docker container.

Usage:
    python services/model-runner/download_model.py

Requirements:
    pip install huggingface_hub
"""

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# The HuggingFace model repository ID to download
MODEL_REPO_ID = "Qwen/Qwen3.5-9B-Instruct"

# Local destination: e:\OCRLLM\storage\models\Qwen3.5-9B-Instruct
# This path is mounted as /models inside the Docker container.
STORAGE_DIR = Path(__file__).resolve().parents[2] / "storage" / "models"
MODEL_LOCAL_DIR = STORAGE_DIR / "Qwen3.5-9B-Instruct"

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def main():
    print(f"Target directory : {MODEL_LOCAL_DIR}")
    print(f"Model repository : {MODEL_REPO_ID}")
    print()

    # Check if the model is already downloaded
    if MODEL_LOCAL_DIR.exists() and any(MODEL_LOCAL_DIR.glob("*.safetensors")):
        print("✅ Model weights already exist locally. Nothing to download.")
        print(f"   Location: {MODEL_LOCAL_DIR}")
        return

    MODEL_LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed.")
        print("Run: pip install huggingface_hub")
        sys.exit(1)

    # Optional: Set HF_TOKEN if the model requires authentication
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("HF_TOKEN found — using authenticated download.")
    else:
        print("No HF_TOKEN set. Downloading as anonymous (works for public models like Qwen).")

    print(f"\nStarting download of {MODEL_REPO_ID}...")
    print("This is approximately 15GB and may take 20-60 minutes depending on your connection.\n")

    snapshot_download(
        repo_id=MODEL_REPO_ID,
        local_dir=str(MODEL_LOCAL_DIR),
        token=hf_token,
        # Ignore .bin files if .safetensors are available (prefer safetensors)
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )

    print(f"\n✅ Download complete!")
    print(f"   Model saved to: {MODEL_LOCAL_DIR}")
    print()
    print("Next step: start the vLLM container with:")
    print("   docker compose -f services/model-runner/docker-compose.yml up -d")

if __name__ == "__main__":
    main()
