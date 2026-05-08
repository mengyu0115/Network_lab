"""
Pre-download selected models into local project cache.

Usage (PowerShell):
  # Optional: use a domestic Hugging Face mirror
  $env:HF_ENDPOINT="https://hf-mirror.com"
  python scripts/precache_models.py
"""
import os

import torchvision.models as tv_models

from src.models.model_loader import (
    HF_CACHE_DIR,
    TORCH_CACHE_DIR,
    ModelLoader,
)


def cache_torchvision_models():
    os.environ.setdefault("TORCH_HOME", TORCH_CACHE_DIR)
    for name in ModelLoader.TORCHVISION_MODELS:
        print(f"[torchvision] caching {name} ...")
        model_fn = getattr(tv_models, name)
        _ = model_fn(weights="DEFAULT")
    print("[torchvision] done")


def cache_clip_models():
    from transformers import CLIPModel, CLIPProcessor, CLIPTextModel, CLIPTokenizer

    for _, repo_id in ModelLoader.CLIP_MODELS.items():
        print(f"[transformers] caching {repo_id} ...")
        _ = CLIPModel.from_pretrained(repo_id, cache_dir=HF_CACHE_DIR)
        _ = CLIPProcessor.from_pretrained(repo_id, cache_dir=HF_CACHE_DIR)
        _ = CLIPTextModel.from_pretrained(repo_id, cache_dir=HF_CACHE_DIR)
        _ = CLIPTokenizer.from_pretrained(repo_id, cache_dir=HF_CACHE_DIR)
    print("[transformers] done")


def main():
    print(f"TORCH cache: {TORCH_CACHE_DIR}")
    print(f"HF cache: {HF_CACHE_DIR}")
    cache_torchvision_models()
    cache_clip_models()
    print("All selected models are cached locally.")


if __name__ == "__main__":
    main()
