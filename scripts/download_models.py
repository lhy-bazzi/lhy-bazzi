#!/usr/bin/env python
"""
Download BGE-M3 and BGE-Reranker-v2-m3 models to local cache.

Usage:
    python scripts/download_models.py [--mirror modelscope]

Options:
    --mirror modelscope   Use ModelScope (recommended in China)
    --mirror huggingface  Use HuggingFace Hub (default)
"""

from __future__ import annotations

import argparse
import sys


MODELS = [
    ("BAAI/bge-m3",              "Embedding model (dense + sparse)"),
    ("BAAI/bge-reranker-v2-m3",  "Reranker model"),
]


def _download_huggingface(model_id: str) -> None:
    from huggingface_hub import snapshot_download
    print(f"  Downloading {model_id} from HuggingFace...")
    path = snapshot_download(repo_id=model_id)
    print(f"  Saved to: {path}")


def _download_modelscope(model_id: str) -> None:
    from modelscope import snapshot_download
    print(f"  Downloading {model_id} from ModelScope...")
    path = snapshot_download(model_id)
    print(f"  Saved to: {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", choices=["huggingface", "modelscope"], default="huggingface")
    args = ap.parse_args()

    download_fn = _download_modelscope if args.mirror == "modelscope" else _download_huggingface

    print(f"Mirror: {args.mirror}\n")
    for model_id, desc in MODELS:
        print(f"[{desc}]")
        try:
            download_fn(model_id)
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        print()

    print("All models downloaded successfully.")


if __name__ == "__main__":
    main()
