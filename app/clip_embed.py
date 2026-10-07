"""CLIP ViT-B/32 keyframe/text embeddings (CPU). Shared by build_index.py and LocalSource.text_search."""

from functools import lru_cache
from pathlib import Path

import numpy as np

import app.config  # noqa: F401  (TLS setup before weights download)

MODEL = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"


@lru_cache(maxsize=1)
def _load():
    import open_clip
    import torch

    torch.set_grad_enabled(False)
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
    model.eval()
    return model, preprocess, open_clip.get_tokenizer(MODEL)


def embed_images(paths: list[Path], batch: int = 32) -> np.ndarray:
    import torch
    from PIL import Image

    model, preprocess, _ = _load()
    out = []
    for i in range(0, len(paths), batch):
        x = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[i : i + batch]])
        out.append(model.encode_image(x).float().numpy())
    emb = np.concatenate(out) if out else np.zeros((0, 512), np.float32)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def embed_text(text: str) -> np.ndarray:
    model, _, tokenizer = _load()
    v = model.encode_text(tokenizer([text])).float().numpy()[0]
    return v / np.linalg.norm(v)
