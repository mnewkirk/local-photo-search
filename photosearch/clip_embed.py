"""CLIP embedding generation for images and text queries.

Uses open_clip for local inference — no API calls.
Embeddings are 512-dimensional float vectors stored in sqlite-vec.
"""

from pathlib import Path
from typing import Optional

import torch
from PIL import Image, ImageFile

# Allow loading of slightly truncated JPEGs — common with camera files
# where the last few bytes are missing but the image is otherwise fine.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Lazy-loaded model state
_model = None
_preprocess = None
_tokenizer = None
_device = None

# Model choice: ViT-B/16 with OpenAI pretrained weights.
#
# OpenAI's CLIP weights are specifically trained for semantic image-text
# alignment and significantly outperform LAION-pretrained weights on natural
# language queries like "people outdoors" or "child playing in park".
#
# ViT-B/16 vs ViT-B/32:
#   - B/16 uses a 16px patch size (vs 32px), giving 4× more image patches
#     and substantially better fine-grained understanding
#   - Both produce 512-dim embeddings — no DB schema change required
#   - B/16 is ~2-3× slower to index than B/32 but still fast on N100/MPS
#   - Model download: ~330 MB (one-time, cached to ~/.cache/huggingface/)
#
# If you need faster indexing at some quality cost, switch back to:
#   MODEL_NAME = "ViT-B-32"
#   PRETRAINED = "openai"
MODEL_NAME = "ViT-B-16"
PRETRAINED = "openai"


def _load_model():
    """Lazy-load the CLIP model on first use."""
    global _model, _preprocess, _tokenizer, _device

    if _model is not None:
        return

    import open_clip

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use MPS (Apple Silicon GPU) if available
    if _device == "cpu" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _device = "mps"

    _model, _, _preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=_device, quick_gelu=True,
    )
    _tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    _model.eval()


def unload_model():
    """Free the CLIP model from memory (GPU and CPU).

    Call this after finishing a batch of embeddings to reclaim memory
    before loading other models (e.g., aesthetic scoring).
    """
    global _model, _preprocess, _tokenizer, _device
    _model = None
    _preprocess = None
    _tokenizer = None
    prev_device = _device
    _device = None
    import gc
    gc.collect()
    if prev_device == "cuda":
        torch.cuda.empty_cache()
    elif prev_device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def embed_image(image_path: str) -> Optional[list[float]]:
    """Generate a CLIP embedding for an image file.

    Returns a list of 512 floats, or None if the image can't be processed.
    """
    _load_model()
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = _preprocess(image).unsqueeze(0).to(_device)
        with torch.no_grad():
            embedding = _model.encode_image(image_tensor)
            # Normalize to unit vector for cosine similarity
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.squeeze().cpu().tolist()
    except Exception as e:
        print(f"  Warning: could not embed {image_path}: {e}")
        return None


def embed_text(text: str) -> Optional[list[float]]:
    """Generate a CLIP embedding for a text query.

    Returns a list of 512 floats, or None on failure.
    """
    _load_model()
    try:
        tokens = _tokenizer([text]).to(_device)
        with torch.no_grad():
            embedding = _model.encode_text(tokens)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.squeeze().cpu().tolist()
    except Exception as e:
        print(f"  Warning: could not embed text '{text}': {e}")
        return None


def embed_images_stream(image_paths: list[str], batch_size: int = 8):
    """Embed images in batches, yielding (index, embedding) as each batch completes.

    Yields results incrementally so callers can store to DB and print progress
    without waiting for the entire dataset to be processed.
    """
    import sys
    _load_model()
    total = len(image_paths)
    report_every = max(batch_size, (total // 20 // batch_size) * batch_size)  # ~20 updates

    for batch_start in range(0, total, batch_size):
        if batch_start > 0 and batch_start % report_every == 0:
            print(f"  [{batch_start}/{total}] CLIP embedding in progress...")
            sys.stdout.flush()
        batch_paths = image_paths[batch_start:batch_start + batch_size]
        batch_images = []
        valid_indices = []

        for i, path in enumerate(batch_paths):
            try:
                img = Image.open(path).convert("RGB")
                batch_images.append(_preprocess(img))
                valid_indices.append(batch_start + i)
            except Exception as e:
                print(f"  Warning: could not load {path}: {e}")

        if not batch_images:
            continue

        batch_tensor = torch.stack(batch_images).to(_device)
        with torch.no_grad():
            embeddings = _model.encode_image(batch_tensor)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        for j, idx in enumerate(valid_indices):
            yield idx, embeddings[j].cpu().tolist()


def embed_images_batch(image_paths: list[str], batch_size: int = 8) -> list[Optional[list[float]]]:
    """Embed multiple images in batches for efficiency.

    Returns a list parallel to image_paths — each entry is either
    a 512-float list or None if that image failed.
    """
    results: list[Optional[list[float]]] = [None] * len(image_paths)
    for idx, emb in embed_images_stream(image_paths, batch_size):
        results[idx] = emb
    return results
