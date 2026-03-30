"""Aesthetic quality scoring using the LAION improved aesthetic predictor.

Scores each photo on a 1–10 scale based on general photographic quality,
composition, and visual appeal. Uses a pretrained linear model on top of
CLIP ViT-L/14 embeddings (768-dim) — a separate model from the ViT-B/16
used for semantic search, but loaded only during scoring and then released.

The model was trained on a combination of the SAC, LAION-Logos, and AVA
datasets. Scores roughly correspond to:
    1–3  : poor quality (blurry, bad exposure, random snapshots)
    3–5  : average (typical phone photos, unremarkable)
    5–7  : good (well-composed, pleasant lighting)
    7–9  : excellent (professional quality, striking composition)
    9–10 : exceptional (gallery-worthy, extraordinary)

Reference: https://github.com/christophschuhmann/improved-aesthetic-predictor
"""

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFile

# Allow loading of slightly truncated JPEGs — common with camera files
# where the last few bytes are missing but the image is otherwise fine.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Lazy-loaded model state — separate from the search CLIP model
_aesthetic_model = None
_clip_model = None
_clip_preprocess = None
_device = None

# The aesthetic predictor uses ViT-L/14 with OpenAI weights (768-dim).
# This is different from the ViT-B/16 used for semantic search.
AESTHETIC_CLIP_MODEL = "ViT-L-14"
AESTHETIC_CLIP_PRETRAINED = "openai"
AESTHETIC_EMBED_DIM = 768

# URL for the pretrained aesthetic predictor weights
AESTHETIC_MODEL_URL = (
    "https://github.com/christophschuhmann/improved-aesthetic-predictor"
    "/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"
)

# Local cache path for downloaded model weights
_CACHE_DIR = Path.home() / ".cache" / "photosearch"
_MODEL_FILENAME = "sac+logos+ava1-l14-linearMSE.pth"


class AestheticScorer(nn.Module):
    """MLP aesthetic predictor on CLIP embeddings.

    Architecture matches the improved-aesthetic-predictor repo:
      Linear(768, 1024) → Dropout → Linear(1024, 128) → Dropout →
      Linear(128, 64) → Dropout → Linear(64, 16) → Linear(16, 1)

    Trained to predict mean human aesthetic ratings (1–10 scale).
    """

    def __init__(self, input_dim: int = AESTHETIC_EMBED_DIM):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def _download_model_weights() -> str:
    """Download pretrained aesthetic model weights if not already cached.

    Returns the path to the local weights file.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = _CACHE_DIR / _MODEL_FILENAME

    if model_path.exists():
        return str(model_path)

    print(f"  Downloading aesthetic model weights to {model_path}...")

    # Try huggingface_hub first (already a dependency of open-clip-torch),
    # fall back to urllib for environments without it.
    try:
        import urllib.request
        urllib.request.urlretrieve(AESTHETIC_MODEL_URL, str(model_path))
    except Exception as e:
        raise RuntimeError(
            f"Could not download aesthetic model weights: {e}\n"
            f"Download manually from:\n  {AESTHETIC_MODEL_URL}\n"
            f"and place at:\n  {model_path}"
        )

    print(f"  ✓ Downloaded aesthetic model weights ({model_path.stat().st_size / 1024:.0f} KB)")
    return str(model_path)


def _load_models():
    """Lazy-load the aesthetic CLIP model and scoring head."""
    global _aesthetic_model, _clip_model, _clip_preprocess, _device

    if _aesthetic_model is not None:
        return

    import open_clip

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    if _device == "cpu" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _device = "mps"

    print(f"  Loading aesthetic model (CLIP {AESTHETIC_CLIP_MODEL} + scorer) on {_device}...")
    t0 = time.time()

    # Load CLIP ViT-L/14 for aesthetic embeddings
    _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
        AESTHETIC_CLIP_MODEL, pretrained=AESTHETIC_CLIP_PRETRAINED,
        device=_device, quick_gelu=True,
    )
    _clip_model.eval()

    # Load the aesthetic scoring head
    weights_path = _download_model_weights()
    _aesthetic_model = AestheticScorer(input_dim=AESTHETIC_EMBED_DIM)

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

    # The checkpoint may use "layers.X.weight" keys (matching our model)
    # or bare keys from a different wrapper. Try direct load first.
    try:
        _aesthetic_model.load_state_dict(state_dict)
    except RuntimeError:
        # If keys don't match, the checkpoint likely wraps layers differently.
        # Log what we got so debugging is easy if this ever breaks again.
        print(f"  Checkpoint keys: {list(state_dict.keys())[:10]}...")
        print(f"  Model expects:   {list(_aesthetic_model.state_dict().keys())[:10]}...")
        raise

    _aesthetic_model.to(_device)
    _aesthetic_model.eval()

    elapsed = time.time() - t0
    print(f"  ✓ Aesthetic model loaded in {elapsed:.1f}s")


def unload_models():
    """Release aesthetic model memory.

    Call this after scoring is complete to free up GPU/RAM for other
    pipeline steps (CLIP search model, face detection, etc.).
    """
    global _aesthetic_model, _clip_model, _clip_preprocess, _device
    _aesthetic_model = None
    _clip_model = None
    _clip_preprocess = None
    _device = None
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def score_photo(image_path: str) -> Optional[float]:
    """Score a single photo's aesthetic quality.

    Returns a float score (roughly 1–10), or None if the image can't
    be processed.
    """
    _load_models()
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = _clip_preprocess(image).unsqueeze(0).to(_device)

        with torch.no_grad():
            embedding = _clip_model.encode_image(image_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            score = _aesthetic_model(embedding)

        return round(float(score.squeeze().cpu()), 3)
    except Exception as e:
        print(f"  Warning: could not score {image_path}: {e}")
        return None


def score_photos_batch(
    image_paths: list[str], batch_size: int = 8
) -> list[Optional[float]]:
    """Score multiple photos in batches for efficiency.

    Returns a list parallel to image_paths — each entry is a float
    score or None if that image failed.
    """
    _load_models()
    results: list[Optional[float]] = [None] * len(image_paths)

    for batch_start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[batch_start : batch_start + batch_size]
        batch_images = []
        valid_indices = []

        for i, path in enumerate(batch_paths):
            try:
                img = Image.open(path).convert("RGB")
                batch_images.append(_clip_preprocess(img))
                valid_indices.append(batch_start + i)
            except Exception as e:
                print(f"  Warning: could not load {path}: {e}")

        if not batch_images:
            continue

        batch_tensor = torch.stack(batch_images).to(_device)
        with torch.no_grad():
            embeddings = _clip_model.encode_image(batch_tensor)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            scores = _aesthetic_model(embeddings)

        for j, idx in enumerate(valid_indices):
            results[idx] = round(float(scores[j].squeeze().cpu()), 3)

    return results
