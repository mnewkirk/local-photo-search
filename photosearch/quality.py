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

import json
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
# Respect PHOTOSEARCH_CACHE env var so Docker can persist to a named volume
_CACHE_DIR = Path(os.environ.get("PHOTOSEARCH_CACHE", Path.home() / ".cache" / "photosearch"))
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

    # See clip_embed._load_model for why PHOTOSEARCH_DEVICE exists.
    forced = os.environ.get("PHOTOSEARCH_DEVICE")
    if forced:
        _device = forced
    else:
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
    prev_device = _device
    _device = None
    import gc
    gc.collect()
    if prev_device == "cuda":
        torch.cuda.empty_cache()
    elif prev_device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


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


def score_photos_stream(image_paths: list[str], batch_size: int = 8):
    """Score photos in batches, yielding (index, score) as each batch completes.

    Yields:
        (idx, score) — index into image_paths and the float aesthetic score.
        Skipped/failed images are not yielded.

    Progress is printed every ~5% of total photos so long-running jobs remain visible.
    """
    import sys
    _load_models()
    total = len(image_paths)
    report_every = max(batch_size, (total // 20 // batch_size) * batch_size)

    for batch_start in range(0, total, batch_size):
        if batch_start > 0 and report_every > 0 and batch_start % report_every == 0:
            print(f"  [{batch_start}/{total}] Quality scoring in progress...", flush=True)
            sys.stdout.flush()

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
            yield idx, round(float(scores[j].squeeze().cpu()), 3)


def score_photos_batch(
    image_paths: list[str], batch_size: int = 8
) -> list[Optional[float]]:
    """Score multiple photos in batches for efficiency.

    Returns a list parallel to image_paths — each entry is a float
    score or None if that image failed.
    Internally calls score_photos_stream for progress reporting.
    """
    results: list[Optional[float]] = [None] * len(image_paths)
    for idx, score in score_photos_stream(image_paths, batch_size=batch_size):
        results[idx] = score
    return results


# ---------------------------------------------------------------------------
# CLIP concept similarity breakdown
# ---------------------------------------------------------------------------
# Embeds aesthetic concept phrases with the same ViT-L/14 model used for
# scoring, then computes cosine similarity between the photo embedding and
# each concept. This gives a fast, interpretable breakdown of *why* a photo
# scores high or low — no LLM call needed, just vector math.

# Positive concepts (things that make a photo score well)
_POSITIVE_CONCEPTS = {
    "composition": "a well-composed photograph with strong visual balance",
    "lighting": "beautiful natural lighting in a photograph",
    "color": "rich vibrant colors in a photograph",
    "sharpness": "a sharp, high-resolution, detailed photograph",
    "depth": "a photograph with pleasing depth of field and bokeh",
    "mood": "an evocative photograph with strong atmosphere and mood",
    "subject": "a photograph with a clear, compelling subject",
}

# Negative concepts (things that drag a score down)
_NEGATIVE_CONCEPTS = {
    "blur": "a blurry, out-of-focus photograph",
    "exposure": "an overexposed or underexposed photograph",
    "clutter": "a cluttered, busy photograph with no clear subject",
    "noise": "a grainy, noisy, low-quality photograph",
}

# All concepts merged for embedding
_ALL_CONCEPTS = {**_POSITIVE_CONCEPTS, **_NEGATIVE_CONCEPTS}

# Cached concept embeddings (computed once per session)
_concept_embeddings: Optional[dict[str, torch.Tensor]] = None


def _get_concept_embeddings() -> dict[str, torch.Tensor]:
    """Embed all aesthetic concept phrases. Cached after first call."""
    global _concept_embeddings
    if _concept_embeddings is not None:
        return _concept_embeddings

    _load_models()
    import open_clip

    tokenizer = open_clip.get_tokenizer(AESTHETIC_CLIP_MODEL)
    _concept_embeddings = {}

    with torch.no_grad():
        for name, phrase in _ALL_CONCEPTS.items():
            tokens = tokenizer([phrase]).to(_device)
            emb = _clip_model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            _concept_embeddings[name] = emb.squeeze(0)

    return _concept_embeddings


def _classify_scores(scores: dict) -> tuple[list[str], list[str]]:
    """Pick strengths and weaknesses from raw concept similarity scores.

    Strengths: positive concepts whose score is above the mean of all
    positive concept scores (i.e. this photo is notably strong there).

    Weaknesses: negative concepts whose score exceeds a fixed threshold,
    meaning the photo genuinely resembles the negative concept. If no
    negative concept clears the threshold, the list is empty — we don't
    fabricate weaknesses for a good photo.

    Returns (strengths, weaknesses).
    """
    pos_scores = {k: scores[k] for k in _POSITIVE_CONCEPTS if k in scores}
    neg_scores = {k: scores[k] for k in _NEGATIVE_CONCEPTS if k in scores}

    # Strengths: above the mean of positive scores
    pos_mean = sum(pos_scores.values()) / len(pos_scores) if pos_scores else 0
    strengths = sorted(
        [k for k, v in pos_scores.items() if v > pos_mean],
        key=lambda k: pos_scores[k], reverse=True,
    )[:3]

    # Weaknesses: only if the negative concept score is higher than
    # the photo's *best* positive score. This means the photo looks
    # more like "blurry photograph" than it looks like its own best
    # quality — a genuine problem. Otherwise the weakness list stays
    # empty, which is the correct answer for a decent photo.
    pos_max = max(pos_scores.values()) if pos_scores else 0
    weaknesses = sorted(
        [k for k, v in neg_scores.items() if v > pos_max],
        key=lambda k: neg_scores[k], reverse=True,
    )[:2]

    return strengths, weaknesses


def analyze_photo_concepts(image_path: str) -> Optional[dict]:
    """Compute CLIP concept similarity breakdown for a single photo.

    Returns a dict like:
        {
            "strengths": ["lighting", "composition"],
            "weaknesses": ["clutter"],
            "scores": {"composition": 0.28, "lighting": 0.31, ...}
        }
    Weaknesses may be empty if no negative concept scores high enough.
    Returns None if the image can't be processed.
    """
    _load_models()
    concepts = _get_concept_embeddings()

    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = _clip_preprocess(image).unsqueeze(0).to(_device)

        with torch.no_grad():
            img_emb = _clip_model.encode_image(image_tensor)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            img_emb = img_emb.squeeze(0)

        scores = {}
        for name, concept_emb in concepts.items():
            sim = float(torch.dot(img_emb, concept_emb).cpu())
            scores[name] = round(sim, 4)

        strengths, weaknesses = _classify_scores(scores)

        return {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "scores": scores,
        }

    except Exception as e:
        print(f"  Warning: concept analysis failed for {image_path}: {e}")
        return None


def analyze_photos_stream(image_paths: list[str], batch_size: int = 8):
    """Compute CLIP concept breakdown per batch, yielding (index, concept_dict) as each completes.

    Progress is printed every ~5% so long-running jobs remain visible.
    """
    import sys
    _load_models()
    concepts = _get_concept_embeddings()
    total = len(image_paths)
    report_every = max(batch_size, (total // 20 // batch_size) * batch_size)

    for batch_start in range(0, total, batch_size):
        if batch_start > 0 and report_every > 0 and batch_start % report_every == 0:
            print(f"  [{batch_start}/{total}] Concept analysis in progress...", flush=True)
            sys.stdout.flush()

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

        for j, idx in enumerate(valid_indices):
            img_emb = embeddings[j]
            scores = {}
            for name, concept_emb in concepts.items():
                sim = float(torch.dot(img_emb, concept_emb).cpu())
                scores[name] = round(sim, 4)
            strengths, weaknesses = _classify_scores(scores)
            yield idx, {
                "strengths": strengths,
                "weaknesses": weaknesses,
                "scores": scores,
            }


def analyze_photos_batch(
    image_paths: list[str], batch_size: int = 8,
) -> list[Optional[dict]]:
    """Compute CLIP concept breakdown for multiple photos in batches.

    Returns a list parallel to image_paths — each entry is a concept
    dict or None if that image failed.
    Internally calls analyze_photos_stream for progress reporting.
    """
    results: list[Optional[dict]] = [None] * len(image_paths)
    for idx, concept_data in analyze_photos_stream(image_paths, batch_size=batch_size):
        results[idx] = concept_data
    return results
