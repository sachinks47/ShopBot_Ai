"""
Image search utility using sentence-transformers CLIP model.
Builds an in-memory FAISS index over product descriptions for image-to-product similarity.
"""

import json
import os
import numpy as np
from pathlib import Path

_DATA_PATH = Path(__file__).parent.parent / "data" / "products.json"

# Lazy-loaded globals
_clip_model = None
_clip_index = None
_clip_products = None


def _load_clip():
    """Lazy-load CLIP model and build FAISS index over product descriptions."""
    global _clip_model, _clip_index, _clip_products

    if _clip_model is not None:
        return

    from sentence_transformers import SentenceTransformer
    import faiss

    with open(_DATA_PATH) as f:
        products = json.load(f)

    # Deduplicate by name for image search (one entry per unique product name)
    seen = set()
    unique_products = []
    for p in products:
        if p["name"] not in seen:
            seen.add(p["name"])
            unique_products.append(p)

    _clip_products = unique_products

    # Use CLIP text encoder to embed product descriptions
    # (image queries encoded with same model for cross-modal similarity)
    model = SentenceTransformer("clip-ViT-B-32")
    _clip_model = model

    descriptions = [
        f"{p['name']} {p['brand']} {p['description']}" for p in unique_products
    ]
    embeddings = model.encode(descriptions, convert_to_numpy=True, normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product = cosine similarity (normalised)
    index.add(embeddings.astype(np.float32))
    _clip_index = index


def search_by_image(pil_image, top_k: int = 5) -> list:
    """
    Find the top-k products visually similar to the given PIL image.

    Parameters
    ----------
    pil_image : PIL.Image
        The query image uploaded by the user.
    top_k : int
        Number of results to return.

    Returns
    -------
    list of dict
        Matching product records from products.json.
    """
    import numpy as np
    _load_clip()

    # Encode the image using CLIP
    img_embedding = _clip_model.encode(pil_image, convert_to_numpy=True, normalize_embeddings=True)
    img_embedding = img_embedding.reshape(1, -1).astype(np.float32)

    distances, indices = _clip_index.search(img_embedding, min(top_k, len(_clip_products)))

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx >= 0:
            product = dict(_clip_products[idx])
            product["similarity_score"] = round(float(dist), 4)
            results.append(product)
    return results
