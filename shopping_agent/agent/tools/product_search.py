"""
Product Search Tool — text-based semantic search + image-based search.

Uses sentence-transformers (all-MiniLM-L6-v2) for text queries and CLIP for image queries.
Both search against the local products.json mock catalog via FAISS.
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from functools import lru_cache
from langchain_core.tools import tool

_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "products.json"

# ---------------------------------------------------------------------------
# Data & index loading (lazy, cached)
# ---------------------------------------------------------------------------

_text_model = None
_text_index = None
_all_products = None  # full list including duplicates across sources


def _load_text_index():
    """Build FAISS text-embedding index over all product records."""
    global _text_model, _text_index, _all_products

    if _text_model is not None:
        return

    from sentence_transformers import SentenceTransformer
    import faiss

    with open(_DATA_PATH) as f:
        products = json.load(f)
    _all_products = products

    model = SentenceTransformer("all-MiniLM-L6-v2")
    _text_model = model

    # Build description strings for indexing
    corpus = []
    for p in products:
        tags = " ".join(p.get("tags", []))
        text = f"{p['name']} {p['brand']} {p['category']} {p['description']} {tags}"
        corpus.append(text)

    embeddings = model.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    _text_index = index


# ---------------------------------------------------------------------------
# Core search functions
# ---------------------------------------------------------------------------

def text_search(query: str, top_k: int = 5) -> list:
    """
    Perform semantic search over the product catalog using the query text.

    Returns list of product dicts (with score) sorted by relevance.
    """
    _load_text_index()

    query_emb = _text_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    query_emb = query_emb.astype(np.float32)

    k = min(top_k * 3, len(_all_products))  # over-fetch to deduplicate by name
    distances, indices = _text_index.search(query_emb, k)

    seen_names = set()
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        product = dict(_all_products[idx])
        name = product["name"]
        # Keep best score per product name
        if name not in seen_names:
            seen_names.add(name)
            product["relevance_score"] = round(float(dist), 4)
            results.append(product)
        if len(results) >= top_k:
            break

    return results


def image_search(pil_image, top_k: int = 5) -> list:
    """
    Perform image-based similarity search over the product catalog.

    Parameters
    ----------
    pil_image : PIL.Image
    top_k : int

    Returns
    -------
    list of product dicts with similarity_score
    """
    from utils.image_search import search_by_image
    return search_by_image(pil_image, top_k=top_k)


# ---------------------------------------------------------------------------
# LangChain tool wrapper
# ---------------------------------------------------------------------------

@tool
def _product_search_tool(query: str) -> str:
    """
    Search the product catalog for items matching the user's query.
    Input should be a text description of what the user is looking for,
    e.g. 'sustainable laptop under 1000 dollars' or 'noise cancelling headphones'.
    Returns a list of matching products with name, price, rating, source, and sustainability score.
    """
    results = text_search(query, top_k=5)
    if not results:
        return "No products found matching your query."

    lines = [f"Found {len(results)} product(s) matching '{query}':\n"]
    for i, p in enumerate(results, 1):
        stock = "In Stock" if p.get("in_stock", True) else "Out of Stock"
        lines.append(
            f"{i}. **{p['name']}** by {p['brand']}\n"
            f"   Category: {p['category']} | Price: ${p['price_usd']} ({p['source']})\n"
            f"   Rating: {p['rating']}/5 ({p['review_count']} reviews) | "
            f"Sustainability: {p['sustainability_score']}/10 | {stock}\n"
            f"   {p['description'][:120]}..."
        )
    return "\n".join(lines)


def get_product_search_tool():
    """Return the product search LangChain tool."""
    return _product_search_tool
