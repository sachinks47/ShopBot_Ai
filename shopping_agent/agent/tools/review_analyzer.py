"""
Review Analyzer Tool — uses IBM Granite via watsonx.ai to summarise
product reviews and detect sentiment. Includes fake-review heuristic.
"""

import json
import sys
import os
from pathlib import Path
from langchain_core.tools import tool

_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "products.json"
_products_cache = None


def _load_products() -> list:
    global _products_cache
    if _products_cache is None:
        with open(_DATA_PATH) as f:
            _products_cache = json.load(f)
    return _products_cache


def _find_product(product_name: str) -> dict | None:
    """Return first catalog entry matching the product name."""
    products = _load_products()
    name_lower = product_name.lower()
    for p in products:
        if p["name"].lower() == name_lower:
            return p
    for p in products:
        if name_lower in p["name"].lower() or any(name_lower in t for t in p.get("tags", [])):
            return p
    return None


def _get_llm():
    """Lazy-load IBM Granite LLM for review summarisation."""
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    import config
    from langchain_ibm import ChatWatsonx
    return ChatWatsonx(
        model_id="ibm/granite-13b-chat-v2",
        url=config.WATSONX_URL,
        api_key=config.WATSONX_API_KEY,
        project_id=config.WATSONX_PROJECT_ID,
        params={
            "max_new_tokens": 256,
            "temperature": 0.2,
        },
    )


def analyze_reviews(product_name: str) -> str:
    """
    Summarise reviews for a product using IBM Granite and return sentiment analysis.
    """
    product = _find_product(product_name)
    if not product:
        return f"No product found matching '{product_name}'."

    reviews = product.get("reviews", [])
    name = product["name"]
    rating = product.get("rating", "N/A")
    review_count = product.get("review_count", 0)

    if not reviews:
        return f"No review text available for '{name}'."

    # Build prompt for Granite
    reviews_text = "\n".join([f"- {r}" for r in reviews])
    prompt = (
        f"Product: {name}\n"
        f"Overall Rating: {rating}/5 from {review_count} reviews\n\n"
        f"Customer Reviews:\n{reviews_text}\n\n"
        "Task: Summarise these customer reviews in 2-3 sentences. "
        "Then rate the overall sentiment as exactly one of: Positive, Mixed, or Negative. "
        "Format your response as:\nSummary: <summary>\nSentiment: <Positive|Mixed|Negative>"
    )

    try:
        llm = _get_llm()
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=prompt)])
        llm_output = response.content.strip()
    except Exception as e:
        llm_output = f"(LLM unavailable: {e})\nSentiment: Mixed"

    # Fake-review heuristic
    fake_warning = ""
    if rating >= 4.9 and review_count > 500:
        fake_warning = (
            f"\n⚠️ **Fake Review Alert:** {review_count} reviews with {rating}/5 rating — "
            "unusually high; potential review inflation detected."
        )

    lines = [f"### Review Analysis: {name}"]
    lines.append(f"- **Rating:** {rating}/5 based on {review_count} reviews")
    lines.append(f"\n{llm_output}")
    if fake_warning:
        lines.append(fake_warning)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LangChain tool
# ---------------------------------------------------------------------------

@tool
def _review_analyzer_tool(product_name: str) -> str:
    """
    Analyse and summarise customer reviews for a product using IBM Granite AI.
    Detects overall sentiment (Positive/Mixed/Negative) and flags potential fake reviews.
    Input should be the product name, e.g. 'GlowSerum Vitamin C Serum' or 'PureAir HEPA Air Purifier'.
    Returns a review summary, sentiment rating, and any fake review warnings.
    """
    return analyze_reviews(product_name)


def get_review_analyzer_tool():
    """Return the review analyzer LangChain tool."""
    return _review_analyzer_tool
