"""
Recommendation Engine Tool — suggests alternative products based on
category, budget, and sustainability preference.
Uses rule-based filtering + sorting (no LLM call).
"""

import json
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


def find_alternatives(
    product_name: str = "",
    category: str = "",
    budget: float = 0.0,
    prefer_sustainable: bool = False,
    top_k: int = 3,
) -> list:
    """
    Find alternative products matching the given criteria.

    Parameters
    ----------
    product_name : str
        Name of the reference product (used to infer category/budget if not provided).
    category : str
        Product category to filter by (electronics, clothing, home, beauty).
    budget : float
        Maximum price in USD. If 0, uses reference product price ±20%.
    prefer_sustainable : bool
        If True, sort by sustainability_score descending; else by rating.
    top_k : int
        Number of alternatives to return.

    Returns
    -------
    list of dict
        Alternative product records.
    """
    products = _load_products()

    # Resolve reference product if name given
    reference = None
    if product_name:
        name_lower = product_name.lower()
        for p in products:
            if p["name"].lower() == name_lower or name_lower in p["name"].lower():
                reference = p
                break

    # Infer category and budget from reference if not explicitly provided
    if reference:
        if not category:
            category = reference["category"]
        if budget == 0.0:
            budget = reference["price_usd"] * 1.2  # up to 20% more expensive

    if not category and budget == 0.0:
        # Fallback: return top-rated products overall
        sorted_all = sorted(products, key=lambda x: x["rating"], reverse=True)
        seen = set()
        result = []
        for p in sorted_all:
            if p["name"] not in seen:
                seen.add(p["name"])
                result.append(p)
            if len(result) >= top_k:
                break
        return result

    # Filter by category and budget
    cat_lower = category.lower()
    candidates = [
        p for p in products
        if (not category or p["category"].lower() == cat_lower)
        and (budget == 0.0 or p["price_usd"] <= budget)
        and (not reference or p["name"] != reference["name"])  # exclude the reference itself
    ]

    # Deduplicate by name (keep best rating per name)
    seen = {}
    for p in candidates:
        name = p["name"]
        if name not in seen or p["rating"] > seen[name]["rating"]:
            seen[name] = p
    candidates = list(seen.values())

    # Sort
    if prefer_sustainable:
        candidates.sort(key=lambda x: (x["sustainability_score"], x["rating"]), reverse=True)
    else:
        candidates.sort(key=lambda x: x["rating"], reverse=True)

    return candidates[:top_k]


def format_recommendations(
    product_name: str = "",
    category: str = "",
    budget: float = 0.0,
    prefer_sustainable: bool = False,
) -> str:
    """Return a formatted string of recommended alternatives."""
    alts = find_alternatives(
        product_name=product_name,
        category=category,
        budget=budget,
        prefer_sustainable=prefer_sustainable,
    )

    if not alts:
        return "No alternative products found matching your criteria."

    ref_label = f"alternatives to '{product_name}'" if product_name else f"top picks in '{category}'"
    lines = [f"### Recommended Products ({ref_label})\n"]

    for i, p in enumerate(alts, 1):
        reason_parts = []
        if prefer_sustainable:
            reason_parts.append(f"sustainability score {p['sustainability_score']}/10")
        reason_parts.append(f"rated {p['rating']}/5")
        if p["price_usd"] <= (budget * 0.8) if budget else False:
            reason_parts.append("well within budget")
        reason = ", ".join(reason_parts)

        stock = "✅" if p.get("in_stock", True) else "❌ Out of Stock"
        lines.append(
            f"{i}. **{p['name']}** by {p['brand']}  {stock}\n"
            f"   💰 ${p['price_usd']} ({p['source']})  "
            f"🌱 Sustainability: {p['sustainability_score']}/10  "
            f"⭐ {p['rating']}/5\n"
            f"   _{p['description'][:100]}..._\n"
            f"   Why recommended: {reason}\n"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LangChain tool
# ---------------------------------------------------------------------------

@tool
def _recommendation_tool(query: str) -> str:
    """
    Recommend alternative or similar products based on user criteria.
    Input should describe what the user wants, e.g.:
    - 'sustainable alternatives to UltraBook Pro 15'
    - 'best electronics under 500 dollars'
    - 'eco-friendly home products'
    - 'clothing alternatives prefer sustainable'
    Returns top-3 recommended products with reasons.
    """
    query_lower = query.lower()

    # Parse sustainability preference
    prefer_sustainable = any(
        kw in query_lower for kw in ["sustainable", "eco", "green", "recycled", "organic"]
    )

    # Parse budget
    budget = 0.0
    import re
    budget_match = re.search(r"under\s+\$?(\d+)", query_lower)
    if budget_match:
        budget = float(budget_match.group(1))

    # Parse category
    category = ""
    for cat in ["electronics", "clothing", "home", "beauty"]:
        if cat in query_lower:
            category = cat
            break

    # Parse reference product name (look for 'alternatives to ...')
    product_name = ""
    alt_match = re.search(r"alternatives?\s+to\s+([A-Za-z0-9 ]+?)(?:\s+under|\s+prefer|$)", query_lower)
    if alt_match:
        product_name = alt_match.group(1).strip().title()

    return format_recommendations(
        product_name=product_name,
        category=category,
        budget=budget,
        prefer_sustainable=prefer_sustainable,
    )


def get_recommendation_tool():
    """Return the recommendation engine LangChain tool."""
    return _recommendation_tool
