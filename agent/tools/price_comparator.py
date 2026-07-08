"""
Price Comparator Tool — aggregates prices for a product across simulated e-commerce sources.
All data is sourced from the local mock products.json dataset.
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


def compare_prices(product_name: str) -> dict:
    """
    Find all listings of a product (by name, case-insensitive fuzzy match)
    and return a comparison dict keyed by source.
    """
    products = _load_products()
    name_lower = product_name.lower()

    # First try exact match, then substring match
    matches = [p for p in products if p["name"].lower() == name_lower]
    if not matches:
        matches = [p for p in products if name_lower in p["name"].lower()
                   or any(name_lower in tag for tag in p.get("tags", []))]

    if not matches:
        return {}

    # Group by source
    by_source = {}
    for p in matches:
        src = p["source"]
        if src not in by_source or p["price_usd"] < by_source[src]["price_usd"]:
            by_source[src] = p

    return by_source


def format_comparison_table(product_name: str) -> str:
    """Return a markdown-formatted price comparison table for the given product."""
    by_source = compare_prices(product_name)
    if not by_source:
        return f"No listings found for '{product_name}' in the catalog."

    rows = sorted(by_source.values(), key=lambda x: x["price_usd"])
    best = rows[0]
    prices = [r["price_usd"] for r in rows]
    avg_price = sum(prices) / len(prices)

    lines = [f"## Price Comparison: {rows[0]['name']}\n"]
    lines.append(f"| Store       | Price (USD) | Rating | Stock       |")
    lines.append(f"|-------------|-------------|--------|-------------|")
    for r in rows:
        stock = "✅ In Stock" if r.get("in_stock", True) else "❌ Out of Stock"
        marker = " ⭐ Best" if r["source"] == best["source"] else ""
        lines.append(
            f"| {r['source']:<11} | ${r['price_usd']:<11.2f} | {r['rating']}/5  | {stock}{marker} |"
        )

    lines.append(f"\n**Average price:** ${avg_price:.2f}")
    lines.append(f"**Best deal:** {best['source']} at ${best['price_usd']:.2f} "
                 f"(saves ${avg_price - best['price_usd']:.2f} vs. average)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LangChain tool
# ---------------------------------------------------------------------------

@tool
def _price_comparator_tool(product_name: str) -> str:
    """
    Compare prices for a specific product across multiple e-commerce stores (Amazon, Flipkart, Walmart).
    Input should be the product name, e.g. 'UltraBook Pro 15' or 'SoundMax Elite Headphones'.
    Returns a price comparison table with the best deal highlighted.
    """
    return format_comparison_table(product_name)


def get_price_comparator_tool():
    """Return the price comparator LangChain tool."""
    return _price_comparator_tool
