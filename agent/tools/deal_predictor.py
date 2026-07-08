"""
Deal Predictor Tool — analyses discount history to predict buy timing,
flag potential fake reviews, and alert about price drops.
All logic is rule-based (no LLM call) for deterministic, fast responses.
"""

import json
from datetime import datetime
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


def _find_product(product_name: str) -> list:
    """Return all catalog entries matching the given product name (fuzzy)."""
    products = _load_products()
    name_lower = product_name.lower()
    matches = [p for p in products if p["name"].lower() == name_lower]
    if not matches:
        matches = [p for p in products if name_lower in p["name"].lower()
                   or any(name_lower in tag for tag in p.get("tags", []))]
    return matches


def analyze_price_trend(product_name: str) -> str:
    """
    Analyse historical pricing to advise on buy timing.

    Rules:
    - If current price <= historical minimum → "Price at historic low — great time to buy!"
    - If current price < average → "Price is below average — good time to buy."
    - If current price > average by >10% → "Price is above average — consider waiting."
    - If trend shows 2+ consecutive drops → "Price is falling — a further drop may be coming."
    """
    matches = _find_product(product_name)
    if not matches:
        return f"No product found matching '{product_name}'."

    # Use the listing with the most history
    product = max(matches, key=lambda p: len(p.get("discount_history", [])))
    history = product.get("discount_history", [])
    current_price = product["price_usd"]
    product_name_actual = product["name"]

    if not history:
        return f"No price history available for '{product_name_actual}'."

    # Parse and sort history by date
    try:
        parsed = sorted(
            [{"date": datetime.strptime(h["date"], "%Y-%m-%d"), "price": h["price"]} for h in history],
            key=lambda x: x["date"],
        )
    except Exception:
        parsed = history

    historic_prices = [h["price"] for h in parsed]
    historic_min = min(historic_prices)
    historic_max = max(historic_prices)
    avg_price = sum(historic_prices) / len(historic_prices)

    lines = [f"### Deal Analysis: {product_name_actual}"]
    lines.append(f"- **Current price:** ${current_price:.2f}")
    lines.append(f"- **Historical average:** ${avg_price:.2f}")
    lines.append(f"- **All-time low:** ${historic_min:.2f}  |  **All-time high:** ${historic_max:.2f}")

    # Trend direction (last 2 history points)
    trend_msg = ""
    if len(parsed) >= 2:
        last_two = [h["price"] for h in parsed[-2:]]
        if last_two[1] < last_two[0]:
            trend_msg = "📉 Price has been **falling** recently."
        elif last_two[1] > last_two[0]:
            trend_msg = "📈 Price has been **rising** recently."
        else:
            trend_msg = "➡️ Price has been **stable** recently."

    # Buy recommendation
    if current_price <= historic_min * 1.02:
        verdict = "🟢 **Price is at or near its historic low — excellent time to buy!**"
    elif current_price <= avg_price * 0.97:
        verdict = "🟡 **Price is below average — good time to buy.**"
    elif current_price >= avg_price * 1.10:
        verdict = "🔴 **Price is above average — consider waiting for a discount.**"
    else:
        verdict = "🟡 **Price is around average — reasonable time to buy.**"

    # Seasonal hint
    month = datetime.now().month
    seasonal = ""
    if month == 11:
        seasonal = "💡 **November tip:** Black Friday deals are likely soon — consider waiting a few days."
    elif month == 12:
        seasonal = "💡 **December tip:** Holiday sales are active — good time to shop."
    elif month in (6, 7):
        seasonal = "💡 **Summer tip:** Mid-year sales often run in June–July."

    if trend_msg:
        lines.append(f"- **Trend:** {trend_msg}")
    lines.append(f"\n{verdict}")
    if seasonal:
        lines.append(seasonal)

    # Fake review check
    fake_warning = _check_fake_reviews(product)
    if fake_warning:
        lines.append(f"\n⚠️ **Review Alert:** {fake_warning}")

    return "\n".join(lines)


def _check_fake_reviews(product: dict) -> str:
    """
    Heuristic fake-review detection.
    Flags if review_count is very high AND rating is suspiciously perfect (>= 4.9).
    Also flags abnormally low review counts with perfect ratings.
    """
    rating = product.get("rating", 0)
    review_count = product.get("review_count", 0)

    if rating >= 4.9 and review_count > 500:
        return (
            f"This product has {review_count} reviews with a near-perfect {rating}/5 rating. "
            "This pattern may indicate inflated or fake reviews. Verify on the retailer's site."
        )
    if rating == 5.0 and review_count < 50:
        return (
            f"Only {review_count} reviews with a perfect 5.0 rating. "
            "Too few reviews to be reliable — treat with caution."
        )
    return ""


# ---------------------------------------------------------------------------
# LangChain tool
# ---------------------------------------------------------------------------

@tool
def _deal_predictor_tool(product_name: str) -> str:
    """
    Analyse price history and predict if now is a good time to buy a product.
    Also detects potential fake reviews and seasonal discount opportunities.
    Input should be the product name, e.g. 'PixelPhone X12' or 'SoundMax Elite Headphones'.
    Returns a deal analysis with buy recommendation, price trend, and any review warnings.
    """
    return analyze_price_trend(product_name)


def get_deal_predictor_tool():
    """Return the deal predictor LangChain tool."""
    return _deal_predictor_tool
