"""
Dashboard Visualizations — Plotly chart helpers for the Smart Shopping Pulse dashboard.
All charts are built from the local products.json mock dataset.
"""

import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

_DATA_PATH = Path(__file__).parent.parent / "data" / "products.json"
_df_cache = None


def _load_df() -> pd.DataFrame:
    """Load products.json into a pandas DataFrame (cached)."""
    global _df_cache
    if _df_cache is None:
        with open(_DATA_PATH) as f:
            data = json.load(f)
        _df_cache = pd.DataFrame(data)
    return _df_cache


# ---------------------------------------------------------------------------
# 1. Price Comparison Chart
# ---------------------------------------------------------------------------

def plot_price_comparison(product_name: str):
    """
    Bar chart comparing prices for a product across sources.

    Parameters
    ----------
    product_name : str
        The exact or partial name of the product.

    Returns
    -------
    plotly.Figure or None
    """
    df = _load_df()
    name_lower = product_name.lower()
    filtered = df[df["name"].str.lower().str.contains(name_lower, na=False)]

    if filtered.empty:
        return None

    # Best price per source
    best = filtered.groupby("source", as_index=False).agg(
        price_usd=("price_usd", "min"),
        rating=("rating", "max"),
        in_stock=("in_stock", "first"),
    ).sort_values("price_usd")

    product_title = filtered.iloc[0]["name"]
    colors = ["#3b82d4" if row["in_stock"] else "#9ca3af" for _, row in best.iterrows()]

    fig = go.Figure(
        go.Bar(
            x=best["source"],
            y=best["price_usd"],
            marker_color=colors,
            text=[f"${p:.2f}" for p in best["price_usd"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Price: $%{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Price Comparison: {product_title}",
        xaxis_title="Store",
        yaxis_title="Price (USD)",
        plot_bgcolor="#f7f8fa",
        paper_bgcolor="#ffffff",
        font=dict(family="Segoe UI, sans-serif", size=13),
        margin=dict(t=60, b=40, l=40, r=20),
        showlegend=False,
    )
    # Add avg line
    avg = best["price_usd"].mean()
    fig.add_hline(
        y=avg,
        line_dash="dot",
        line_color="#e74c3c",
        annotation_text=f"Avg ${avg:.2f}",
        annotation_position="top right",
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Sustainability Leaderboard
# ---------------------------------------------------------------------------

def plot_sustainability_leaderboard(top_n: int = 10):
    """
    Horizontal bar chart of top products by sustainability score.
    Deduplicates by product name (best score per name).
    """
    df = _load_df()

    # Best sustainability score per product name
    best = df.groupby("name", as_index=False).agg(
        sustainability_score=("sustainability_score", "max"),
        category=("category", "first"),
        brand=("brand", "first"),
    ).sort_values("sustainability_score", ascending=False).head(top_n)

    category_colors = {
        "electronics": "#3b82d4",
        "clothing": "#7c5cd8",
        "home": "#22c55e",
        "beauty": "#f59e0b",
    }
    colors = [category_colors.get(c, "#9ca3af") for c in best["category"]]

    fig = go.Figure(
        go.Bar(
            x=best["sustainability_score"],
            y=best["name"],
            orientation="h",
            marker_color=colors,
            text=best["sustainability_score"].astype(str) + "/10",
            textposition="inside",
            hovertemplate="<b>%{y}</b><br>Score: %{x}/10<extra></extra>",
        )
    )
    fig.update_layout(
        title="🌱 Sustainability Leaderboard",
        xaxis_title="Sustainability Score (out of 10)",
        yaxis_title="",
        plot_bgcolor="#f7f8fa",
        paper_bgcolor="#ffffff",
        font=dict(family="Segoe UI, sans-serif", size=12),
        margin=dict(t=60, b=40, l=200, r=40),
        xaxis=dict(range=[0, 11]),
    )

    # Legend annotation
    legend_items = [
        ("electronics", "#3b82d4"),
        ("clothing", "#7c5cd8"),
        ("home", "#22c55e"),
        ("beauty", "#f59e0b"),
    ]
    for i, (label, color) in enumerate(legend_items):
        fig.add_annotation(
            x=10.5, y=top_n - 1 - i * 1.2,
            text=f"<span style='color:{color}'>■</span> {label}",
            showarrow=False, xanchor="left", font=dict(size=11),
        )

    return fig


# ---------------------------------------------------------------------------
# 3. Trending Products
# ---------------------------------------------------------------------------

def plot_trending_products(top_n: int = 10):
    """
    Bubble/bar chart of trending products sorted by review_count (proxy for popularity).
    Deduplicates by product name.
    """
    df = _load_df()

    trending = df.groupby("name", as_index=False).agg(
        review_count=("review_count", "max"),
        rating=("rating", "max"),
        category=("category", "first"),
        price_usd=("price_usd", "min"),
    ).sort_values("review_count", ascending=False).head(top_n)

    fig = px.scatter(
        trending,
        x="review_count",
        y="rating",
        size="review_count",
        color="category",
        text="name",
        hover_data={"price_usd": True, "review_count": True},
        color_discrete_map={
            "electronics": "#3b82d4",
            "clothing": "#7c5cd8",
            "home": "#22c55e",
            "beauty": "#f59e0b",
        },
        title="🔥 Trending Products (by Popularity & Rating)",
        labels={"review_count": "Number of Reviews", "rating": "Average Rating (out of 5)"},
    )
    fig.update_traces(
        textposition="top center",
        marker=dict(opacity=0.8, line=dict(width=1, color="white")),
    )
    fig.update_layout(
        plot_bgcolor="#f7f8fa",
        paper_bgcolor="#ffffff",
        font=dict(family="Segoe UI, sans-serif", size=12),
        margin=dict(t=60, b=60, l=60, r=40),
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Category Distribution (bonus panel)
# ---------------------------------------------------------------------------

def plot_category_distribution():
    """Pie chart of products per category."""
    df = _load_df()
    counts = df.groupby("category", as_index=False).agg(
        count=("id", "nunique")
    )
    fig = px.pie(
        counts,
        names="category",
        values="count",
        title="📦 Product Distribution by Category",
        color="category",
        color_discrete_map={
            "electronics": "#3b82d4",
            "clothing": "#7c5cd8",
            "home": "#22c55e",
            "beauty": "#f59e0b",
        },
        hole=0.35,
    )
    fig.update_traces(textinfo="label+percent")
    fig.update_layout(
        paper_bgcolor="#ffffff",
        font=dict(family="Segoe UI, sans-serif", size=12),
        margin=dict(t=60, b=20, l=20, r=20),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Helper: get all unique product names for dropdowns
# ---------------------------------------------------------------------------

def get_unique_product_names() -> list:
    df = _load_df()
    return sorted(df["name"].unique().tolist())


def get_all_products_df() -> pd.DataFrame:
    return _load_df()
