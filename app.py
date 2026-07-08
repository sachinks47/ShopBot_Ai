"""
Intelligent Shopping Agent — Streamlit Application
Powered by IBM watsonx.ai (Granite) + LangGraph ReAct Agent

Run with:
    streamlit run app.py
"""

import sys
import os

# Ensure imports resolve from the shopping_agent directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from PIL import Image
import json
import io

# ---------------------------------------------------------------------------
# Page config (must be the very first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ShopBot — Intelligent Shopping Agent",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2328;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #57606a;
        margin-bottom: 1.5rem;
    }
    .product-card {
        background: #f7f8fa;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .badge-green {
        background: #dcfce7;
        color: #166534;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-blue {
        background: #dbeafe;
        color: #1e40af;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-amber {
        background: #fef3c7;
        color: #92400e;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .stChatMessage { border-radius: 8px; }
    section[data-testid="stSidebar"] { min-width: 260px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "shopping_list" not in st.session_state:
        st.session_state.shopping_list = []
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = "session_001"
    if "image_results" not in st.session_state:
        st.session_state.image_results = []


_init_session()


# ---------------------------------------------------------------------------
# Agent initialisation (lazy — first message triggers load)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading IBM Granite model…")
def _load_agent():
    """Load the LangGraph agent once and cache it for the session."""
    from agent.tools.product_search import get_product_search_tool
    from agent.tools.price_comparator import get_price_comparator_tool
    from agent.tools.deal_predictor import get_deal_predictor_tool
    from agent.tools.review_analyzer import get_review_analyzer_tool
    from agent.tools.recommendation_engine import get_recommendation_tool
    from agent.core import build_agent

    tools = [
        get_product_search_tool(),
        get_price_comparator_tool(),
        get_deal_predictor_tool(),
        get_review_analyzer_tool(),
        get_recommendation_tool(),
    ]
    return build_agent(tools)


# ---------------------------------------------------------------------------
# Helper: render a single product card
# ---------------------------------------------------------------------------

def render_product_card(product: dict, show_add_button: bool = True):
    """Render a styled product card inside a Streamlit container."""
    sustainability = product.get("sustainability_score", 0)
    in_stock = product.get("in_stock", True)

    # Sustainability badge colour
    if sustainability >= 8:
        badge_cls = "badge-green"
        badge_label = f"🌱 Eco Score: {sustainability}/10"
    elif sustainability >= 5:
        badge_cls = "badge-blue"
        badge_label = f"♻️ Eco Score: {sustainability}/10"
    else:
        badge_cls = "badge-amber"
        badge_label = f"Eco Score: {sustainability}/10"

    stock_icon = "✅ In Stock" if in_stock else "❌ Out of Stock"

    with st.container():
        st.markdown(
            f"""
            <div class="product-card">
                <strong>{product['name']}</strong> &nbsp;
                <span class="{badge_cls}">{badge_label}</span><br>
                <small>{product['brand']} &nbsp;|&nbsp; {product['category'].title()}</small><br>
                <strong style="color:#3b82d4; font-size:1.1rem">${product['price_usd']:.2f}</strong>
                &nbsp;— {product['source']} &nbsp;|&nbsp;
                ⭐ {product['rating']}/5 ({product['review_count']:,} reviews)
                &nbsp;|&nbsp; {stock_icon}<br>
                <small style="color:#57606a">{product['description'][:130]}...</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if show_add_button:
            btn_key = f"add_{product['id']}_{product['source']}"
            existing = [p["id"] for p in st.session_state.shopping_list]
            if product["id"] not in existing:
                if st.button(f"＋ Add to Shopping List", key=btn_key):
                    st.session_state.shopping_list.append(product)
                    st.success(f"Added **{product['name']}** to your shopping list!")
            else:
                st.caption("✓ Already in your shopping list")


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🛍️ ShopBot")
    st.markdown("<small style='color:#57606a'>Powered by IBM Granite on watsonx.ai</small>",
                unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigate",
        ["💬 Chat Assistant", "📊 Shopping Dashboard"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("### 🗂️ Shopping List")
    if st.session_state.shopping_list:
        for i, item in enumerate(st.session_state.shopping_list):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"<small>**{item['name']}**<br>${item['price_usd']} — {item['source']}</small>",
                            unsafe_allow_html=True)
            with col2:
                if st.button("✕", key=f"remove_{i}"):
                    st.session_state.shopping_list.pop(i)
                    st.rerun()
        st.divider()
        total = sum(p["price_usd"] for p in st.session_state.shopping_list)
        st.markdown(f"**Total: ${total:.2f}**")
        if st.button("🗑️ Clear List"):
            st.session_state.shopping_list = []
            st.rerun()
    else:
        st.caption("Your shopping list is empty.")

    st.divider()
    if st.button("🔄 Reset Chat"):
        st.session_state.messages = []
        st.session_state.image_results = []
        st.rerun()


# ===========================================================================
# PAGE 1 — CHAT ASSISTANT
# ===========================================================================

if page == "💬 Chat Assistant":
    st.markdown('<p class="main-header">🛍️ ShopBot — Intelligent Shopping Assistant</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Ask me to find products, compare prices, analyse reviews, '
        'predict deals, or recommend sustainable alternatives.</p>',
        unsafe_allow_html=True,
    )

    # Quick-action example prompts
    st.markdown("**💡 Try asking:**")
    cols = st.columns(4)
    examples = [
        "Find a sustainable laptop under $1000",
        "Compare prices for SoundMax Elite Headphones",
        "Is now a good time to buy PixelPhone X12?",
        "Recommend eco-friendly home products",
    ]
    for col, example in zip(cols, examples):
        with col:
            if st.button(example, use_container_width=True):
                st.session_state._prefill_query = example

    st.divider()

    # ---- Multimodal input area ----
    with st.expander("📷 Image Search / 🎙️ Voice Search", expanded=False):
        tab_img, tab_voice = st.tabs(["📷 Image Search", "🎙️ Voice Search (Demo)"])

        with tab_img:
            st.markdown("Upload a product photo to find visually similar items.")
            uploaded_file = st.file_uploader(
                "Choose an image", type=["jpg", "jpeg", "png", "webp"], key="img_uploader"
            )
            if uploaded_file is not None:
                img = Image.open(uploaded_file).convert("RGB")
                st.image(img, caption="Query Image", width=240)
                if st.button("🔍 Search by Image", key="btn_img_search"):
                    with st.spinner("Searching by image similarity (CLIP)…"):
                        try:
                            from utils.image_search import search_by_image
                            results = search_by_image(img, top_k=5)
                            st.session_state.image_results = results
                            # Also add a chat message so results appear in chat
                            st.session_state.messages.append({
                                "role": "user",
                                "content": "🖼️ [Image uploaded — searching for similar products]"
                            })
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": (
                                    f"I found **{len(results)}** product(s) visually similar to your image. "
                                    "See the product cards below the chat."
                                ),
                                "products": results,
                            })
                            st.rerun()
                        except Exception as e:
                            st.error(f"Image search failed: {e}")

        with tab_voice:
            st.markdown("Simulate a voice query — click a demo query to send it as speech input.")
            from utils.voice_stub import get_demo_queries
            demo_queries = get_demo_queries()
            selected_voice = st.selectbox("Select a demo voice query:", demo_queries)
            if st.button("🎙️ Send Voice Query", key="btn_voice"):
                st.session_state._prefill_query = selected_voice

    st.divider()

    # ---- Chat history display ----
    for msg in st.session_state.messages:
        role = msg["role"]
        with st.chat_message(role, avatar="🛍️" if role == "assistant" else "👤"):
            st.markdown(msg["content"])
            # Render product cards if attached to this message
            if msg.get("products"):
                for product in msg["products"][:5]:
                    render_product_card(product)

    # ---- Chat input ----
    prefill = st.session_state.pop("_prefill_query", None)
    user_input = st.chat_input("Ask ShopBot anything about products…") or prefill

    if user_input:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        # Call the agent
        with st.chat_message("assistant", avatar="🛍️"):
            with st.spinner("ShopBot is thinking…"):
                try:
                    agent = _load_agent()
                    from agent.core import run_agent
                    response = run_agent(
                        agent,
                        user_input,
                        thread_id=st.session_state.thread_id,
                    )
                except Exception as e:
                    response = (
                        f"⚠️ I encountered an error connecting to IBM watsonx.ai: `{e}`\n\n"
                        "Please check your `.env` credentials and try again."
                    )

            st.markdown(response)

            # Try to surface product results alongside the response
            product_hits = []
            try:
                from agent.tools.product_search import text_search
                keywords = [w for w in user_input.split() if len(w) > 3]
                if keywords:
                    product_hits = text_search(" ".join(keywords[:8]), top_k=3)
            except Exception:
                pass

            for product in product_hits:
                render_product_card(product)

        # Persist to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "products": product_hits,
        })

    # Show image search results if any (displayed below chat)
    if st.session_state.image_results:
        st.markdown("### 🖼️ Image Search Results")
        cols_img = st.columns(min(len(st.session_state.image_results), 3))
        for i, product in enumerate(st.session_state.image_results[:3]):
            with cols_img[i]:
                render_product_card(product)


# ===========================================================================
# PAGE 2 — SMART SHOPPING PULSE DASHBOARD
# ===========================================================================

elif page == "📊 Shopping Dashboard":
    st.markdown('<p class="main-header">📊 Smart Shopping Pulse</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Real-time insights: price comparisons, sustainability rankings, '
        'trending products, and your personalized shopping list.</p>',
        unsafe_allow_html=True,
    )

    from dashboard.visualizations import (
        plot_price_comparison,
        plot_sustainability_leaderboard,
        plot_trending_products,
        plot_category_distribution,
        get_unique_product_names,
        get_all_products_df,
    )

    # ---- Row 1: Price Comparison + Category Distribution ----
    st.markdown("### 💰 Price Comparison")
    product_names = get_unique_product_names()
    selected_product = st.selectbox(
        "Select a product to compare prices across stores:",
        product_names,
        key="dash_product_select",
    )

    col_chart, col_meta = st.columns([3, 1])
    with col_chart:
        fig_price = plot_price_comparison(selected_product)
        if fig_price:
            st.plotly_chart(fig_price, use_container_width=True)
        else:
            st.info("No price comparison data available for this product.")

    with col_meta:
        # Show quick stats for selected product
        df = get_all_products_df()
        product_rows = df[df["name"].str.lower() == selected_product.lower()]
        if not product_rows.empty:
            p = product_rows.iloc[0]
            st.markdown(f"**{p['name']}**")
            st.metric("Best Price", f"${product_rows['price_usd'].min():.2f}")
            st.metric("Avg Rating", f"{product_rows['rating'].max():.1f} ⭐")
            st.metric("Sustainability", f"{p['sustainability_score']}/10 🌱")
            # show deal predictor verdict inline
            try:
                from agent.tools.deal_predictor import analyze_price_trend
                verdict = analyze_price_trend(selected_product)
                # Extract just the verdict line
                for line in verdict.split("\n"):
                    if any(icon in line for icon in ["🟢", "🟡", "🔴"]):
                        st.markdown(line)
                        break
            except Exception:
                pass

    st.divider()

    # ---- Row 2: Sustainability Leaderboard ----
    st.markdown("### 🌱 Sustainability Leaderboard")
    fig_sustain = plot_sustainability_leaderboard(top_n=10)
    st.plotly_chart(fig_sustain, use_container_width=True)

    st.divider()

    # ---- Row 3: Trending + Category split ----
    col_trend, col_cat = st.columns([3, 2])
    with col_trend:
        st.markdown("### 🔥 Trending Products")
        fig_trend = plot_trending_products(top_n=10)
        st.plotly_chart(fig_trend, use_container_width=True)
    with col_cat:
        st.markdown("### 📦 Category Breakdown")
        fig_cat = plot_category_distribution()
        st.plotly_chart(fig_cat, use_container_width=True)

    st.divider()

    # ---- Row 4: Shopping List Panel ----
    st.markdown("### 🗂️ Your Shopping List")
    if st.session_state.shopping_list:
        import pandas as pd
        list_df = pd.DataFrame([
            {
                "Product": p["name"],
                "Brand": p["brand"],
                "Price ($)": p["price_usd"],
                "Source": p["source"],
                "Rating": f"{p['rating']}/5",
                "Sustainability": f"{p['sustainability_score']}/10",
                "In Stock": "✅" if p.get("in_stock", True) else "❌",
            }
            for p in st.session_state.shopping_list
        ])
        st.dataframe(list_df, use_container_width=True, hide_index=True)
        total = sum(p["price_usd"] for p in st.session_state.shopping_list)
        avg_sustain = sum(p["sustainability_score"] for p in st.session_state.shopping_list) / len(st.session_state.shopping_list)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Cost", f"${total:.2f}")
        c2.metric("Items", len(st.session_state.shopping_list))
        c3.metric("Avg Sustainability", f"{avg_sustain:.1f}/10 🌱")
    else:
        st.info("Your shopping list is empty. Add products from the Chat Assistant.")

    st.divider()

    # ---- Row 5: Full Catalog Explorer ----
    st.markdown("### 🔍 Full Product Catalog")
    df_all = get_all_products_df()

    # Filters
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        cat_filter = st.multiselect("Filter by Category", df_all["category"].unique().tolist())
    with filter_col2:
        src_filter = st.multiselect("Filter by Source", df_all["source"].unique().tolist())
    with filter_col3:
        min_sustain = st.slider("Min Sustainability Score", 1, 10, 1)

    filtered_df = df_all.copy()
    if cat_filter:
        filtered_df = filtered_df[filtered_df["category"].isin(cat_filter)]
    if src_filter:
        filtered_df = filtered_df[filtered_df["source"].isin(src_filter)]
    filtered_df = filtered_df[filtered_df["sustainability_score"] >= min_sustain]

    display_cols = ["name", "brand", "category", "price_usd", "source", "rating",
                    "sustainability_score", "in_stock"]
    st.dataframe(
        filtered_df[display_cols].rename(columns={
            "name": "Product", "brand": "Brand", "category": "Category",
            "price_usd": "Price ($)", "source": "Store", "rating": "Rating",
            "sustainability_score": "Eco Score", "in_stock": "In Stock",
        }),
        use_container_width=True,
        hide_index=True,
    )
