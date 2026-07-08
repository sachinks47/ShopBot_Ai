# 🛍️ Intelligent Shopping Agent

> An agentic AI shopping assistant powered by **IBM watsonx.ai (Granite)** + **LangGraph** + **Streamlit**

---

## Overview

ShopBot is a fully-functional AI shopping companion that helps users find the best products,
compare prices across e-commerce platforms, analyse reviews, predict deal timing, and discover
sustainable alternatives — all through a conversational chat interface backed by IBM Granite.

### Key Features

| Feature | Description |
|---|---|
| 🔍 **Semantic Product Search** | Text-based search using `sentence-transformers` + FAISS |
| 🖼️ **Image Search** | Upload a photo to find visually similar products (CLIP) |
| 🎙️ **Voice Search (Demo)** | Simulated voice-to-text query input |
| 💰 **Price Comparison** | Multi-source price table across Amazon, Flipkart, Walmart |
| 📉 **Deal Predictor** | Rule-based price trend analysis + buy timing advice |
| ⭐ **Review Analyzer** | IBM Granite LLM-powered review summarisation + sentiment |
| 🌱 **Recommendation Engine** | Sustainable + rated alternatives within budget |
| 📊 **Smart Dashboard** | Plotly charts: price comparisons, sustainability leaderboard, trending |
| 🗂️ **Shopping List** | Persistent in-session shopping cart with total cost |

---

## Architecture

```
User (Streamlit UI)
    │
    ├── Text / Image / Voice Input
    ▼
LangGraph ReAct Agent  ←→  IBM watsonx.ai (ibm/granite-13b-chat-v2)
    │
    ├── Tool: product_search       (semantic text + CLIP image search)
    ├── Tool: price_comparator     (multi-source price aggregation)
    ├── Tool: deal_predictor       (price trend + fake review detection)
    ├── Tool: review_analyzer      (Granite LLM review summarisation)
    └── Tool: recommendation_engine (sustainable alternatives)
    │
    ▼
Mock Product Catalog (data/products.json — 50+ products, 4 categories)
    │
    ▼
Streamlit App (Chat Interface + Smart Shopping Pulse Dashboard)
```

---

## Project Structure

```
shopping_agent/
├── app.py                          # Streamlit entry point (Chat + Dashboard)
├── config.py                       # Env/credential loader
├── requirements.txt
├── .env.example                    # Credential template
├── data/
│   └── products.json               # Mock product catalog (50+ products)
├── agent/
│   ├── core.py                     # LangGraph agent + IBM Granite setup
│   └── tools/
│       ├── product_search.py       # Text & image search tool
│       ├── price_comparator.py     # Multi-source price comparison tool
│       ├── deal_predictor.py       # Deal timing + fake review tool
│       ├── review_analyzer.py      # LLM-powered review analysis tool
│       └── recommendation_engine.py# Sustainable alternatives tool
├── utils/
│   ├── image_search.py             # CLIP-based image similarity search
│   └── voice_stub.py               # Voice transcription stub
└── dashboard/
    └── visualizations.py           # Plotly chart builders
```

---

## Setup Instructions

### 1. Clone / Navigate to the project

```bash
cd shopping_agent
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure IBM watsonx.ai credentials

Copy the example env file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```
WATSONX_API_KEY=your_actual_api_key
WATSONX_PROJECT_ID=your_actual_project_id
WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

> **Get credentials at:** https://cloud.ibm.com/apidocs/watsonx-ai

### 5. Run the app

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## Demo Flows

### Flow 1 — Text Query (Chat)
1. Open the **Chat Assistant** page
2. Type: *"Find me a sustainable laptop under $1000"*
3. ShopBot uses `product_search` → returns matching products with eco scores
4. Click **"＋ Add to Shopping List"** on any product card

### Flow 2 — Image Search
1. Open the **Chat Assistant** page → expand **📷 Image Search**
2. Upload any product photo (laptop, headphones, etc.)
3. Click **🔍 Search by Image** → CLIP model finds similar catalog items

### Flow 3 — Price Comparison + Deal Prediction
1. Type: *"Compare prices for SoundMax Elite Headphones"*
2. Then: *"Is now a good time to buy?"*
3. ShopBot shows multi-source table + price trend analysis

### Flow 4 — Dashboard
1. Switch to **📊 Shopping Dashboard**
2. Use the dropdown to select any product for price comparison
3. Explore the Sustainability Leaderboard, Trending Products, and Category charts

---

## IBM Models Used

| Model | Purpose |
|---|---|
| `ibm/granite-13b-chat-v2` | Main chat agent reasoning + review summarisation |

Model is configurable in [`agent/core.py`](agent/core.py) and [`agent/tools/review_analyzer.py`](agent/tools/review_analyzer.py).

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | IBM watsonx.ai — `ibm/granite-13b-chat-v2` |
| Agent Framework | LangGraph `create_react_agent` + LangChain tools |
| Frontend | Streamlit |
| Semantic Search | `sentence-transformers` (MiniLM + CLIP) + FAISS |
| Visualisation | Plotly |
| Data | Mock JSON dataset (50+ products, 4 categories) |

---

## Feature Checklist

- [x] Multi-source product search (text + image + voice stub)
- [x] Price comparison across Amazon, Flipkart, Walmart
- [x] Deal timing prediction with price trend analysis
- [x] Fake review detection heuristic
- [x] LLM-powered review summarisation (IBM Granite)
- [x] Sustainable product recommendations
- [x] Smart Shopping Pulse dashboard (4 Plotly charts)
- [x] Personalized in-session shopping list
- [x] Conversational memory across turns (LangGraph MemorySaver)
- [x] Fully self-contained — no external APIs required beyond watsonx.ai
