# Intelligent Shopping Agent — Implementation Plan

## Top-Level Overview

Build a fully-functional, agentic AI shopping assistant using:
- **Python** as the primary language
- **IBM watsonx.ai** (Granite models) as the LLM backbone
- **LangChain** as the agent orchestration framework (tool-calling ReAct agents)
- **Streamlit** as the single-app frontend (chat interface + smart shopping dashboard)
- **Mock product dataset** (JSON/CSV) simulating multi-source e-commerce data
- **Multimodal input**: text + image search functional; voice stubbed/simulated

The system operates as a conversational agent with specialized tools for product search,
price comparison, review analysis, deal prediction, and dashboard visualization.

---

## Architecture Summary

```
User (Streamlit UI)
    |
    |-- Text / Image / Voice Input
    v
LangChain ReAct Agent
    |-- IBM watsonx.ai Granite LLM
    |-- Tool: product_search (text + image similarity)
    |-- Tool: price_comparator (multi-source aggregation)
    |-- Tool: review_analyzer (fake review detection, sentiment)
    |-- Tool: deal_predictor (discount timing, price drop alerts)
    |-- Tool: recommendation_engine (alternatives, sustainability)
    |
    v
Mock Product Database (JSON/CSV)
    |
    v
Streamlit Dashboard (comparisons, trends, sustainability, shopping lists)
```

---

## Sub-Tasks

---

### Sub-Task 1 — Project Scaffold & Mock Data Layer

**Intent**
Establish the project structure, dependencies, configuration, and mock product dataset that
all other components will build upon. This is the foundation layer.

**Expected Outcomes**
- A clean directory structure with all modules in place (empty stubs ok)
- `requirements.txt` listing all dependencies
- `.env.example` with IBM watsonx.ai credential placeholders
- `data/products.json` — mock dataset with 50+ products across categories (electronics,
  clothing, home, beauty) containing: name, brand, category, price_usd, source (site name),
  rating (1–5), review_count, sustainability_score (1–10), tags, image_url (placeholder),
  discount_history (list of past prices with dates)
- `config.py` — loads env vars (WATSONX_API_KEY, WATSONX_PROJECT_ID, WATSONX_URL)

**Todo List**
1. Create the top-level project directory layout:
   ```
   shopping_agent/
   ├── app.py                  # Streamlit entry point
   ├── config.py               # Env/config loader
   ├── agent/
   │   ├── __init__.py
   │   ├── core.py             # LangChain agent initialization
   │   └── tools/
   │       ├── __init__.py
   │       ├── product_search.py
   │       ├── price_comparator.py
   │       ├── review_analyzer.py
   │       ├── deal_predictor.py
   │       └── recommendation_engine.py
   ├── data/
   │   └── products.json
   ├── utils/
   │   ├── __init__.py
   │   ├── image_search.py     # Image feature extraction utility
   │   └── voice_stub.py       # Voice input stub
   ├── dashboard/
   │   ├── __init__.py
   │   └── visualizations.py   # Streamlit chart/panel helpers
   ├── .env.example
   └── requirements.txt
   ```
2. Write `requirements.txt` with: `langchain`, `langchain-ibm`, `ibm-watsonx-ai`,
   `streamlit`, `pandas`, `plotly`, `pillow`, `sentence-transformers`, `scikit-learn`,
   `python-dotenv`, `numpy`, `faiss-cpu`
3. Write `.env.example` with `WATSONX_API_KEY=`, `WATSONX_PROJECT_ID=`, `WATSONX_URL=`
4. Write `config.py` to load env vars via `python-dotenv`
5. Generate `data/products.json` with 50+ realistic mock products covering 4+ categories,
   each entry having all required fields including `discount_history`

**Relevant Context**
- IBM watsonx.ai Python SDK: `ibm-watsonx-ai` package
- LangChain IBM integration: `langchain-ibm` (WatsonxLLM class)
- No docker — all Python, run locally

**Status** `[x] done`

---

### Sub-Task 2 — IBM watsonx.ai LLM Integration & LangChain Agent Core

**Intent**
Wire up the IBM Granite model via LangChain and build the central ReAct agent that
will orchestrate all specialized tools. This is the intelligence core of the system.

**Expected Outcomes**
- `agent/core.py` initializes a LangChain ReAct agent backed by `ibm/granite-13b-chat-v2`
  (or `ibm/granite-3-8b-instruct`) on watsonx.ai
- The agent can accept a user query string and return a structured response
- Tool registration scaffolding in place (tools wired to the agent even if stubs)
- A simple CLI smoke-test confirming the agent responds to "Find me a good laptop under $800"

**Todo List**
1. In `agent/core.py`, instantiate `WatsonxLLM` from `langchain_ibm` using credentials
   from `config.py`; set model to `ibm/granite-13b-chat-v2`
2. Define a system prompt that instructs the agent to behave as a smart shopping assistant
3. Initialize a LangChain `initialize_agent` with `AgentType.ZERO_SHOT_REACT_DESCRIPTION`
   and register all tool functions from `agent/tools/`
4. Expose a `run_agent(query: str, context: dict) -> str` function
5. Add a `__main__` block in `core.py` for smoke-testing the agent end-to-end

**Relevant Context**
- `langchain_ibm.WatsonxLLM` — the LangChain wrapper for watsonx.ai models
- `langchain.agents.initialize_agent` and `AgentType.ZERO_SHOT_REACT_DESCRIPTION`
- Credentials loaded from `config.py`
- Tools registered here will be implemented in Sub-Tasks 3–6

**Status** `[x] done`

---

### Sub-Task 3 — Product Search Tool (Text + Image)

**Intent**
Implement the product search tool that the agent can invoke. Supports text-based semantic
search and image-based similarity search over the mock product catalog.

**Expected Outcomes**
- `agent/tools/product_search.py` implements a LangChain `Tool` named `product_search`
- Text search: uses `sentence-transformers` to embed queries and find top-N matching
  products via cosine similarity against pre-embedded product descriptions
- Image search: `utils/image_search.py` accepts a PIL image, extracts a CLIP-like embedding
  (using `sentence-transformers` CLIP model), and returns top-N visually similar products
- Voice: `utils/voice_stub.py` accepts a string (simulated transcript) and routes to
  text search
- Tool returns a structured JSON string: list of matching products with key fields

**Todo List**
1. In `utils/image_search.py`, use `sentence-transformers` (`clip-ViT-B-32` model) to
   encode product descriptions and a query image; build an in-memory FAISS index at startup
2. In `agent/tools/product_search.py`, build product text embeddings from `products.json`
   descriptions at module load time using `all-MiniLM-L6-v2`
3. Implement `text_search(query: str, top_k: int = 5) -> list` using FAISS cosine search
4. Implement `image_search(image: PIL.Image, top_k: int = 5) -> list` routing through
   `utils/image_search.py`
5. Wrap as a LangChain `Tool` with a clear description so the agent knows when to call it
6. In `utils/voice_stub.py`, implement `transcribe_stub(audio_path: str) -> str` that
   returns a hardcoded or filename-derived query string (stub)

**Relevant Context**
- `sentence-transformers` library: `SentenceTransformer('all-MiniLM-L6-v2')` for text,
  `SentenceTransformer('clip-ViT-B-32')` for image
- `faiss-cpu` for fast nearest-neighbor search
- Mock product catalog in `data/products.json`

**Status** `[x] done`

---

### Sub-Task 4 — Price Comparator & Deal Predictor Tools

**Intent**
Implement two agent tools: one that aggregates and compares prices for a product across
simulated e-commerce sources, and another that predicts deal timing / alerts the user to
price drops or suspicious pricing patterns.

**Expected Outcomes**
- `agent/tools/price_comparator.py` — LangChain `Tool` named `price_comparator`
  Returns a comparison table (as formatted string) of the same product across multiple
  "sources" (Amazon, Flipkart, Walmart — all simulated from the mock dataset)
- `agent/tools/deal_predictor.py` — LangChain `Tool` named `deal_predictor`
  Analyzes `discount_history` in the mock data to predict if now is a good time to buy,
  whether a price drop is likely soon, and whether any seasonal offer applies
- Both tools return human-readable strings the agent can embed in its response

**Todo List**
1. In `price_comparator.py`, filter `products.json` by product name/category and group
   results by `source` field; compute lowest price, average price, best deal source
2. Format comparator output as a markdown table string
3. In `deal_predictor.py`, implement `analyze_price_trend(product_name: str) -> str`:
   - Look up `discount_history` for the matched product
   - Compute average historical price, current price delta, trend direction
   - Use simple rule logic (not LLM) to output: "Good time to buy", "Wait for discount",
     or "Price at historic low"
4. Add a fake-review alert heuristic in `deal_predictor.py`: if `review_count` is very
   high but `rating` suspiciously uniform (e.g., all 5.0s on many reviews), flag as
   "Potential fake reviews detected"
5. Wrap each as a LangChain `Tool` with descriptive docstrings

**Relevant Context**
- Mock `products.json` `source` field simulates multi-platform data
- `discount_history` field is a list of `{date, price}` objects
- Rule-based logic only — no LLM call inside these tools

**Status** `[x] done`

---

### Sub-Task 5 — Review Analyzer & Recommendation Engine Tools

**Intent**
Implement the sentiment/review analysis tool and the recommendation engine tool that
the agent uses to surface alternatives, sustainability scores, and personalized lists.

**Expected Outcomes**
- `agent/tools/review_analyzer.py` — LangChain `Tool` named `review_analyzer`
  Uses the IBM Granite LLM (via a direct watsonx.ai call) to summarize product reviews
  and detect tone/quality signals from a product's mock review text
- `agent/tools/recommendation_engine.py` — LangChain `Tool` named `recommendation_engine`
  Returns top-3 alternative products based on category + sustainability score + price range

**Todo List**
1. Enrich `data/products.json` with a `reviews` field (3–5 short mock review strings per
   product) as part of this sub-task
2. In `review_analyzer.py`, call watsonx.ai Granite directly with a prompt:
   "Summarize these reviews and rate sentiment as Positive/Mixed/Negative: {reviews}"
3. Append a fake-review warning based on the heuristic from Sub-Task 4
4. In `recommendation_engine.py`, implement `find_alternatives(product_name, budget, prefer_sustainable)`:
   - Filter by same category, price within ±20% of budget
   - Sort by sustainability_score desc if `prefer_sustainable=True`, else by rating desc
   - Return top-3 as formatted list with reason for recommendation
5. Wrap both as LangChain `Tool` objects

**Relevant Context**
- Direct watsonx.ai call reuses the `WatsonxLLM` instance from `agent/core.py`
- `sustainability_score` field in `products.json` (1–10)
- Recommendation is rule-based filtering + sorting; LLM only for review summary

**Status** `[x] done`

---

### Sub-Task 6 — Streamlit App: Chat Interface

**Intent**
Build the primary Streamlit page — a conversational chat interface where users can type
queries, upload an image, or use the voice stub, and see the agent's responses in a
chat-bubble layout with product cards.

**Expected Outcomes**
- `app.py` launches a multi-page Streamlit app (2 pages: Chat, Dashboard)
- Chat page: text input box, image uploader widget, "Simulate Voice" button
- Submitted queries route to `agent/core.py run_agent()`
- Agent responses rendered as chat bubbles (using `st.chat_message`)
- Product results from tools rendered as expandable cards (name, price, rating, source,
  sustainability badge)
- Session state persists conversation history across turns

**Todo List**
1. Set up `app.py` with `st.set_page_config` and `st.sidebar` navigation (Chat / Dashboard)
2. Implement chat history using `st.session_state["messages"]`
3. Add `st.chat_input` for text queries and `st.file_uploader` for image input
4. Add a "🎙️ Simulate Voice" button that calls `voice_stub.transcribe_stub()` with a
   preset demo query and feeds it into the agent
5. On query submission, call `run_agent(query)` and append the response to chat history
6. Render product cards using `st.expander` or `st.columns` showing all key fields
7. Handle image input: pass the uploaded PIL image to `image_search` tool before
   calling the agent with an enriched query

**Relevant Context**
- `st.chat_message` (Streamlit ≥ 1.23) for chat bubbles
- `st.session_state` for conversation persistence
- Agent entry point: `agent/core.py run_agent()`

**Status** `[x] done`

---

### Sub-Task 7 — Streamlit Dashboard: Smart Shopping Pulse

**Intent**
Build the second Streamlit page — the visual "Smart Shopping Pulse" dashboard that gives
users a data-driven overview: price comparisons, sustainability rankings, trending products,
and their personalized shopping list.

**Expected Outcomes**
- Dashboard page with 4 panels using Plotly charts:
  1. **Price Comparison Chart** — bar chart of the same product across sources
  2. **Sustainability Leaderboard** — horizontal bar chart of top products by sustainability score
  3. **Trending Products** — simulated trending list (sorted by review_count desc)
  4. **Personalized Shopping List** — items the user has flagged during the chat session
- All charts rendered via Plotly inside `dashboard/visualizations.py`
- Shopping list stored in `st.session_state["shopping_list"]` and editable (add/remove)

**Todo List**
1. In `dashboard/visualizations.py`, implement:
   - `plot_price_comparison(product_name) -> plotly.Figure`
   - `plot_sustainability_leaderboard(top_n=10) -> plotly.Figure`
   - `plot_trending_products(top_n=10) -> plotly.Figure`
2. In `app.py` Dashboard page, render all 4 panels using `st.plotly_chart`
3. Add a product search box on the dashboard to select which product to compare prices for
4. Implement shopping list UI: `st.multiselect` or checkbox list from session state
5. Add an "Add to Shopping List" button in the chat product cards (Sub-Task 6) that appends
   to `st.session_state["shopping_list"]`

**Relevant Context**
- `plotly.express` for quick chart construction
- `data/products.json` loaded once via `@st.cache_data`
- `dashboard/visualizations.py` helpers called from `app.py`

**Status** `[x] done`

---

### Sub-Task 8 — Integration, Polish & README

**Intent**
Wire all components together, perform end-to-end integration testing with the mock data,
and produce a clear README so the project is fully runnable and presentable.

**Expected Outcomes**
- The full app runs with `streamlit run app.py` without errors
- At least 3 end-to-end demo flows work:
  1. Text query: "Find me a sustainable laptop under $1000"
  2. Image upload: user uploads any product image, gets similar product recommendations
  3. Dashboard: price comparison and sustainability leaderboard render correctly
- `README.md` with: project overview, architecture diagram (ASCII), setup instructions,
  how to add watsonx.ai credentials, how to run, and a feature checklist

**Todo List**
1. Run the full app and fix any import errors, missing dependencies, or broken tool calls
2. Verify the LangChain agent correctly routes to each tool for different query types
3. Ensure all Streamlit pages load without exception on startup
4. Write `README.md` covering: description, architecture, setup steps, usage, and demo notes
5. Add inline comments to `agent/core.py` and key tool files explaining the agent flow
6. Final check: confirm `.env.example` matches all vars used in `config.py`

**Relevant Context**
- Entry point: `streamlit run app.py`
- All credentials via `.env` file (never hardcoded)
- Mock data makes the app fully self-contained for demo

**Status** `[x] done`

---

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| LLM Backend | IBM watsonx.ai `ibm/granite-13b-chat-v2` | Mandatory IBM models requirement |
| Agent Framework | LangChain ReAct (ZERO_SHOT_REACT_DESCRIPTION) | Mandatory LangChain/orchestration requirement |
| Frontend | Streamlit | Rapid full-stack Python UI, multimodal widget support |
| Product Data | Mock JSON dataset | No external API keys needed, fully self-contained demo |
| Image Search | sentence-transformers CLIP + FAISS | Runs locally, no external API, functional demo quality |
| Voice Input | Stub/simulated | Functional demo scope; real Whisper can be swapped in later |
| Review Analysis | LLM-powered (Granite) | Showcases IBM model capability beyond search |
| Pricing Logic | Rule-based | Deterministic, explainable, no LLM cost for simple math |
