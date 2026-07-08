"""
Agent Core — Intelligent Shopping Agent
Uses IBM watsonx.ai Granite model via ChatWatsonx + LangGraph ReAct agent.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_ibm import ChatWatsonx
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

import config

# ---------------------------------------------------------------------------
# LLM initialisation
# ---------------------------------------------------------------------------

def _build_llm() -> ChatWatsonx:
    """Instantiate the IBM Granite chat model via watsonx.ai."""
    return ChatWatsonx(
        model_id="ibm/granite-13b-chat-v2",
        url=config.WATSONX_URL,
        api_key=config.WATSONX_API_KEY,
        project_id=config.WATSONX_PROJECT_ID,
        params={
            "max_new_tokens": 1024,
            "temperature": 0.3,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
        },
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are ShopBot, an intelligent shopping assistant powered by IBM Granite.
You help users find the best products, compare prices across stores, analyse reviews, predict deals, and recommend sustainable alternatives.

You have access to the following tools:
- product_search: Search for products by text description or keywords.
- price_comparator: Compare prices for a product across multiple e-commerce sources.
- deal_predictor: Predict if now is a good time to buy, detect fake reviews, and alert about price drops.
- review_analyzer: Summarise product reviews and detect overall sentiment.
- recommendation_engine: Recommend alternative products based on budget, category, and sustainability preference.

Always use the tools to ground your answers in real product data. Be concise, helpful, and honest.
When recommending products, mention the sustainability score when it is 7 or above.
Format product lists clearly using bullet points."""


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def build_agent(tools: list):
    """
    Build a LangGraph ReAct agent with memory.

    Parameters
    ----------
    tools : list
        List of LangChain-compatible tool objects.

    Returns
    -------
    graph : CompiledGraph
        The runnable LangGraph agent.
    """
    llm = _build_llm()
    memory = MemorySaver()
    graph = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=memory,
    )
    return graph


# ---------------------------------------------------------------------------
# Public run interface
# ---------------------------------------------------------------------------

def run_agent(agent_graph, query: str, thread_id: str = "default") -> str:
    """
    Run a user query through the agent and return the assistant's reply.

    Parameters
    ----------
    agent_graph : CompiledGraph
        The agent returned by build_agent().
    query : str
        The user's natural-language query.
    thread_id : str
        Conversation thread ID for memory persistence.

    Returns
    -------
    str
        The final text response from the agent.
    """
    config_dict = {"configurable": {"thread_id": thread_id}}
    result = agent_graph.invoke(
        {"messages": [HumanMessage(content=query)]},
        config=config_dict,
    )
    # Last message in the graph output is the AI reply
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg.content
    return "Sorry, I could not generate a response. Please try again."


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from agent.tools.product_search import get_product_search_tool
    from agent.tools.price_comparator import get_price_comparator_tool
    from agent.tools.deal_predictor import get_deal_predictor_tool
    from agent.tools.review_analyzer import get_review_analyzer_tool
    from agent.tools.recommendation_engine import get_recommendation_tool

    tools = [
        get_product_search_tool(),
        get_price_comparator_tool(),
        get_deal_predictor_tool(),
        get_review_analyzer_tool(),
        get_recommendation_tool(),
    ]

    agent = build_agent(tools)
    response = run_agent(agent, "Find me a good laptop under $1000")
    print("\n=== Agent Response ===")
    print(response)
