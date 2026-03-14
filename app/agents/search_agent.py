"""
Módulo 3 - Modern Tool: DuckDuckGo Search
Agente com acesso a busca web via DuckDuckGo.
Demonstra integração de ferramentas externas no LangGraph.
"""

from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from app.agents import get_llm


# ── Tools ─────────────────────────────────────────────────────────
ddg_search = DuckDuckGoSearchResults(max_results=5)


@tool
def web_search(query: str) -> str:
    """Busca na web usando DuckDuckGo. Use para informações atualizadas."""
    try:
        results = ddg_search.invoke(query)
        return str(results)
    except Exception as e:
        return f"Erro na busca: {e}"


@tool
def summarize_search(query: str) -> str:
    """Busca na web e retorna um resumo estruturado dos resultados."""
    try:
        results = ddg_search.invoke(query)
        return f"Resultados para '{query}':\n{results}\n\nResuma estes resultados de forma clara e objetiva."
    except Exception as e:
        return f"Erro na busca: {e}"


TOOLS = [web_search, summarize_search]


# ── State ─────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ── Graph ─────────────────────────────────────────────────────────
def build_search_graph():
    llm = get_llm().bind_tools(TOOLS)

    def agent_node(state: AgentState):
        system = SystemMessage(content=(
            "Você é um assistente de pesquisa com acesso à web via DuckDuckGo. "
            "Use a ferramenta web_search para buscar informações atualizadas. "
            "Sempre cite as fontes encontradas. "
            "Responda em português brasileiro de forma clara e objetiva."
        ))
        messages = [system] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


async def run_search(user_input: str, history: list[dict] | None = None) -> str:
    graph = build_search_graph()
    messages = []
    if history:
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=user_input))
    result = await graph.ainvoke({"messages": messages})
    return result["messages"][-1].content
