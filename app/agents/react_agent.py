"""
Módulo 1 - ReAct Agent
Reasoning + Acting: o agente raciocina sobre a tarefa e decide qual ação tomar.
Usa o ciclo: Thought -> Action -> Observation -> Thought ...
"""

from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from app.agents import get_llm
import math


# ── Tools ─────────────────────────────────────────────────────────
@tool
def calculator(expression: str) -> str:
    """Avalia uma expressão matemática. Ex: '2 + 2', 'math.sqrt(16)', 'math.pi * 3**2'"""
    try:
        allowed = {
            "math": math, "abs": abs, "round": round,
            "min": min, "max": max, "sum": sum,
            "int": int, "float": float,
        }
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Erro: {e}"


@tool
def string_tool(text: str) -> str:
    """Manipula texto: conta palavras, caracteres, inverte, converte para maiúsculas/minúsculas."""
    words = text.split()
    return (
        f"Texto: {text}\n"
        f"Palavras: {len(words)}\n"
        f"Caracteres: {len(text)}\n"
        f"Maiúsculas: {text.upper()}\n"
        f"Invertido: {text[::-1]}"
    )


TOOLS = [calculator, string_tool]


# ── State ─────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ── Graph ─────────────────────────────────────────────────────────
def build_react_graph():
    llm = get_llm().bind_tools(TOOLS)

    def agent_node(state: AgentState):
        system = SystemMessage(content=(
            "Você é um assistente ReAct. Raciocine passo a passo antes de agir. "
            "Use as ferramentas disponíveis quando necessário. "
            "Sempre explique seu raciocínio antes de usar uma ferramenta."
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


async def run_react(user_input: str, history: list[dict] | None = None) -> str:
    graph = build_react_graph()
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
