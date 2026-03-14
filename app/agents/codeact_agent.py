"""
Módulo 2 - CodeAct Agent
O agente gera e executa código Python para resolver tarefas.
Em vez de usar ferramentas pré-definidas, ele ESCREVE código.
"""

from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from app.agents import get_llm
import io
import contextlib


# ── Tools ─────────────────────────────────────────────────────────
@tool
def execute_python(code: str) -> str:
    """Executa código Python e retorna o output. Use print() para exibir resultados."""
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    safe_globals = {
        "__builtins__": {
            "print": print, "range": range, "len": len, "str": str,
            "int": int, "float": float, "list": list, "dict": dict,
            "tuple": tuple, "set": set, "sorted": sorted, "enumerate": enumerate,
            "zip": zip, "map": map, "filter": filter, "sum": sum,
            "min": min, "max": max, "abs": abs, "round": round,
            "isinstance": isinstance, "type": type, "bool": bool,
            "True": True, "False": False, "None": None,
        }
    }
    try:
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exec(code, safe_globals)
        output = stdout_capture.getvalue()
        errors = stderr_capture.getvalue()
        result = ""
        if output:
            result += f"Output:\n{output}"
        if errors:
            result += f"Errors:\n{errors}"
        return result if result else "Código executado com sucesso (sem output)."
    except Exception as e:
        return f"Erro na execução: {type(e).__name__}: {e}"


TOOLS = [execute_python]


# ── State ─────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ── Graph ─────────────────────────────────────────────────────────
def build_codeact_graph():
    llm = get_llm(temperature=0.0).bind_tools(TOOLS)

    def agent_node(state: AgentState):
        system = SystemMessage(content=(
            "Você é um agente CodeAct. Para resolver tarefas, você ESCREVE e EXECUTA código Python. "
            "Use a ferramenta execute_python para rodar seu código. "
            "Sempre use print() para exibir resultados. "
            "Raciocine sobre o problema, escreva o código, execute, analise o resultado."
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


async def run_codeact(user_input: str, history: list[dict] | None = None) -> str:
    graph = build_codeact_graph()
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
