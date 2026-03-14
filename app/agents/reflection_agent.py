"""
Módulo 4 - Self-Reflection Agent
O agente gera uma resposta, depois a critica e refina iterativamente.
Implementa o padrão Reflexion: Generate -> Reflect -> Refine.
"""

from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from app.agents import get_llm


# ── State ─────────────────────────────────────────────────────────
class ReflectionState(TypedDict):
    messages: Annotated[list, add_messages]
    draft: str
    critique: str
    iteration: int
    max_iterations: int


# ── Graph ─────────────────────────────────────────────────────────
def build_reflection_graph():
    llm = get_llm(temperature=0.3)

    def generate_node(state: ReflectionState):
        if state["iteration"] == 0:
            system = SystemMessage(content=(
                "Você é um assistente especialista. Gere uma resposta completa e detalhada "
                "para a pergunta do usuário. Seja preciso e abrangente."
            ))
            messages = [system] + state["messages"]
        else:
            system = SystemMessage(content=(
                "Você é um assistente especialista. Revise sua resposta anterior com base na crítica recebida. "
                "Melhore os pontos fracos identificados mantendo os pontos fortes.\n\n"
                f"Resposta anterior:\n{state['draft']}\n\n"
                f"Crítica:\n{state['critique']}\n\n"
                "Gere uma versão melhorada da resposta."
            ))
            messages = [system] + state["messages"]

        response = llm.invoke(messages)
        return {
            "draft": response.content,
            "iteration": state["iteration"] + 1,
        }

    def reflect_node(state: ReflectionState):
        system = SystemMessage(content=(
            "Você é um crítico rigoroso. Analise a resposta abaixo e identifique:\n"
            "1. Pontos fortes\n"
            "2. Pontos fracos ou imprecisões\n"
            "3. Informações faltantes\n"
            "4. Sugestões de melhoria\n"
            "5. Nota de 1-10 para qualidade geral\n\n"
            f"Resposta a avaliar:\n{state['draft']}"
        ))
        response = llm.invoke([system])
        return {"critique": response.content}

    def should_continue(state: ReflectionState):
        if state["iteration"] >= state["max_iterations"]:
            return "finalize"
        return "reflect"

    def finalize_node(state: ReflectionState):
        final = AIMessage(content=(
            f"{state['draft']}\n\n"
            f"---\n*Reflexão (iteração {state['iteration']}):*\n{state['critique']}"
        ))
        return {"messages": [final]}

    graph = StateGraph(ReflectionState)
    graph.add_node("generate", generate_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("generate")
    graph.add_conditional_edges("generate", should_continue, {
        "reflect": "reflect",
        "finalize": "finalize",
    })
    graph.add_edge("reflect", "generate")
    graph.add_edge("finalize", END)

    return graph.compile()


async def run_reflection(user_input: str, max_iterations: int = 2, history: list[dict] | None = None) -> str:
    graph = build_reflection_graph()
    messages = []
    if history:
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=user_input))
    result = await graph.ainvoke({
        "messages": messages,
        "draft": "",
        "critique": "",
        "iteration": 0,
        "max_iterations": max_iterations,
    })
    return result["messages"][-1].content
