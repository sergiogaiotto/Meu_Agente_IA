"""
Módulo 5 - Multi-Agent Workflow
Orquestra múltiplos agentes especializados:
  - Researcher: busca informações
  - Analyst: analisa dados
  - Writer: redige a resposta final
Padrão: Supervisor que roteia para agentes especializados.
"""

from typing import Annotated, TypedDict, Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from app.agents import get_llm


# ── State ─────────────────────────────────────────────────────────
class MultiAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    research: str
    analysis: str
    next_agent: str


# ── Graph ─────────────────────────────────────────────────────────
def build_multi_agent_graph():
    llm = get_llm(temperature=0.2)

    def supervisor_node(state: MultiAgentState):
        system = SystemMessage(content=(
            "Você é o Supervisor de uma equipe de agentes. "
            "Analise a solicitação e determine a sequência: "
            "1) researcher → 2) analyst → 3) writer → FINISH. "
            "Responda APENAS com o próximo agente: 'researcher', 'analyst', 'writer' ou 'FINISH'."
        ))
        context = f"\nPesquisa: {state.get('research', 'pendente')}\nAnálise: {state.get('analysis', 'pendente')}"
        msg = HumanMessage(content=f"Tarefa: {state['messages'][-1].content}{context}")
        response = llm.invoke([system, msg])
        next_agent = response.content.strip().lower()
        if "researcher" in next_agent:
            return {"next_agent": "researcher"}
        elif "analyst" in next_agent:
            return {"next_agent": "analyst"}
        elif "writer" in next_agent:
            return {"next_agent": "writer"}
        return {"next_agent": "FINISH"}

    def researcher_node(state: MultiAgentState):
        system = SystemMessage(content=(
            "Você é o Agente Pesquisador. Sua função é levantar informações relevantes "
            "sobre o tema solicitado. Seja detalhado e estruturado na pesquisa."
        ))
        user_msg = state["messages"][-1] if state["messages"] else HumanMessage(content="")
        response = llm.invoke([system, user_msg])
        return {"research": response.content, "next_agent": "supervisor"}

    def analyst_node(state: MultiAgentState):
        system = SystemMessage(content=(
            "Você é o Agente Analista. Com base na pesquisa abaixo, "
            "faça uma análise crítica, identifique padrões e insights.\n\n"
            f"Pesquisa:\n{state.get('research', '')}"
        ))
        user_msg = state["messages"][-1] if state["messages"] else HumanMessage(content="")
        response = llm.invoke([system, user_msg])
        return {"analysis": response.content, "next_agent": "supervisor"}

    def writer_node(state: MultiAgentState):
        system = SystemMessage(content=(
            "Você é o Agente Redator. Com base na pesquisa e análise abaixo, "
            "redija uma resposta final clara, objetiva e bem estruturada.\n\n"
            f"Pesquisa:\n{state.get('research', '')}\n\n"
            f"Análise:\n{state.get('analysis', '')}"
        ))
        user_msg = state["messages"][-1] if state["messages"] else HumanMessage(content="")
        response = llm.invoke([system, user_msg])
        return {"messages": [AIMessage(content=response.content)], "next_agent": "FINISH"}

    def route_agent(state: MultiAgentState) -> str:
        next_a = state.get("next_agent", "FINISH")
        if next_a == "researcher":
            return "researcher"
        elif next_a == "analyst":
            return "analyst"
        elif next_a == "writer":
            return "writer"
        return "FINISH"

    graph = StateGraph(MultiAgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("writer", writer_node)

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges("supervisor", route_agent, {
        "researcher": "researcher",
        "analyst": "analyst",
        "writer": "writer",
        "FINISH": END,
    })
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("analyst", "supervisor")
    graph.add_edge("writer", END)

    return graph.compile()


async def run_multi_agent(user_input: str, history: list[dict] | None = None) -> str:
    graph = build_multi_agent_graph()
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
        "research": "",
        "analysis": "",
        "next_agent": "",
    })
    return result["messages"][-1].content
