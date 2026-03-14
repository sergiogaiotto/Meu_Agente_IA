"""
Módulo 6 - Agentic RAG
Retrieval-Augmented Generation com:
  - Upload de documentos (.txt, .pdf, .md)
  - Chunking configurável (chunk_size, chunk_overlap)
  - Embedding via OpenAI
  - Vector store via FAISS
  - Top-K retrieval configurável
  - Agente que decide quando buscar no KB vs responder direto
"""

import os
import json
from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from app.agents import get_llm
from app.config import settings


# ── Knowledge Base Manager ────────────────────────────────────────
class KnowledgeBaseManager:
    def __init__(self):
        self.vectorstores: dict[str, FAISS] = {}
        self.kb_metadata: dict[str, dict] = {}
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )

    def add_document(
        self,
        kb_name: str,
        content: str,
        filename: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> dict:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        docs = splitter.create_documents(
            [content],
            metadatas=[{"source": filename, "kb_name": kb_name}],
        )
        if kb_name in self.vectorstores:
            self.vectorstores[kb_name].add_documents(docs)
            self.kb_metadata[kb_name]["num_chunks"] += len(docs)
            self.kb_metadata[kb_name]["files"].append(filename)
        else:
            self.vectorstores[kb_name] = FAISS.from_documents(docs, self.embeddings)
            self.kb_metadata[kb_name] = {
                "name": kb_name,
                "num_chunks": len(docs),
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "files": [filename],
            }
        return {
            "kb_name": kb_name,
            "chunks_created": len(docs),
            "total_chunks": self.kb_metadata[kb_name]["num_chunks"],
        }

    def search(self, kb_name: str, query: str, top_k: int = 4) -> list[Document]:
        if kb_name not in self.vectorstores:
            return []
        return self.vectorstores[kb_name].similarity_search(query, k=top_k)

    def list_kbs(self) -> list[dict]:
        return list(self.kb_metadata.values())

    def delete_kb(self, kb_name: str) -> bool:
        if kb_name in self.vectorstores:
            del self.vectorstores[kb_name]
            del self.kb_metadata[kb_name]
            return True
        return False


# Singleton
kb_manager = KnowledgeBaseManager()


# ── RAG Agent ─────────────────────────────────────────────────────
class RAGState(TypedDict):
    messages: Annotated[list, add_messages]
    kb_name: str
    top_k: int


def build_rag_graph(kb_name: str, top_k: int = 4):
    llm = get_llm(temperature=0.1)

    @tool
    def retrieve_from_kb(query: str) -> str:
        """Busca informações relevantes na base de conhecimento."""
        docs = kb_manager.search(kb_name, query, top_k=top_k)
        if not docs:
            return "Nenhum documento relevante encontrado na base de conhecimento."
        context = "\n\n---\n\n".join(
            f"[Fonte: {d.metadata.get('source', 'desconhecida')}]\n{d.page_content}"
            for d in docs
        )
        return context

    tools = [retrieve_from_kb]
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: RAGState):
        system = SystemMessage(content=(
            "Você é um assistente RAG (Retrieval-Augmented Generation). "
            "Você tem acesso a uma base de conhecimento. "
            "SEMPRE use a ferramenta retrieve_from_kb para buscar informações antes de responder. "
            "Baseie suas respostas nos documentos recuperados. "
            "Se a informação não estiver na base, informe ao usuário. "
            "Cite as fontes dos documentos quando possível."
        ))
        messages = [system] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: RAGState):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(RAGState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


async def run_rag(
    user_input: str,
    kb_name: str,
    top_k: int = 4,
    history: list[dict] | None = None,
) -> str:
    if kb_name not in kb_manager.vectorstores:
        return f"Base de conhecimento '{kb_name}' não encontrada. Faça upload de documentos primeiro."

    graph = build_rag_graph(kb_name, top_k)
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
        "kb_name": kb_name,
        "top_k": top_k,
    })
    return result["messages"][-1].content
