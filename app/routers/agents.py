from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.agents.react_agent import run_react
from app.agents.codeact_agent import run_codeact
from app.agents.search_agent import run_search
from app.agents.reflection_agent import run_reflection
from app.agents.multi_agent import run_multi_agent
from app.agents.rag_agent import run_rag, kb_manager

router = APIRouter(prefix="/api", tags=["agents"])


class ChatRequest(BaseModel):
    message: str
    history: list[dict] | None = None


class RAGChatRequest(BaseModel):
    message: str
    kb_name: str
    top_k: int = 4
    history: list[dict] | None = None


class ReflectionRequest(BaseModel):
    message: str
    max_iterations: int = 2
    history: list[dict] | None = None


# ── Health ────────────────────────────────────────────────────────
@router.get("/health")
async def health():
    return {"status": "ok"}


# ── 1. ReAct Agent ────────────────────────────────────────────────
@router.post("/react")
async def react_endpoint(req: ChatRequest):
    try:
        response = await run_react(req.message, req.history)
        return {"response": response, "agent": "react"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 2. CodeAct Agent ─────────────────────────────────────────────
@router.post("/codeact")
async def codeact_endpoint(req: ChatRequest):
    try:
        response = await run_codeact(req.message, req.history)
        return {"response": response, "agent": "codeact"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 3. DuckDuckGo Search Agent ───────────────────────────────────
@router.post("/search")
async def search_endpoint(req: ChatRequest):
    try:
        response = await run_search(req.message, req.history)
        return {"response": response, "agent": "search"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 4. Self-Reflection Agent ─────────────────────────────────────
@router.post("/reflection")
async def reflection_endpoint(req: ReflectionRequest):
    try:
        response = await run_reflection(req.message, req.max_iterations, req.history)
        return {"response": response, "agent": "reflection"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 5. Multi-Agent Workflow ──────────────────────────────────────
@router.post("/multi-agent")
async def multi_agent_endpoint(req: ChatRequest):
    try:
        response = await run_multi_agent(req.message, req.history)
        return {"response": response, "agent": "multi-agent"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 6. Agentic RAG ──────────────────────────────────────────────
@router.post("/rag/upload")
async def rag_upload(
    file: UploadFile = File(...),
    kb_name: str = Form("default"),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(50),
):
    try:
        content = await file.read()
        text = content.decode("utf-8", errors="ignore")
        result = kb_manager.add_document(
            kb_name=kb_name,
            content=text,
            filename=file.filename,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return {"status": "success", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag/knowledge-bases")
async def list_knowledge_bases():
    return {"knowledge_bases": kb_manager.list_kbs()}


@router.delete("/rag/knowledge-bases/{kb_name}")
async def delete_knowledge_base(kb_name: str):
    if kb_manager.delete_kb(kb_name):
        return {"status": "deleted", "kb_name": kb_name}
    raise HTTPException(status_code=404, detail=f"KB '{kb_name}' não encontrada")


@router.post("/rag/chat")
async def rag_chat(req: RAGChatRequest):
    try:
        response = await run_rag(req.message, req.kb_name, req.top_k, req.history)
        return {"response": response, "agent": "rag"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
