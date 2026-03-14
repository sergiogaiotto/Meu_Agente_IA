from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
from app.routers.agents import router as agents_router

app = FastAPI(
    title="AI Agents Lab - Fala Gaiotto",
    description="Curso Prático de Agentes de IA com LangGraph",
    version="1.0.0",
)

app.include_router(agents_router)

BASE_DIR = Path(__file__).resolve().parent.parent


@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = BASE_DIR / "templates" / "default.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
