"""
HR Agent — FastAPI App (Groq mode)
No GPU needed — all inference via Groq API.
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("─" * 50)
    print("  HR Agent — GROQ mode")
    print("─" * 50)
    from src.agent.dynamic_lora import DynamicLoRAAgent
    app.state.agent = DynamicLoRAAgent(base_model=None, tokenizer=None)
    print("  Agent ready (Groq)")
    yield
    print("  Shutdown complete")


app = FastAPI(
    title    = "HR Agent — Groq API",
    version  = "1.0.0",
    lifespan = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

from src.api.routes      import router
from src.api.routes_rank import router as rank_router

app.include_router(router)
app.include_router(rank_router)

UI_PATH = Path(__file__).parent / "ui" / "index.html"

@app.get("/", response_class=HTMLResponse)
def ui():
    if UI_PATH.exists():
        return UI_PATH.read_text(encoding="utf-8")
    return "<h2>HR Agent (Groq) running. <a href='/docs'>Docs</a></h2>"