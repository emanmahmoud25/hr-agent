"""
HR Agent — Proxy App (Qwen on Colab)
VS Code forwards everything to Colab GPU via localtunnel.

Setup .env:
  API_BASE_URL=https://xxxxx.loca.lt
"""
import os
import httpx
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

COLAB_URL = os.getenv("API_BASE_URL", "").rstrip("/")
UI_PATH   = Path(__file__).parent / "ui" / "index.html"

print("─" * 50)
print("  HR Agent — QWEN / COLAB mode")
print(f"  Colab URL : {COLAB_URL or 'NOT SET'}")
print(f"  Local UI  : http://localhost:8000")
print("─" * 50)

if not COLAB_URL:
    print("  WARNING: API_BASE_URL not set in .env!")


def check_colab():
    if not COLAB_URL:
        raise HTTPException(503, "API_BASE_URL not set — add Colab URL to .env")


app = FastAPI(title="HR Agent — Qwen Proxy", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── include ranking router (يشتغل Groq fallback لو مفيش Colab) ──
from src.api.routes_rank import router as rank_router
app.include_router(rank_router)


@app.get("/", response_class=HTMLResponse)
def ui():
    if UI_PATH.exists():
        return UI_PATH.read_text(encoding="utf-8")
    return "<h2>HR Agent (Qwen/Colab) running. <a href='/docs'>Docs</a></h2>"


@app.post("/cv/upload")
async def upload_cv(
    file: UploadFile = File(...),
    task: str = Query("classify", enum=["classify","skills","interview","improve"]),
):
    check_colab()
    content = await file.read()
    async with httpx.AsyncClient(timeout=600) as client:
        try:
            resp = await client.post(
                f"{COLAB_URL}/cv/upload",
                params  = {"task": task},
                files   = {"file": (file.filename, content, file.content_type)},
                headers = {"bypass-tunnel-reminder": "true"},
            )
            return JSONResponse(content=resp.json(), status_code=resp.status_code)
        except httpx.TimeoutException:
            raise HTTPException(504, "Colab request timed out")
        except httpx.ConnectError:
            raise HTTPException(503, f"Cannot connect to Colab: {COLAB_URL}")


@app.post("/cv/upload/full")
async def upload_full(file: UploadFile = File(...)):
    check_colab()
    content = await file.read()
    async with httpx.AsyncClient(timeout=600) as client:
        try:
            resp = await client.post(
                f"{COLAB_URL}/cv/upload/full",
                files   = {"file": (file.filename, content, file.content_type)},
                headers = {"bypass-tunnel-reminder": "true"},
            )
            return JSONResponse(content=resp.json(), status_code=resp.status_code)
        except httpx.TimeoutException:
            raise HTTPException(504, "Colab request timed out")
        except httpx.ConnectError:
            raise HTTPException(503, f"Cannot connect to Colab: {COLAB_URL}")


@app.get("/status")
async def status():
    if not COLAB_URL:
        return {"mode": "proxy→qwen", "colab": "not configured"}
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(
                f"{COLAB_URL}/status",
                headers={"bypass-tunnel-reminder": "true"},
            )
            data = resp.json()
            data["mode"]      = "proxy→qwen"
            data["colab_url"] = COLAB_URL
            return data
        except Exception as e:
            return {"mode": "proxy→qwen", "colab": "unreachable", "error": str(e)}


@app.delete("/loras/unload")
async def unload():
    check_colab()
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.delete(
            f"{COLAB_URL}/loras/unload",
            headers={"bypass-tunnel-reminder": "true"},
        )
        return resp.json()