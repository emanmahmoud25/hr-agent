"""All FastAPI endpoints."""
import time
from fastapi import APIRouter, HTTPException, UploadFile, File, Query, Request
from fastapi.responses import JSONResponse

from src.data.extractor import extract_text
from src.api.models import SingleResult, FullResult, StatusResponse

router = APIRouter()


def get_agent(request: Request):
    return request.app.state.agent


# ── Single task upload ────────────────────────────────────────
@router.post("/cv/upload", response_model=SingleResult)
async def upload_cv(
    request: Request,
    file: UploadFile = File(...),
    task: str = Query("classify", enum=["classify","skills","interview","improve"]),
):
    agent   = get_agent(request)
    content = await file.read()
    try:
        cv_text = extract_text(file.filename, content)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if len(cv_text) < 20:
        raise HTTPException(422, "Could not extract enough text from the file")

    t0     = time.time()
    result = agent.run(task, cv_text)
    return SingleResult(
        filename   = file.filename,
        task       = task,
        char_count = len(cv_text),
        result     = result,
        elapsed_s  = round(time.time() - t0, 2),
    )


# ── Full pipeline upload ──────────────────────────────────────
@router.post("/cv/upload/full", response_model=FullResult)
async def upload_full(request: Request, file: UploadFile = File(...)):
    agent   = get_agent(request)
    content = await file.read()
    try:
        cv_text = extract_text(file.filename, content)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if len(cv_text) < 20:
        raise HTTPException(422, "Could not extract enough text from the file")

    t0      = time.time()
    results = agent.process_cv_full(cv_text)
    return FullResult(
        filename   = file.filename,
        char_count = len(cv_text),
        elapsed_s  = round(time.time() - t0, 2),
        **results,
    )


# ── Status ────────────────────────────────────────────────────
@router.get("/status", response_model=StatusResponse)
def status(request: Request):
    return request.app.state.agent.status


# ── Memory info ───────────────────────────────────────────────
@router.get("/memory")
def memory_info():
    import torch
    if torch.cuda.is_available():
        used  = torch.cuda.memory_allocated(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return {"gpu_used_gb": round(used,2), "gpu_total_gb": round(total,2),
                "gpu_name": torch.cuda.get_device_name(0)}
    return {"gpu": "not available"}


# ── Unload ────────────────────────────────────────────────────
@router.delete("/loras/unload")
def unload(request: Request):
    request.app.state.agent.unload_all()
    return {"status": "unloaded"}
