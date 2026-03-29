"""
HR Agent — Entry Point
Run: python main.py
UI:  http://localhost:8000
API: http://localhost:8000/docs
"""
import uvicorn
from src.api.app import app  # noqa: F401

if __name__ == "__main__":
    uvicorn.run(
        "src.api.app:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = False,
        workers = 1,          # keep 1 — model is not thread-safe
    )
