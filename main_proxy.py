"""
HR Agent — Qwen/Colab Entry Point

Setup .env:
  API_BASE_URL=https://xxxxx.loca.lt

Run:
  python main_proxy.py
  Open: http://localhost:8000
"""
import uvicorn
from src.api.app_proxy import app  # noqa: F401

if __name__ == "__main__":
    uvicorn.run(
        "src.api.app_proxy:app",
        host   = "0.0.0.0",
        port   = 8000,
        reload = False,
    )