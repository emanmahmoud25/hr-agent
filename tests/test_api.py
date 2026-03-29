"""FastAPI endpoint tests using TestClient."""
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from src.api.app import app


@pytest.fixture
def client():
    mock_agent         = MagicMock()
    mock_agent.status  = {
        "device":"cpu","active_lora":None,
        "switch_count":0,"call_count":0,"adapters":{}
    }
    mock_agent.run.return_value            = "INFORMATION-TECHNOLOGY"
    mock_agent.process_cv_full.return_value= {
        "classify":"INFORMATION-TECHNOLOGY",
        "skills":"Python, ML",
        "interview":"Q1...",
        "improve":"Add more details",
    }
    app.state.agent = mock_agent
    return TestClient(app)


def test_status(client):
    r = client.get("/status")
    assert r.status_code == 200
    assert "device" in r.json()


def test_upload_txt(client):
    r = client.post(
        "/cv/upload?task=classify",
        files={"file": ("cv.txt", b"Data Scientist with 5 years experience in Python and ML", "text/plain")},
    )
    assert r.status_code == 200
    assert r.json()["task"] == "classify"


def test_upload_empty_fails(client):
    r = client.post(
        "/cv/upload?task=classify",
        files={"file": ("cv.txt", b"short", "text/plain")},
    )
    assert r.status_code == 422
