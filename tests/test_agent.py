"""Basic agent tests (no GPU required — mocks the model)."""
import pytest
from unittest.mock import MagicMock, patch
from src.config import cfg


def make_mock_agent():
    from src.agent.dynamic_lora import DynamicLoRAAgent
    base   = MagicMock()
    tok    = MagicMock()
    agent  = DynamicLoRAAgent.__new__(DynamicLoRAAgent)
    agent.base_model    = base
    agent.tokenizer     = tok
    agent.current_lora  = None
    agent.active_model  = None
    agent._switch_count = 0
    agent._call_count   = 0
    return agent


def test_task_map_valid():
    for task in ["classify","skills","interview","improve"]:
        assert task in cfg.TASK_MAP


def test_valid_labels_not_empty():
    assert len(cfg.VALID_LABELS) == 24


def test_keyword_map_covers_all_labels():
    for label in cfg.VALID_LABELS:
        assert label in cfg.KEYWORD_MAP, f"{label} missing from KEYWORD_MAP"


def test_status_keys():
    agent = make_mock_agent()
    s = agent.status
    assert "device"       in s
    assert "active_lora"  in s
    assert "switch_count" in s
    assert "call_count"   in s
    assert "adapters"     in s


def test_unknown_task_raises():
    agent = make_mock_agent()
    with pytest.raises(ValueError):
        agent.run("unknown_task", "some cv text")
