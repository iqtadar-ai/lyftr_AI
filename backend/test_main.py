from types import SimpleNamespace

import litellm
import pytest
from fastapi.testclient import TestClient

import database
import llm_gateway
from main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    test_db = tmp_path / "test_app.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{test_db}")
    monkeypatch.setenv("LLM_API_KEYS", "gsk_test_1,gsk_test_2,AIza_test_3")
    monkeypatch.setenv("ENABLE_FAILURE_SIMULATION", "true")

    llm_gateway._RESPONSE_CACHE.clear()

    def fake_completion(model, messages, api_key, caching=False):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=f"Reply from {model}")
                )
            ],
            usage=SimpleNamespace(prompt_tokens=20, completion_tokens=10),
        )

    monkeypatch.setattr(litellm, "completion", fake_completion)

    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_rejects_blank_input(client):
    res = client.post("/chat", json={"session_id": "s1", "message": "   "})
    assert res.status_code == 422


def test_chat_persistence_and_sidebar_sessions(client):
    res = client.post(
        "/chat", json={"session_id": "sess-1", "message": "Hello SQLite"}
    )
    assert res.status_code == 200
    body = res.json()
    assert body["reply"] == "Reply from groq/openai/gpt-oss-20b"
    assert body["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert body["metadata"]["key_id"] == "key_1"

    # Verify history endpoint
    hist = client.get("/sessions/sess-1/messages").json()
    assert len(hist["messages"]) == 2
    assert hist["messages"][0]["role"] == "user"
    assert hist["messages"][1]["role"] == "assistant"

    # Verify sidebar session listing and deletion
    sessions = client.get("/sessions").json()["sessions"]
    assert any(s["session_id"] == "sess-1" for s in sessions)

    del_res = client.delete("/sessions/sess-1")
    assert del_res.status_code == 200
    assert client.get("/sessions/sess-1/messages").json()["messages"] == []


def test_response_cache_hit(client):
    first = client.post(
        "/chat", json={"session_id": "cache-1", "message": "Same prompt"}
    ).json()
    assert first["metadata"]["resolution_source"] == "LIVE_INFERENCE"

    second = client.post(
        "/chat", json={"session_id": "cache-2", "message": "Same prompt"}
    ).json()
    assert second["metadata"]["resolution_source"] == "LLM_CACHE"
    assert second["metadata"]["key_id"] is None
    assert second["metadata"]["fallback_attempts"] == 0


def test_simulated_cross_provider_failover(client):
    res = client.post(
        "/chat",
        headers={"X-Simulate-Failure": "key_1=429,key_2=500"},
        json={"session_id": "failover-1", "message": "Test failover"},
    )
    assert res.status_code == 200
    meta = res.json()["metadata"]
    assert meta["key_id"] == "key_3"
    assert meta["model"] == "gemini/gemini-3.8-flash"
    assert meta["fallback_attempts"] == 2


def test_all_providers_down_does_not_corrupt_history(client):
    res = client.post(
        "/chat",
        headers={"X-Simulate-Failure": "*=429"},
        json={"session_id": "down-1", "message": "Should not save"},
    )
    assert res.status_code == 503
    assert res.json()["detail"] == "all providers unavailable"

    # Confirm nothing was written to SQLite on failure
    hist = client.get("/sessions/down-1/messages").json()
    assert hist["messages"] == []