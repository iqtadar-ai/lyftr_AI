import litellm
import pytest
from fastapi.testclient import TestClient
from litellm.caching.caching import Cache

from main import app

_real_completion = litellm.completion


def _mock_completion(**kwargs):
    return _real_completion(
        mock_response=f"Reply from {kwargs['model']}", **kwargs
    )


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test_app.db'}")
    monkeypatch.setenv("LLM_API_KEYS", "gsk_test_1,gsk_test_2,AIza_test_3")
    monkeypatch.setenv("ENABLE_FAILURE_SIMULATION", "true")
    monkeypatch.setenv("ENABLE_CACHE", "true")
    litellm.cache = Cache()
    monkeypatch.setattr(litellm, "completion", _mock_completion)
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {
        "status": "ok",
        "keys_configured": 3,
        "cache_enabled": True,
    }


def test_unknown_session_returns_404(client):
    res = client.get("/sessions/does-not-exist/messages")
    assert res.status_code == 404
    assert res.json() == {"detail": "session not found"}


def test_validation_limits_return_422(client):
    assert (
        client.post("/chat", json={"session_id": "s1", "message": "   "}).status_code
        == 422
    )
    assert (
        client.post(
            "/chat", json={"session_id": "s1", "message": "a" * 4001}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/chat", json={"session_id": "s" * 129, "message": "hello"}
        ).status_code
        == 422
    )


def test_chat_persistence_and_custom_system_prompt(client):
    res = client.post(
        "/chat",
        json={
            "session_id": "sess-custom",
            "message": "Hello SQLite",
            "system_prompt": "Respond like a pirate.",
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["reply"] == "Reply from groq/openai/gpt-oss-20b"
    assert body["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert body["metadata"]["key_id"] == "key_1"

    hist = client.get("/sessions/sess-custom/messages")
    assert hist.status_code == 200
    msgs = hist.json()["messages"]
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"


def test_same_provider_failover_key1_to_key2(client):
    res = client.post(
        "/chat",
        headers={"X-Simulate-Failure": "key_1=429"},
        json={"session_id": "failover-same", "message": "Test key_2"},
    )
    assert res.status_code == 200
    meta = res.json()["metadata"]
    assert meta["key_id"] == "key_2"
    assert meta["model"] == "groq/openai/gpt-oss-20b"
    assert meta["fallback_attempts"] == 1


def test_cross_provider_failover_to_gemini(client):
    res = client.post(
        "/chat",
        headers={"X-Simulate-Failure": "key_1=429,key_2=500"},
        json={"session_id": "failover-cross", "message": "Test key_3"},
    )
    assert res.status_code == 200
    meta = res.json()["metadata"]
    assert meta["key_id"] == "key_3"
    assert meta["model"] == "gemini/gemini-3.8-flash"
    assert meta["fallback_attempts"] == 2


def test_litellm_cache_hit_and_post_failover_lookup(client, monkeypatch):
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

    def groq_down_completion(**kwargs):
        if kwargs["model"].startswith("groq/"):
            raise RuntimeError("429 Rate Limit")
        return _mock_completion(**kwargs)

    monkeypatch.setattr(litellm, "completion", groq_down_completion)

    failover_live = client.post(
        "/chat", json={"session_id": "cache-fb-1", "message": "Fallback cached prompt"}
    ).json()
    assert failover_live["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert failover_live["metadata"]["model"] == "gemini/gemini-3.8-flash"
    assert failover_live["metadata"]["fallback_attempts"] == 2

    failover_cached = client.post(
        "/chat", json={"session_id": "cache-fb-2", "message": "Fallback cached prompt"}
    ).json()
    assert failover_cached["metadata"]["resolution_source"] == "LLM_CACHE"
    assert failover_cached["metadata"]["model"] == "gemini/gemini-3.8-flash"
    assert failover_cached["metadata"]["key_id"] is None
    assert failover_cached["metadata"]["fallback_attempts"] == 0


def test_all_providers_down_does_not_corrupt_history(client):
    res = client.post(
        "/chat",
        headers={"X-Simulate-Failure": "*=429"},
        json={"session_id": "down-1", "message": "Should not save"},
    )
    assert res.status_code == 503
    assert res.json()["detail"] == "all providers unavailable"

    hist = client.get("/sessions/down-1/messages")
    assert hist.status_code == 404
    assert hist.json() == {"detail": "session not found"}