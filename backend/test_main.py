from types import SimpleNamespace

import litellm
import pytest
from fastapi.testclient import TestClient
from litellm.caching.caching import Cache

import llm_gateway
from main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    test_db = tmp_path / "test_app.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{test_db}")
    monkeypatch.setenv("LLM_API_KEYS", "gsk_test_1,gsk_test_2,AIza_test_3")
    monkeypatch.setenv("ENABLE_FAILURE_SIMULATION", "true")
    monkeypatch.setenv("ENABLE_CACHE", "true")

    litellm.cache = Cache()

    def fake_completion(model, messages, api_key, **kwargs):
        cache_opts = kwargs.get("cache") or {}
        if not cache_opts.get("no-cache") and litellm.cache is not None:
            ck = litellm.cache.get_cache_key(model=model, messages=messages)
            cached = litellm.cache.get_cache(ck)
            if cached is not None:
                cached._hidden_params = {"cache_hit": True}
                return cached

        resp = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=f"Reply from {model}")
                )
            ],
            usage=SimpleNamespace(prompt_tokens=20, completion_tokens=10),
            _hidden_params={},
        )
        if not cache_opts.get("no-store") and litellm.cache is not None:
            ck = litellm.cache.get_cache_key(model=model, messages=messages)
            litellm.cache.set_cache(ck, resp)
        return resp

    monkeypatch.setattr(litellm, "completion", fake_completion)

    with TestClient(app) as test_client:
        yield test_client


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
    # Blank message
    assert (
        client.post("/chat", json={"session_id": "s1", "message": "   "}).status_code
        == 422
    )
    # Message exceeding 4000 characters
    assert (
        client.post(
            "/chat", json={"session_id": "s1", "message": "a" * 4001}
        ).status_code
        == 422
    )
    # Session ID exceeding 128 characters
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


def test_litellm_cache_hit_and_post_failover_lookup(client):
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


def test_all_providers_down_does_not_corrupt_history(client):
    res = client.post(
        "/chat",
        headers={"X-Simulate-Failure": "*=429"},
        json={"session_id": "down-1", "message": "Should not save"},
    )
    assert res.status_code == 503
    assert res.json()["detail"] == "all providers unavailable"

    # Unknown session still returns 404 because nothing was persisted
    hist = client.get("/sessions/down-1/messages")
    assert hist.status_code == 404
    assert hist.json() == {"detail": "session not found"}