import os
import time
from typing import Any, Optional

import litellm
from litellm.caching.caching import Cache

litellm.drop_params = True


def configure_cache() -> bool:
    """Initializes LiteLLM's built-in in-memory cache unless disabled."""
    enabled = os.getenv("ENABLE_CACHE", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if enabled:
        if litellm.cache is None:
            litellm.cache = Cache()
    else:
        litellm.cache = None
    return enabled


configure_cache()


class AllProvidersUnavailableError(Exception):
    """Raised when every configured API key fails."""


def is_cache_enabled() -> bool:
    return litellm.cache is not None


def load_key_pool() -> list[tuple[str, str]]:
    """Parses LLM_API_KEYS into [(key_1, raw_key), (key_2, raw_key), ...]."""
    raw = os.getenv("LLM_API_KEYS", "")
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    return [(f"key_{idx + 1}", key) for idx, key in enumerate(keys)]


def resolve_model_for_key(api_key: str) -> str:
    """Maps key prefixes to their provider-specific LiteLLM model IDs."""
    if api_key.startswith("gsk_"):
        return os.getenv("GROQ_MODEL", "groq/openai/gpt-oss-20b").strip()
    if api_key.startswith(("AIza", "AQ.")):
        return os.getenv("GEMINI_MODEL", "gemini/gemini-3.8-flash").strip()
    return os.getenv("GROQ_MODEL", "groq/openai/gpt-oss-20b").strip()


def parse_failure_simulation(header_value: Optional[str]) -> dict[str, int]:
    """Parses 'key_1=429,key_2=500' or '*=429' when simulation is enabled."""
    if not header_value:
        return {}
    enabled = os.getenv("ENABLE_FAILURE_SIMULATION", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not enabled:
        return {}

    rules: dict[str, int] = {}
    for item in header_value.split(","):
        part = item.strip()
        if "=" not in part:
            continue
        target, code_str = part.split("=", 1)
        target = target.strip()
        try:
            rules[target] = int(code_str.strip())
        except ValueError:
            continue
    return rules


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_response_payload(
    response: Any,
    model: str,
    resolution_source: str,
    key_id: Optional[str],
    latency_ms: int,
    fallback_attempts: int,
) -> tuple[str, dict[str, Any]]:
    content = ""
    choices = _get_field(response, "choices") or []
    if choices:
        msg = _get_field(choices[0], "message")
        content = (_get_field(msg, "content") or "").strip()

    usage = _get_field(response, "usage")
    input_tokens = int(_get_field(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(_get_field(usage, "completion_tokens", 0) or 0)

    metadata: dict[str, Any] = {
        "resolution_source": resolution_source,
        "model": model,
        "key_id": key_id,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "fallback_attempts": fallback_attempts,
    }
    return content, metadata


def _lookup_pool_cache(
    models: list[str], messages: list[dict[str, str]]
) -> Optional[tuple[str, Any]]:
    """
    Checks LiteLLM's built-in cache across all unique configured models
    so replies previously stored under a fallback model still hit.
    """
    if litellm.cache is None:
        return None
    for model in models:
        try:
            cache_key = litellm.cache.get_cache_key(model=model, messages=messages)
            if not cache_key:
                continue
            cached = litellm.cache.get_cache(cache_key)
            if cached is not None:
                return model, cached
        except Exception:
            continue
    return None


def generate_reply(
    messages: list[dict[str, str]],
    simulate_failure_header: Optional[str] = None,
) -> tuple[str, dict[str, Any]]:
    """
    Executes chat completion through LiteLLM with built-in caching
    and sequential API key + cross-provider failover.
    """
    pool = load_key_pool()
    if not pool:
        raise AllProvidersUnavailableError("No LLM API keys configured.")

    sim_rules = parse_failure_simulation(simulate_failure_header)
    is_simulating = bool(sim_rules)

    unique_models: list[str] = []
    for _, raw_key in pool:
        m = resolve_model_for_key(raw_key)
        if m not in unique_models:
            unique_models.append(m)

    # Check LiteLLM's cache across configured models (including fallback models)
    if not is_simulating and litellm.cache is not None:
        start = time.perf_counter()
        cached_hit = _lookup_pool_cache(unique_models, messages)
        if cached_hit is not None:
            cached_model, cached_resp = cached_hit
            latency_ms = int(round((time.perf_counter() - start) * 1000))
            return _extract_response_payload(
                response=cached_resp,
                model=cached_model,
                resolution_source="LLM_CACHE",
                key_id=None,
                latency_ms=latency_ms,
                fallback_attempts=0,
            )

    fallback_attempts = 0
    for key_label, raw_key in pool:
        model = resolve_model_for_key(raw_key)
        start = time.perf_counter()

        # Simulated failure injection
        sim_status = sim_rules.get(key_label) or sim_rules.get("*")
        if sim_status is not None:
            fallback_attempts += 1
            continue

        try:
            call_kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "api_key": raw_key,
            }
            if is_simulating:
                call_kwargs["cache"] = {"no-cache": True, "no-store": True}

            response = litellm.completion(**call_kwargs)
            latency_ms = int(round((time.perf_counter() - start) * 1000))

            hidden = _get_field(response, "_hidden_params") or {}
            is_cache_hit = bool(_get_field(hidden, "cache_hit")) and not is_simulating

            return _extract_response_payload(
                response=response,
                model=model,
                resolution_source="LLM_CACHE" if is_cache_hit else "LIVE_INFERENCE",
                key_id=None if is_cache_hit else key_label,
                latency_ms=latency_ms,
                fallback_attempts=0 if is_cache_hit else fallback_attempts,
            )
        except Exception:
            fallback_attempts += 1
            continue

    raise AllProvidersUnavailableError("all providers unavailable")