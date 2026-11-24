import hashlib
import json
import os
import time
from typing import Any

import litellm
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)

# Turn off noisy LiteLLM debug logs in the terminal
litellm.suppress_debug_info = True

# In-memory response cache keyed by SHA-256 hash of (model, messages)
_RESPONSE_CACHE: dict[str, dict[str, Any]] = {}
CACHE_ENABLED = True

VALID_SIMULATED_STATUSES = {429, 500, 502, 503}


class AllProvidersUnavailableError(Exception):
    """Raised when every configured API key fails."""


def resolve_model_for_key(api_key: str) -> str:
    """Picks the right LiteLLM model based on the API key's provider prefix."""
    if api_key.startswith("gsk_"):
        return os.getenv("GROQ_MODEL", "groq/openai/gpt-oss-20b").strip()
    if api_key.startswith(("AIza", "AQ.")):
        return os.getenv("GEMINI_MODEL", "gemini/gemini-3.8-flash").strip()
    return os.getenv("LLM_MODEL", "groq/openai/gpt-oss-20b").strip()


def load_key_pool() -> list[tuple[str, str]]:
    """
    Reads LLM_API_KEYS from the environment and assigns key_1, key_2, ... in order.
    Example: 'abc,def' -> [('key_1', 'abc'), ('key_2', 'def')]
    """
    raw_keys = os.getenv("LLM_API_KEYS", "")
    keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
    return [(f"key_{idx}", key) for idx, key in enumerate(keys, start=1)]


def is_failure_simulation_enabled() -> bool:
    """Checks whether X-Simulate-Failure is allowed by environment config."""
    return os.getenv("ENABLE_FAILURE_SIMULATION", "false").strip().lower() == "true"


def parse_failure_header(header_value: str | None) -> dict[str, int]:
    """
    Parses 'X-Simulate-Failure' header into a dictionary.
    Examples:
      'key_1=429'            -> {'key_1': 429}
      'key_1=429, key_2=500' -> {'key_1': 429, 'key_2': 500}
      '*=429'                -> {'*': 429}
    """
    if not header_value or not is_failure_simulation_enabled():
        return {}

    simulated: dict[str, int] = {}
    for part in header_value.split(","):
        part = part.strip()
        if "=" not in part:
            continue
        key_id, status_str = part.split("=", 1)
        key_id = key_id.strip()
        try:
            status_code = int(status_str.strip())
        except ValueError:
            continue
        if status_code in VALID_SIMULATED_STATUSES:
            simulated[key_id] = status_code

    return simulated


def _make_cache_key(model: str, messages: list[dict[str, str]]) -> str:
    """Creates a deterministic SHA-256 key from the model name and full message history."""
    payload = json.dumps({"model": model, "messages": messages}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_retryable_error(exc: Exception) -> bool:
    """
    Returns True if the error is a rate limit (429), quota issue, or 5xx server error
    that should trigger failover to the next API key.
    """
    if isinstance(
        exc,
        (
            RateLimitError,
            ServiceUnavailableError,
            InternalServerError,
            APIConnectionError,
            Timeout,
        ),
    ):
        return True

    if isinstance(exc, APIError):
        status = getattr(exc, "status_code", None)
        if status == 429 or (isinstance(status, int) and status >= 500):
            return True

    msg = str(exc).lower()
    if any(word in msg for word in ("429", "rate limit", "quota", "resource exhausted")):
        return True

    return False


def generate_reply(
    messages: list[dict[str, str]],
    simulate_failure_header: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Sends the conversation to LiteLLM with caching and automatic key failover.
    Returns (reply_text, metadata_dict).
    Raises AllProvidersUnavailableError if every key fails.
    """
    key_pool = load_key_pool()
    if not key_pool:
        raise AllProvidersUnavailableError("all providers unavailable")

    simulated_failures = parse_failure_header(simulate_failure_header)
    use_cache = CACHE_ENABLED and len(simulated_failures) == 0

    start_time = time.perf_counter()

    # 1. Check in-memory cache first (using the primary key's model)
    primary_model = resolve_model_for_key(key_pool[0][1])
    if use_cache:
        cache_key = _make_cache_key(primary_model, messages)
        cached = _RESPONSE_CACHE.get(cache_key)
        if cached is not None:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            return cached["reply"], {
                "resolution_source": "LLM_CACHE",
                "model": cached["model"],
                "latency_ms": latency_ms,
                "input_tokens": cached["input_tokens"],
                "output_tokens": cached["output_tokens"],
                "key_id": None,
                "fallback_attempts": 0,
            }

    # 2. Live inference with key failover loop
    fallback_attempts = 0

    for key_id, api_key in key_pool:
        if "*" in simulated_failures or key_id in simulated_failures:
            fallback_attempts += 1
            continue

        model = resolve_model_for_key(api_key)

        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                api_key=api_key,
                caching=False,
            )
        except Exception as exc:
            if _is_retryable_error(exc):
                fallback_attempts += 1
                continue
            fallback_attempts += 1
            continue

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        reply_text = (response.choices[0].message.content or "").strip()
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

        # Save to cache for identical future requests
        if CACHE_ENABLED:
            cache_key = _make_cache_key(model, messages)
            _RESPONSE_CACHE[cache_key] = {
                "reply": reply_text,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

        metadata = {
            "resolution_source": "LIVE_INFERENCE",
            "model": model,
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "key_id": key_id,
            "fallback_attempts": fallback_attempts,
        }
        return reply_text, metadata

    raise AllProvidersUnavailableError("all providers unavailable")