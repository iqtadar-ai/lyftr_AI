# Lyftr AI — Resilient Multi-Turn Chatbot

Full-stack AI chat application built with **FastAPI**, **LiteLLM**, **SQLite (WAL mode)**, and **React (Vite)**, featuring automatic cross-provider API key failover, deterministic SHA-256 response caching, persistent session history, and multi-stage Docker deployment.

## Architecture

- **Backend (`backend/`)**: FastAPI service exposing `/health`, `/chat`, `/sessions`, and `/sessions/{session_id}/messages`.
- **LLM Gateway (`backend/llm_gateway.py`)**: Rotates across `LLM_API_KEYS` (`key_1`, `key_2`, ...) on HTTP `429` or `5xx` errors, switches provider models automatically by key prefix, and caches identical conversation states in memory (`LLM_CACHE`).
- **Persistence (`backend/database.py`)**: SQLite database in WAL mode storing chat turns atomically only after successful inference.
- **Frontend (`frontend/`)**: Responsive React interface with collapsible chat history sidebar, Markdown rendering, active model badge, and telemetry pills.
## Request & Failover Flow

```mermaid
sequenceDiagram
    autonumber
    actor User as React UI
    participant API as FastAPI (main.py)
    participant DB as SQLite (WAL)
    participant GW as LLM Gateway
    participant LLM as Groq / Gemini

    User->>API: POST /chat (session_id, message)
    API->>DB: Fetch prior session messages
    DB-->>API: Conversation history
    API->>GW: generate_reply(messages)

    alt SHA-256 Cache Hit (no simulation header)
        GW-->>API: Cached reply (LLM_CACHE, 0 ms)
    else Cache Miss
        loop Rotate through LLM_API_KEYS (key_1 → key_2 → key_3)
            GW->>LLM: litellm.completion(model, api_key)
            alt 429 Rate Limit or 5xx Error
                LLM-->>GW: Retryable failure (increment fallback_attempts)
            else 200 OK
                LLM-->>GW: Generated reply + token usage
            end
        end
    end

    alt Inference Succeeded
        API->>DB: Atomically save user & assistant turn
        API-->>User: 200 OK (reply + metadata)
    else All Keys Exhausted
        Note over API,DB: Skip SQLite write to keep history clean
        API-->>User: 503 Service Unavailable
    end
```


## Quick Start (Docker Compose)


## Container & Persistence Topology

```mermaid
flowchart LR
    Browser["Browser Client"]

    subgraph Docker["Docker Compose Network"]
        FE["lyftr-frontend<br/>Nginx (3000 → 80)"]
        BE["lyftr-backend<br/>FastAPI + LiteLLM (8000 → 8000)"]
        VOL[("sqlite_data Volume<br/>/data/app.db (WAL)")]
    end

    subgraph Providers["External LLM Providers"]
        GROQ["Groq API<br/>(key_1, key_2)"]
        GEMINI["Gemini API<br/>(key_3 fallback)"]
    end

    Browser -->|"GET / (Static SPA)"| FE
    Browser -->|"REST (/chat, /sessions, /health)"| BE
    BE <-->|"Atomic Turn Persistence"| VOL
    BE -->|"Primary Inference"| GROQ
    BE -.->|"Failover on 429 / 5xx"| GEMINI
```

1. Stop any local Uvicorn or Vite dev servers occupying ports `8000` or `3000`.
2. Copy `.env.example` to `.env` and add your API keys:
   ```bash
   cp .env.example .env