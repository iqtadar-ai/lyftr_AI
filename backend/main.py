import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

import database
import llm_gateway

# Load root .env when running locally (Docker injects env vars directly)
ROOT_ENV = Path(__file__).resolve().parent.parent / ".env"
if ROOT_ENV.exists():
    load_dotenv(ROOT_ENV)

SYSTEM_PROMPT = (
    "You are a helpful, concise AI assistant. "
    "Answer clearly and accurately using the conversation history for context."
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Runs once on startup to ensure the SQLite schema exists."""
    database.init_db()
    yield


app = FastAPI(title="Lyftr AI Chatbot API", lifespan=lifespan)

# Configure CORS from ALLOWED_ORIGINS
raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173")
allowed_origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)

    @field_validator("session_id", "message")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be empty or whitespace")
        return cleaned


@app.get("/health")
def health() -> dict[str, str]:
    """Returns 200 OK only if SQLite is reachable."""
    if not database.check_db_health():
        raise HTTPException(status_code=503, detail="database unavailable")
    return {"status": "ok"}


@app.get("/sessions/{session_id}/messages")
def get_messages(session_id: str) -> dict[str, Any]:
    """Returns chronological chat history for a session (empty list if new)."""
    messages = database.get_session_messages(session_id)
    return {
        "session_id": session_id,
        "messages": messages,
    }

@app.get("/sessions")
def get_sessions() -> dict[str, Any]:
    """Returns all stored sessions for the sidebar and the primary configured model."""
    pool = llm_gateway.load_key_pool()
    default_model = (
        llm_gateway.resolve_model_for_key(pool[0][1])
        if pool
        else os.getenv("GROQ_MODEL", "groq/openai/gpt-oss-20b")
    )
    return {
        "sessions": database.list_sessions(),
        "default_model": default_model,
    }


@app.delete("/sessions/{session_id}")
def remove_session(session_id: str) -> dict[str, str]:
    """Deletes a conversation from SQLite."""
    database.delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}


@app.post("/chat")
def chat(
    payload: ChatRequest,
    x_simulate_failure: str | None = Header(default=None, alias="X-Simulate-Failure"),
) -> dict[str, Any]:
    """
    Processes one chat turn:
    1. Loads prior history from SQLite.
    2. Calls LiteLLM gateway (with cache + key failover).
    3. Saves user + assistant messages atomically ONLY on success.
    """
    history = database.get_session_messages(payload.session_id)

    # Build the messages array for the LLM (system prompt is NOT saved to SQLite)
    llm_messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in history:
        llm_messages.append({"role": item["role"], "content": item["content"]})
    llm_messages.append({"role": "user", "content": payload.message})

    try:
        reply, metadata = llm_gateway.generate_reply(
            messages=llm_messages,
            simulate_failure_header=x_simulate_failure,
        )
    except llm_gateway.AllProvidersUnavailableError:
        raise HTTPException(status_code=503, detail="all providers unavailable")

    database.save_turn(
        session_id=payload.session_id,
        user_message=payload.message,
        assistant_reply=reply,
        metadata=metadata,
    )

    return {
        "session_id": payload.session_id,
        "reply": reply,
        "metadata": metadata,
    }