import os
from contextlib import asynccontextmanager
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

load_dotenv()

import database
import llm_gateway

DEFAULT_SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are Lyftr AI, a clear, accurate, and concise assistant.",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db()
    llm_gateway.configure_cache()
    yield


app = FastAPI(title="Lyftr AI Gateway", lifespan=lifespan)

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1)
    system_prompt: Optional[str] = Field(default=None, max_length=4000)

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("session_id must not be blank")
        if len(trimmed) > 128:
            raise ValueError("session_id must be at most 128 characters")
        return trimmed

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("message must not be blank")
        if len(trimmed) > 4000:
            raise ValueError("message must be at most 4000 characters")
        return trimmed

    @field_validator("system_prompt")
    @classmethod
    def validate_system_prompt(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        trimmed = value.strip()
        return trimmed if trimmed else None


@app.get("/health")
def health_check() -> dict[str, Any]:
    if not database.check_db_health():
        raise HTTPException(status_code=503, detail="database unavailable")
    return {
        "status": "ok",
        "keys_configured": len(llm_gateway.load_key_pool()),
        "cache_enabled": llm_gateway.is_cache_enabled(),
    }


@app.post("/chat")
def chat_endpoint(
    payload: ChatRequest,
    x_simulate_failure: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    try:
        prior_messages = database.get_session_messages(payload.session_id)
    except KeyError:
        prior_messages = []

    stored_prompt = database.get_session_system_prompt(payload.session_id)
    effective_system_prompt = (
        payload.system_prompt
        if payload.system_prompt is not None
        else (stored_prompt or DEFAULT_SYSTEM_PROMPT)
    )

    llm_messages: list[dict[str, str]] = [
        {"role": "system", "content": effective_system_prompt}
    ]
    for msg in prior_messages:
        llm_messages.append({"role": msg["role"], "content": msg["content"]})
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
        system_prompt=payload.system_prompt,
    )

    return {
        "session_id": payload.session_id,
        "reply": reply,
        "metadata": metadata,
    }


@app.get("/sessions/{session_id}/messages")
def session_messages_endpoint(session_id: str) -> dict[str, Any]:
    try:
        messages = database.get_session_messages(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    return {
        "session_id": session_id,
        "messages": messages,
    }