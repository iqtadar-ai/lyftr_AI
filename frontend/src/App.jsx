import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import "./App.css";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const STARTER_PROMPTS = [
  "Explain how SQLite WAL mode works in 3 bullets",
  "What is API key failover and why does it matter?",
  "Write a short Python function to reverse a linked list",
];

function getOrCreateSessionId() {
  const existing = localStorage.getItem("lyftr_session_id");
  if (existing) return existing;
  const fresh = crypto.randomUUID();
  localStorage.setItem("lyftr_session_id", fresh);
  return fresh;
}

function formatModelLabel(modelString) {
  if (!modelString) return "gpt-oss-20b";
  const parts = modelString.split("/");
  return parts[parts.length - 1];
}

export default function App() {
  const [sessionId, setSessionId] = useState(getOrCreateSessionId);
  const [sessions, setSessions] = useState([]);
  const [activeModel, setActiveModel] = useState("gpt-oss-20b");
  const [didFailover, setDidFailover] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(() => window.innerWidth > 768);

  const bottomRef = useRef(null);
  const textareaRef = useRef(null);

  const refreshSessions = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/sessions`);
      if (!res.ok) return;
      const data = await res.json();
      setSessions(Array.isArray(data.sessions) ? data.sessions : []);
      if (data.default_model) {
        setActiveModel((prev) =>
          prev === "gpt-oss-20b" ? data.default_model : prev
        );
      }
    } catch {
      // Ignore sidebar refresh errors if backend is momentarily unreachable
    }
  }, []);

  useEffect(() => {
    refreshSessions();
  }, [refreshSessions]);

  useEffect(() => {
    let cancelled = false;
    async function fetchHistory() {
      try {
        setError("");
        const res = await fetch(
          `${API_BASE_URL}/sessions/${encodeURIComponent(sessionId)}/messages`
        );
        if (!res.ok) throw new Error("Failed to load conversation history.");
        const data = await res.json();
        if (!cancelled) {
          const loaded = Array.isArray(data.messages) ? data.messages : [];
          setMessages(loaded);
          const lastAssistant = [...loaded]
            .reverse()
            .find((m) => m.role === "assistant" && m.metadata?.model);
          if (lastAssistant) {
            setActiveModel(lastAssistant.metadata.model);
            setDidFailover((lastAssistant.metadata.fallback_attempts || 0) > 0);
          } else {
            setDidFailover(false);
          }
        }
      } catch (err) {
        if (!cancelled) setError(err.message || "Unable to reach backend.");
      }
    }
    fetchHistory();
    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [input]);

  async function sendMessage(text) {
    const trimmed = text.trim();
    if (!trimmed || loading) return;

    setLoading(true);
    setError("");

    try {
      const res = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          message: trimmed,
        }),
      });

      const data = await res.json().catch(() => ({}));

      if (!res.ok) {
        throw new Error(data.detail || `Request failed (${res.status})`);
      }

      setMessages((prev) => [
        ...prev,
        { role: "user", content: trimmed },
        { role: "assistant", content: data.reply, metadata: data.metadata },
      ]);

      if (data.metadata?.model) {
        setActiveModel(data.metadata.model);
        setDidFailover((data.metadata.fallback_attempts || 0) > 0);
      }

      setInput("");
      refreshSessions();
      textareaRef.current?.focus();
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  function handleSubmit(e) {
    e.preventDefault();
    sendMessage(input);
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  }

  function handleNewSession() {
    const fresh = crypto.randomUUID();
    localStorage.setItem("lyftr_session_id", fresh);
    setSessionId(fresh);
    setMessages([]);
    setError("");
    setInput("");
    setDidFailover(false);
    if (window.innerWidth <= 768) setSidebarOpen(false);
    textareaRef.current?.focus();
  }

  function handleSelectSession(id) {
    localStorage.setItem("lyftr_session_id", id);
    setSessionId(id);
    setError("");
    if (window.innerWidth <= 768) setSidebarOpen(false);
  }

  async function handleDeleteSession(e, id) {
    e.stopPropagation();
    try {
      await fetch(`${API_BASE_URL}/sessions/${encodeURIComponent(id)}`, {
        method: "DELETE",
      });
      setSessions((prev) => prev.filter((s) => s.session_id !== id));
      if (id === sessionId) {
        handleNewSession();
      }
    } catch {
      // Ignore delete errors
    }
  }

  return (
    <div className="layout-root">
      {sidebarOpen && (
        <div
          className="sidebar-backdrop"
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* History Sidebar */}
      <aside className={`sidebar ${sidebarOpen ? "open" : "closed"}`}>
        <div className="sidebar-top">
          <button
            type="button"
            className="sidebar-new-btn"
            onClick={handleNewSession}
            data-testid="new-session-button"
            disabled={loading}
          >
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
            >
              <path d="M12 5v14M5 12h14" />
            </svg>
            <span>New chat</span>
          </button>
        </div>

        <div className="sidebar-label">Recent</div>

        <nav className="session-list" aria-label="Chat history">
          {sessions.length === 0 ? (
            <div className="sidebar-empty">No saved chats yet</div>
          ) : (
            sessions.map((s) => (
              <div
                key={s.session_id}
                role="button"
                tabIndex={0}
                className={`session-item ${
                  s.session_id === sessionId ? "active" : ""
                }`}
                onClick={() => handleSelectSession(s.session_id)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleSelectSession(s.session_id);
                }}
              >
                <span className="session-title">{s.title}</span>
                <button
                  type="button"
                  className="session-del-btn"
                  onClick={(e) => handleDeleteSession(e, s.session_id)}
                  title="Delete chat"
                  aria-label="Delete chat"
                >
                  ×
                </button>
              </div>
            ))
          )}
        </nav>
      </aside>

      {/* Main Column */}
      <div className="app-shell">
        <header className="app-header">
          <div className="header-left">
            <span
                data-testid="session-id"
                style={{
                  position: "absolute",
                  width: "1px",
                  height: "1px",
                  overflow: "hidden",
                  clip: "rect(0, 0, 0, 0)",
                  whiteSpace: "nowrap",
                }}
              >
                {sessionId}
              </span>
            <button
              type="button"
              className="icon-btn"
              onClick={() => setSidebarOpen((v) => !v)}
              aria-label="Toggle sidebar"
              title="Toggle history"
            >
              <svg
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
              >
                <path d="M3 6h18M3 12h18M3 18h18" />
              </svg>
            </button>
            <h1>Lyftr AI</h1>
          </div>

          {!sidebarOpen && (
            <button
              type="button"
              className="icon-btn"
              onClick={handleNewSession}
              disabled={loading}
              title="New chat"
              aria-label="New chat"
            >
              <svg
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M12 20h9" />
                <path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z" />
              </svg>
            </button>
          )}
        </header>

        <main className="chat-scroll" data-testid="message-list">
          <div className="chat-container">
            {messages.length === 0 && !loading && (
              <section className="empty-state">
                <h2>What can I help with?</h2>
                <div className="starter-grid">
                  {STARTER_PROMPTS.map((prompt) => (
                    <button
                      key={prompt}
                      type="button"
                      className="starter-chip"
                      onClick={() => sendMessage(prompt)}
                    >
                      {prompt}
                    </button>
                  ))}
                </div>
              </section>
            )}

            {messages.map((msg, idx) => (
              <article
                key={idx}
                className={`message-row ${msg.role}`}
                data-testid={
                  msg.role === "user" ? "user-message" : "assistant-message"
                }
              >
                {msg.role === "assistant" && (
                  <div className="avatar assistant-avatar" aria-hidden="true">
                    ✦
                  </div>
                )}

                <div className="message-bubble">
                  {msg.role === "assistant" ? (
                    <div className="markdown-body">
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                    </div>
                  ) : (
                    <div className="user-text">{msg.content}</div>
                  )}

                  {msg.role === "assistant" && msg.metadata && (
                    <div className="metadata-bar" data-testid="metadata-badge">
                      <span
                        className={`badge ${
                          msg.metadata.resolution_source === "LLM_CACHE"
                            ? "cache"
                            : "live"
                        }`}
                      >
                        {msg.metadata.resolution_source === "LLM_CACHE"
                          ? "CACHE"
                          : "LIVE"}
                      </span>
                      <span className="meta-item">
                        {msg.metadata.latency_ms} ms
                      </span>
                      <span className="meta-item">
                        {msg.metadata.input_tokens}→
                        {msg.metadata.output_tokens} tok
                      </span>
                      {msg.metadata.fallback_attempts > 0 && (
                        <span className="badge warn">
                          Failover → {formatModelLabel(msg.metadata.model)}
                        </span>
                      )}
                    </div>
                  )}
                </div>
              </article>
            ))}

            {loading && (
              <div
                className="message-row assistant"
                data-testid="loading-indicator"
              >
                <div className="avatar assistant-avatar" aria-hidden="true">
                  ✦
                </div>
                <div className="typing-dots">
                  <span />
                  <span />
                  <span />
                </div>
              </div>
            )}

            <div ref={bottomRef} />
          </div>
        </main>

        <footer className="composer-wrap">
          <div className="composer-inner">
            {error && (
              <div
                className="error-banner"
                role="alert"
                data-testid="error-banner"
              >
                <span>{error}</span>
                <button
                  type="button"
                  className="error-dismiss"
                  onClick={() => setError("")}
                  aria-label="Dismiss error"
                >
                  ×
                </button>
              </div>
            )}

            <form className="composer-card" onSubmit={handleSubmit}>
              <textarea
                ref={textareaRef}
                rows={1}
                className="chat-input"
                placeholder="Ask Lyftr AI..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={loading}
                data-testid="chat-input"
              />

              <div className="composer-toolbar">
                <div className="toolbar-right">
                  <div
                    className={`model-pill ${didFailover ? "failover" : ""}`}
                    title={`Active model: ${activeModel}`}
                  >
                    <span>{formatModelLabel(activeModel)}</span>
                    {didFailover && (
                      <span className="failover-tag">· Failover</span>
                    )}
                  </div>

                  <button
                    type="submit"
                    className="send-btn"
                    disabled={loading || !input.trim()}
                    data-testid="send-button"
                    aria-label="Send message"
                  >
                    <svg
                      width="16"
                      height="16"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2.4"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <path d="M12 19V5" />
                      <path d="M5 12l7-7 7 7" />
                    </svg>
                  </button>
                </div>
              </div>
            </form>
          </div>
        </footer>
      </div>
    </div>
  );
}