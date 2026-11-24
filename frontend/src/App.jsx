import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import "./App.css";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const STARTER_PROMPTS = [
  "Explain how SQLite WAL mode works in 3 bullets",
  "What is API key failover and why does it matter?",
  "Write a short Python function to reverse a linked list",
];

export default function App() {
  const [sessionId, setSessionId] = useState(() => crypto.randomUUID());
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const bottomRef = useRef(null);
  const textareaRef = useRef(null);

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

    // Clear composer immediately on send
    setInput("");
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
        const message = Array.isArray(data.detail)
          ? data.detail
              .map((item) => item?.msg || JSON.stringify(item))
              .join("; ")
          : typeof data.detail === "string"
          ? data.detail
          : `Request failed (${res.status})`;
        throw new Error(message);
      }

      setMessages((prev) => [
        ...prev,
        { role: "user", content: trimmed },
        { role: "assistant", content: data.reply, metadata: data.metadata },
      ]);
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
    setSessionId(crypto.randomUUID());
    setMessages([]);
    setError("");
    setInput("");
    textareaRef.current?.focus();
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="header-left">
          <span className="brand-dot" aria-hidden="true" />
          <h1>Lyftr AI</h1>
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
        </div>

        <button
          type="button"
          className="new-chat-btn"
          onClick={handleNewSession}
          data-testid="new-session-button"
          disabled={loading}
        >
          + New Chat
        </button>
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
                  <div className="metadata-bar">
                    <span
                      data-testid="meta-source"
                      className={`badge ${
                        msg.metadata.resolution_source === "LLM_CACHE"
                          ? "cache"
                          : "live"
                      }`}
                    >
                      {msg.metadata.resolution_source}
                    </span>
                    <span data-testid="meta-model" className="meta-item">
                      {msg.metadata.model}
                    </span>
                    <span data-testid="meta-latency" className="meta-item">
                      {msg.metadata.latency_ms} ms
                    </span>
                    <span data-testid="meta-input-tokens" className="meta-item">
                      {msg.metadata.input_tokens} in
                    </span>
                    <span
                      data-testid="meta-output-tokens"
                      className="meta-item"
                    >
                      {msg.metadata.output_tokens} out
                    </span>
                    {msg.metadata.fallback_attempts > 0 && (
                      <span className="badge warn">
                        Fallbacks: {msg.metadata.fallback_attempts}
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
          </form>
        </div>
      </footer>
    </div>
  );
}