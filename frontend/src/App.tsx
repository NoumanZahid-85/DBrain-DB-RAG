import React, { useEffect, useRef, useState } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

const DB_CONFIG = {
  postgresql: { label: "PostgreSQL", color: "#818cf8", bg: "rgba(99,102,241,0.08)", border: "rgba(99,102,241,0.28)", dot: "#6366f1" },
  mysql:      { label: "MySQL",      color: "#22d3ee", bg: "rgba(6,182,212,0.08)",  border: "rgba(6,182,212,0.28)",  dot: "#06b6d4" },
  mongodb:    { label: "MongoDB",    color: "#34d399", bg: "rgba(16,185,129,0.08)", border: "rgba(16,185,129,0.28)", dot: "#10b981" },
  ambiguous:  { label: "All DBs",   color: "#94a3b8", bg: "rgba(148,163,184,0.06)",border: "rgba(148,163,184,0.2)", dot: "#64748b" },
};

type DBFilter = "postgresql" | "mysql" | "mongodb" | null;

interface QueryResponse {
  query: string; answer: string; sources: string[];
  dbs_used: string[]; detected_db: string | null;
  chunks_retrieved: number; chunks_used: number; refused: boolean;
  top_chunk_score: number; latency_ms: number;
  request_id?: string; session_id?: string;
}

interface ChatMessage { role: "user" | "assistant"; content: string; }

const SAMPLE_QUERIES: { q: string; db: DBFilter }[] = [
  { q: "What types of indexes does PostgreSQL support?", db: "postgresql" },
  { q: "How does InnoDB handle deadlocks in MySQL?", db: "mysql" },
  { q: "How does the MongoDB aggregation pipeline work?", db: "mongodb" },
  { q: "Compare index creation syntax: PostgreSQL vs MySQL vs MongoDB", db: null },
  { q: "How does VACUUM work and when should I run it?", db: "postgresql" },
  { q: "What is replication lag and how do I monitor it?", db: null },
];

const SESSION_KEY = "dbrain_session_id";

const CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

  *, *::before, *::after { box-sizing: border-box; }

  body {
    margin: 0;
    background: #080a10;
    color: #c8d0e0;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    -webkit-font-smoothing: antialiased;
    min-height: 100vh;
  }

  /* ── Layout ── */
  .app-shell { max-width: 880px; margin: 0 auto; padding: 52px 24px 80px; }

  /* ── Hero ── */
  .hero-eyebrow {
    display: inline-flex; align-items: center; gap: 7px;
    background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.3);
    border-radius: 20px; padding: 4px 13px; margin-bottom: 18px;
  }
  .hero-dot {
    width: 7px; height: 7px; border-radius: 50%; background: #6366f1;
    animation: pulse 2s ease-in-out infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(99,102,241,0.6); }
    50%       { opacity: 0.7; box-shadow: 0 0 0 5px rgba(99,102,241,0); }
  }
  .hero-eyebrow-text { font-size: 11px; font-weight: 700; color: #818cf8; letter-spacing: 0.07em; text-transform: uppercase; }

  .hero-title {
    font-size: clamp(26px, 4vw, 38px); font-weight: 800; color: #f1f5f9;
    letter-spacing: -0.03em; line-height: 1.1; margin: 0 0 10px;
  }
  .hero-title-accent { color: #818cf8; }
  .hero-sub { color: #64748b; font-size: 14.5px; font-weight: 400; margin: 0; line-height: 1.6; }

  /* ── New-Session Button ── */
  .btn-new-session {
    display: flex; align-items: center; gap: 6px;
    padding: 8px 16px; background: #0e1018;
    color: #64748b; border: 1px solid #1c2030; border-radius: 8px;
    cursor: pointer; font-size: 12.5px; font-weight: 600;
    font-family: 'Inter', sans-serif;
    transition: border-color 0.2s, color 0.2s;
  }
  .btn-new-session:hover { border-color: #6366f1; color: #c7d2fe; }

  /* ── DB Filter Tabs ── */
  .db-tabs { display: flex; flex-wrap: wrap; gap: 7px; margin-bottom: 14px; }
  .db-tab {
    padding: 6px 15px; border-radius: 24px; font-size: 11px; font-weight: 700;
    letter-spacing: 0.04em; text-transform: uppercase; cursor: pointer;
    border: 1px solid #1c2030; background: #0c0e16; color: #4b5873;
    font-family: 'Inter', sans-serif;
    transition: all 0.18s cubic-bezier(0.4,0,0.2,1);
  }
  .db-tab:hover { border-color: #6366f1; color: #c7d2fe; }
  .db-tab.active-pg  { background: rgba(99,102,241,0.1); color: #818cf8; border-color: rgba(99,102,241,0.4); }
  .db-tab.active-my  { background: rgba(6,182,212,0.08); color: #22d3ee; border-color: rgba(6,182,212,0.4); }
  .db-tab.active-mg  { background: rgba(16,185,129,0.08);color: #34d399; border-color: rgba(16,185,129,0.4); }
  .db-tab.active-all { background: rgba(99,102,241,0.06);color: #818cf8; border-color: rgba(99,102,241,0.3); }

  /* ── Query Input ── */
  .query-bar {
    display: flex; gap: 10px; margin-bottom: 26px;
    background: #0c0e16; border: 1px solid #1c2030; border-radius: 12px;
    padding: 4px 4px 4px 0;
    transition: border-color 0.2s, box-shadow 0.2s;
  }
  .query-bar:focus-within {
    border-color: #6366f1;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12);
  }
  .query-prompt {
    display: flex; align-items: center; padding: 0 14px;
    font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 500;
    color: #6366f1; user-select: none; flex-shrink: 0;
  }
  .query-prompt::after {
    content: ''; display: inline-block; width: 8px; height: 16px;
    background: #6366f1; margin-left: 6px;
    animation: blink 1.1s step-end infinite; border-radius: 1px;
  }
  @keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: 0; } }
  .query-input {
    flex: 1; background: transparent; border: none; outline: none;
    color: #e2e8f0; font-size: 14.5px; font-family: 'Inter', sans-serif;
    padding: 13px 0; caret-color: #6366f1;
  }
  .query-input::placeholder { color: #2d3650; }
  .btn-search {
    padding: 0 26px; border-radius: 8px; border: none; cursor: pointer;
    font-size: 14px; font-weight: 600; font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
    color: #fff; flex-shrink: 0;
    transition: opacity 0.2s, transform 0.15s;
    box-shadow: 0 4px 16px rgba(99,102,241,0.3);
  }
  .btn-search:hover:not(:disabled) { opacity: 0.9; transform: translateY(-1px); }
  .btn-search:disabled { background: #1c2030; color: #334060; cursor: not-allowed; box-shadow: none; }

  /* ── Error Banner ── */
  .error-banner {
    background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.25);
    border-radius: 10px; padding: 14px 18px; color: #f87171; font-size: 13.5px;
    margin-bottom: 22px;
  }

  /* ── Chat History ── */
  .chat-list { display: flex; flex-direction: column; gap: 14px; margin-bottom: 22px; }
  .chat-row { display: flex; gap: 10px; align-items: flex-start; animation: slideUp 0.28s ease forwards; }
  .chat-row.user { flex-direction: row-reverse; }
  @keyframes slideUp { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }

  .chat-avatar {
    width: 28px; height: 28px; border-radius: 7px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 700;
  }
  .chat-avatar.usr { background: rgba(99,102,241,0.15); color: #818cf8; border: 1px solid rgba(99,102,241,0.25); }
  .chat-avatar.bot { background: rgba(16,185,129,0.1); color: #34d399; border: 1px solid rgba(16,185,129,0.2); }

  .chat-bubble {
    padding: 12px 16px; border-radius: 10px;
    font-size: 14px; line-height: 1.65; max-width: 82%;
  }
  .chat-bubble.user {
    background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.25);
    color: #c7d2fe; font-weight: 500;
  }
  .chat-bubble.assistant {
    background: #0c0e16; border: 1px solid #1a1e2e; color: #c8d0e0;
  }

  .chat-typing {
    display: flex; gap: 4px; align-items: center; padding: 14px 16px;
    background: #0c0e16; border: 1px solid #1a1e2e; border-radius: 10px; max-width: 90px;
  }
  .typing-dot { width: 5px; height: 5px; border-radius: 50%; background: #6366f1; }
  .typing-dot:nth-child(1) { animation: bounce 1.2s 0s infinite; }
  .typing-dot:nth-child(2) { animation: bounce 1.2s 0.15s infinite; }
  .typing-dot:nth-child(3) { animation: bounce 1.2s 0.3s infinite; }
  @keyframes bounce { 0%,60%,100% { transform:translateY(0); } 30% { transform:translateY(-5px); } }

  /* ── Metrics Grid ── */
  .metrics-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(150px,1fr));
    gap: 10px; margin-bottom: 18px;
  }
  .metric-card {
    background: #0c0e16; border: 1px solid #1a1e2e; border-radius: 10px;
    padding: 14px 18px; transition: border-color 0.18s;
  }
  .metric-card:hover { border-color: #6366f1; }
  .metric-label { font-size: 10px; color: #3d4a60; font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 6px; }
  .metric-value { font-size: 17px; font-weight: 700; color: #f1f5f9; font-family: 'JetBrains Mono', monospace; }
  .metric-sub { font-size: 11px; color: #3d4a60; margin-top: 2px; }

  /* ── Latency Bar ── */
  .latency-bar-row { display: flex; gap: 2px; height: 4px; border-radius: 2px; overflow: hidden; margin-top: 8px; }
  .latency-seg { height: 100%; border-radius: 2px; transition: width 0.6s ease; }

  /* ── Refusal Alert ── */
  .refusal-alert {
    display: flex; align-items: flex-start; gap: 12px;
    background: rgba(239,68,68,0.06); border: 1px solid rgba(239,68,68,0.2);
    border-radius: 10px; padding: 14px 18px; margin-bottom: 16px;
  }
  .refusal-icon { font-size: 18px; flex-shrink: 0; }
  .refusal-text { font-size: 13.5px; color: #fca5a5; line-height: 1.55; }

  /* ── Propose Action ── */
  .btn-propose {
    padding: 8px 20px;
    background: rgba(16,185,129,0.08); color: #34d399;
    border: 1px solid rgba(16,185,129,0.3); border-radius: 8px;
    cursor: pointer; font-size: 12px; font-weight: 700;
    font-family: 'Inter', sans-serif; letter-spacing: 0.05em; text-transform: uppercase;
    transition: all 0.18s;
  }
  .btn-propose:hover { background: rgba(16,185,129,0.14); border-color: rgba(16,185,129,0.5); }

  /* ── Sources ── */
  .source-pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 20px; font-size: 11.5px; font-weight: 500;
  }
  .source-dot { width: 5px; height: 5px; border-radius: 50%; flex-shrink: 0; }
  .sources-label { font-size: 10px; color: #2d3650; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 10px; }

  /* ── Markdown ── */
  .md-body p { margin: 0 0 14px; line-height: 1.75; }
  .md-body h1, .md-body h2, .md-body h3 { color: #f1f5f9; font-weight: 700; margin: 20px 0 10px; }
  .md-body ul, .md-body ol { margin: 0 0 14px; padding-left: 20px; }
  .md-body li { margin-bottom: 5px; line-height: 1.6; }
  .md-body pre { background: #080a10 !important; border: 1px solid #1a1e2e; border-radius: 8px; padding: 14px; overflow-x: auto; margin: 14px 0; }
  .md-body code { font-family: 'JetBrains Mono', monospace !important; font-size: 12.5px !important; color: #818cf8 !important; background: rgba(99,102,241,0.08); padding: 2px 6px; border-radius: 4px; }
  .md-body pre code { background: transparent !important; padding: 0 !important; }

  /* ── Sample Queries ── */
  .samples-label { font-size: 10.5px; color: #2d3650; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 14px; }
  .samples-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  @media (max-width: 560px) { .samples-grid { grid-template-columns: 1fr; } }

  .sample-card {
    display: flex; flex-direction: column; align-items: flex-start; gap: 9px;
    padding: 15px 17px; background: #0c0e16; border: 1px solid #1a1e2e;
    border-radius: 10px; cursor: pointer; text-align: left;
    transition: all 0.2s cubic-bezier(0.4,0,0.2,1);
    font-family: 'Inter', sans-serif;
  }
  .sample-card:hover {
    transform: translateY(-2px); border-color: #6366f1;
    background: #0d1020;
    box-shadow: 0 8px 24px rgba(99,102,241,0.1);
  }
  .sample-text { font-size: 13.5px; color: #b0bccf; font-weight: 500; line-height: 1.45; }

  /* ── Divider ── */
  .section-divider { height: 1px; background: #111525; margin: 40px 0 28px; }

  /* ── Result fade-in ── */
  .result-enter { animation: fadeIn 0.32s ease forwards; }
  @keyframes fadeIn { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:translateY(0); } }
`;

export default function App() {
  const [query, setQuery] = useState("");
  const [dbFilter, setDbFilter] = useState<DBFilter>(null);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [history, setHistory] = useState<ChatMessage[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [actionMsg, setActionMsg] = useState("");
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const stored = localStorage.getItem(SESSION_KEY);
    if (stored) {
      setSessionId(stored);
    } else {
      axios.post(`${API_URL}/api/v1/sessions`, {})
        .then((res) => {
          localStorage.setItem(SESSION_KEY, res.data.session_id);
          setSessionId(res.data.session_id);
        })
        .catch(() => {});
    }
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history, loading]);

  const startNewSession = async () => {
    setHistory([]); setResult(null); setError(""); setActionMsg("");
    try {
      const res = await axios.post(`${API_URL}/api/v1/sessions`, {});
      localStorage.setItem(SESSION_KEY, res.data.session_id);
      setSessionId(res.data.session_id);
    } catch { /* offline */ }
  };

  const handleQuery = async (q?: string, db?: DBFilter) => {
    const activeQuery = q ?? query;
    const activeDb = db !== undefined ? db : dbFilter;
    if (!activeQuery.trim()) return;
    setLoading(true); setError(""); setActionMsg(""); setResult(null);
    setHistory((prev) => [...prev, { role: "user", content: activeQuery }]);
    try {
      const res = await axios.post<QueryResponse>(`${API_URL}/api/v1/query`, {
        query: activeQuery, db_filter: activeDb, session_id: sessionId || undefined,
      });
      setResult(res.data);
      setHistory((prev) => [...prev, { role: "assistant", content: res.data.answer }]);
      if (q) setQuery(q);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Could not connect to the server. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  const proposeAction = async () => {
    if (!result || !sessionId) return;
    setActionMsg("");
    try {
      await axios.post(`${API_URL}/api/v1/actions`, {
        session_id: sessionId, query: result.query,
        action: result.answer, risk_level: result.refused ? "high" : "medium",
      });
      setActionMsg("Submitted to the Module 5/6 approval workflow.");
    } catch (err: any) {
      setActionMsg(err.response?.data?.detail || "Failed to submit.");
    }
  };

  const activeTabClass = (db: DBFilter) => {
    if (dbFilter !== db) return "db-tab";
    if (db === "postgresql") return "db-tab active-pg";
    if (db === "mysql")      return "db-tab active-my";
    if (db === "mongodb")    return "db-tab active-mg";
    return "db-tab active-all";
  };

  return (
    <div>
      <style>{CSS}</style>
      <div className="app-shell">

        {/* ── Header ── */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 16, marginBottom: 36 }}>
          <div>
            <div className="hero-eyebrow">
              <span className="hero-dot" />
              <span className="hero-eyebrow-text">Module 7 · RAG Pipeline · v1.1.0</span>
            </div>
            <h1 className="hero-title">
              DBrain <span className="hero-title-accent">Performance</span> Co-Pilot
            </h1>
            <p className="hero-sub">
              Hybrid BM25 + semantic retrieval over PostgreSQL, MySQL &amp; MongoDB documentation.
              Ask anything — index strategies, deadlocks, replication, VACUUM.
            </p>
          </div>
          <button className="btn-new-session" onClick={startNewSession}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
              <path d="M19 13H13v6h-2v-6H5v-2h6V5h2v6h6v2z"/>
            </svg>
            New Session
          </button>
        </div>

        {/* ── DB Filter Tabs ── */}
        <div className="db-tabs">
          {([null, "postgresql", "mysql", "mongodb"] as DBFilter[]).map((db) => {
            const cfg = DB_CONFIG[db || "ambiguous"];
            return (
              <button key={db ?? "all"} className={activeTabClass(db)} onClick={() => setDbFilter(db)}>
                {db ? cfg.label : "All Databases"}
              </button>
            );
          })}
        </div>

        {/* ── Query Bar ── */}
        <div className="query-bar">
          <span className="query-prompt">dbrain</span>
          <input
            id="query-input"
            className="query-input"
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleQuery()}
            placeholder="Ask about indexing, VACUUM, deadlocks, aggregation pipelines…"
            autoComplete="off"
          />
          <button className="btn-search" onClick={() => handleQuery()} disabled={loading}>
            {loading ? "Searching…" : "Search"}
          </button>
        </div>

        {/* ── Error ── */}
        {error && <div className="error-banner">⚠ {error}</div>}

        {/* ── Chat History ── */}
        {history.length > 0 && (
          <div className="chat-list">
            {history.map((msg, i) => (
              <div key={i} className={`chat-row ${msg.role}`}>
                <div className={`chat-avatar ${msg.role === "user" ? "usr" : "bot"}`}>
                  {msg.role === "user" ? "U" : "AI"}
                </div>
                <div className={`chat-bubble ${msg.role}`}>
                  {msg.role === "assistant" ? (
                    <div className="md-body"><ReactMarkdown>{msg.content}</ReactMarkdown></div>
                  ) : (
                    msg.content
                  )}
                </div>
              </div>
            ))}
            {loading && (
              <div className="chat-row">
                <div className="chat-avatar bot">AI</div>
                <div className="chat-typing">
                  <div className="typing-dot" />
                  <div className="typing-dot" />
                  <div className="typing-dot" />
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>
        )}

        {/* ── Result Panel ── */}
        {result && (
          <div className="result-enter">
            {/* Metrics */}
            <div className="metrics-grid">
              {/* Latency */}
              <div className="metric-card">
                <div className="metric-label">Total Latency</div>
                <div className="metric-value">{result.latency_ms.toFixed(0)}<span style={{ fontSize: 12, color: "#4b5873", fontWeight: 500 }}> ms</span></div>
                <div className="metric-sub">end-to-end</div>
              </div>

              {/* Chunks */}
              <div className="metric-card">
                <div className="metric-label">Chunks</div>
                <div className="metric-value">{result.chunks_used}<span style={{ fontSize: 12, color: "#4b5873", fontWeight: 500 }}>/{result.chunks_retrieved}</span></div>
                <div className="metric-sub">used / retrieved</div>
              </div>

              {/* Top Score */}
              <div className="metric-card">
                <div className="metric-label">Max Relevance</div>
                <div className="metric-value">{result.top_chunk_score.toFixed(3)}</div>
                <div className="latency-bar-row">
                  <div className="latency-seg" style={{ width: `${result.top_chunk_score * 100}%`, background: result.top_chunk_score > 0.6 ? "#4ade80" : result.top_chunk_score > 0.3 ? "#facc15" : "#f87171" }} />
                  <div className="latency-seg" style={{ flex: 1, background: "#1a1e2e" }} />
                </div>
              </div>

              {/* Routing */}
              <div className="metric-card">
                <div className="metric-label">Route</div>
                <div className="metric-value" style={{ fontSize: 14, color: result.detected_db ? DB_CONFIG[result.detected_db as keyof typeof DB_CONFIG]?.color : "#64748b" }}>
                  {result.detected_db ? DB_CONFIG[result.detected_db as keyof typeof DB_CONFIG]?.label ?? result.detected_db : "Multi-DB"}
                </div>
                <div className="metric-sub">{result.detected_db ? "auto-detected" : "cross-db search"}</div>
              </div>
            </div>

            {/* Refusal */}
            {result.refused && (
              <div className="refusal-alert">
                <span className="refusal-icon">🛡</span>
                <div className="refusal-text">
                  <strong>Response withheld.</strong> The retrieved documentation does not contain sufficient evidence to answer this question reliably, or the request was identified as a destructive command.
                </div>
              </div>
            )}

            {/* Propose Action */}
            {!result.refused && result.sources.length > 0 && (
              <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
                <button className="btn-propose" onClick={proposeAction}>Propose Action</button>
                {actionMsg && (
                  <span style={{ fontSize: 12, color: actionMsg.includes("Failed") ? "#f87171" : "#64748b" }}>
                    {actionMsg}
                  </span>
                )}
              </div>
            )}

            {/* Sources */}
            {result.sources.length > 0 && (
              <div style={{ marginBottom: 32 }}>
                <div className="sources-label">Retrieved References</div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 7 }}>
                  {result.sources.map((s) => {
                    const db = result.dbs_used.find((d) => s.toLowerCase().includes(d)) || "ambiguous";
                    const c = DB_CONFIG[db as keyof typeof DB_CONFIG] || DB_CONFIG.ambiguous;
                    return (
                      <span key={s} className="source-pill" style={{ background: c.bg, color: c.color, border: `1px solid ${c.border}` }}>
                        <span className="source-dot" style={{ background: c.dot }} />
                        {s}
                      </span>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── Sample Queries ── */}
        <div className="section-divider" />
        <p className="samples-label">Suggested Queries</p>
        <div className="samples-grid">
          {SAMPLE_QUERIES.map(({ q, db }) => {
            const c = DB_CONFIG[db || "ambiguous"];
            return (
              <button key={q} className="sample-card" onClick={() => handleQuery(q, db)}>
                <span className="source-pill" style={{ background: c.bg, color: c.color, border: `1px solid ${c.border}`, fontSize: 10, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase" }}>
                  <span className="source-dot" style={{ background: c.dot }} />
                  {db ?? "cross-db"}
                </span>
                <span className="sample-text">{q}</span>
              </button>
            );
          })}
        </div>

      </div>
    </div>
  );
}