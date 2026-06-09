import React, { useState } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

// Color coding per database — matches DBrain's dashboard style
const DB_COLORS: Record<string, { bg: string; text: string; border: string; accent: string }> = {
  postgresql: { bg: "rgba(59, 130, 246, 0.08)", text: "#60a5fa", border: "rgba(59, 130, 246, 0.25)", accent: "#3b82f6" },
  mysql:      { bg: "rgba(6, 182, 212, 0.08)", text: "#22d3ee", border: "rgba(6, 182, 212, 0.25)", accent: "#06b6d4" },
  mongodb:    { bg: "rgba(16, 185, 129, 0.08)", text: "#34d399", border: "rgba(16, 185, 129, 0.25)", accent: "#10b981" },
  ambiguous:  { bg: "rgba(148, 163, 184, 0.08)", text: "#94a3b8", border: "rgba(148, 163, 184, 0.25)", accent: "#94a3b8" },
};

type DBFilter = "postgresql" | "mysql" | "mongodb" | null;

interface QueryResponse {
  query: string;
  answer: string;
  sources: string[];
  dbs_used: string[];
  detected_db: string | null;
  chunks_retrieved: number;
  chunks_used: number;
  refused: boolean;
  top_chunk_score: number;
  latency_ms: number;
}

const SAMPLE_QUERIES: { q: string; db: DBFilter }[] = [
  { q: "What types of indexes does PostgreSQL support?", db: "postgresql" },
  { q: "How does InnoDB handle deadlocks in MySQL?", db: "mysql" },
  { q: "How does the MongoDB aggregation pipeline work?", db: "mongodb" },
  { q: "Compare index creation syntax: PostgreSQL vs MySQL vs MongoDB", db: null },
  { q: "How does VACUUM work and when should I run it?", db: "postgresql" },
  { q: "What is replication lag and how do I monitor it?", db: null },
];

export default function App() {
  const [query, setQuery] = useState("");
  const [dbFilter, setDbFilter] = useState<DBFilter>(null);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleQuery = async (q?: string, db?: DBFilter) => {
    const activeQuery = q ?? query;
    const activeDb = db !== undefined ? db : dbFilter;
    if (!activeQuery.trim()) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const res = await axios.post<QueryResponse>(`${API_URL}/query`, {
        query: activeQuery,
        db_filter: activeDb,
      });
      setResult(res.data);
      if (q) setQuery(q);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Something went wrong connection with the server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ minHeight: "100vh", backgroundColor: "#06070a", color: "#e2e8f0", paddingBottom: 60 }}>
      {/* Dynamic CSS Styling Injection */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        
        body {
          margin: 0;
          background-color: #06070a;
          color: #e2e8f0;
          font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
          -webkit-font-smoothing: antialiased;
        }

        .tab-button:hover {
          border-color: #3b82f6 !important;
          color: #f3f4f6 !important;
        }

        .query-card {
          transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .query-card:hover {
          transform: translateY(-2px);
          border-color: #3b82f6 !important;
          background-color: #12141c !important;
        }

        pre {
          background-color: #0b0c10 !important;
          padding: 16px;
          border-radius: 8px;
          border: 1px solid #1f212a;
          overflow-x: auto;
          margin: 16px 0;
        }

        code {
          font-family: 'JetBrains Mono', monospace !important;
          font-size: 13.5px !important;
          color: #60a5fa !important;
        }

        .markdown-content p {
          margin-top: 0;
          margin-bottom: 16px;
          line-height: 1.75;
        }

        .markdown-content h1, .markdown-content h2, .markdown-content h3 {
          color: #f3f4f6;
          margin-top: 24px;
          margin-bottom: 12px;
          font-weight: 600;
        }

        .markdown-content ul, .markdown-content ol {
          margin-top: 0;
          margin-bottom: 16px;
          padding-left: 20px;
        }

        .markdown-content li {
          margin-bottom: 6px;
          line-height: 1.6;
        }

        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }

        .fade-in {
          animation: fadeIn 0.3s cubic-bezier(0.4, 0, 0.2, 1) forwards;
        }
      `}</style>

      <div className="app-container" style={{ maxWidth: 900, margin: "0 auto", padding: "48px 24px" }}>
        
        {/* Header */}
        <div style={{ marginBottom: 36, display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 16 }}>
          <div>
            <div style={{ display: "inline-flex", alignItems: "center", gap: 8, background: "rgba(59, 130, 246, 0.1)", border: "1px solid rgba(59, 130, 246, 0.25)", borderRadius: 20, padding: "4px 12px", marginBottom: 12 }}>
              <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#3b82f6", display: "inline-block" }}></span>
              <span style={{ fontSize: 11, fontWeight: 700, color: "#60a5fa", letterSpacing: "0.05em", textTransform: "uppercase" }}>Module 7 Chat & RAG</span>
            </div>
            <h1 style={{ fontSize: 32, fontWeight: 700, margin: 0, letterSpacing: "-0.02em", color: "#f3f4f6" }}>
              DBrain Performance RAG
            </h1>
            <p style={{ color: "#8a8d9a", margin: "8px 0 0", fontSize: 15, fontWeight: 400 }}>
              AI-Powered Performance Co-Pilot for PostgreSQL, MySQL, and MongoDB
            </p>
          </div>
        </div>

        {/* DB Filter Tabs */}
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 16 }}>
          {([null, "postgresql", "mysql", "mongodb"] as (DBFilter)[]).map((db) => {
            const label = db ? db.toUpperCase() : "ALL DATABASES";
            const colors = DB_COLORS[db || "ambiguous"];
            const active = dbFilter === db;
            return (
              <button
                key={db || "all"}
                onClick={() => setDbFilter(db)}
                className="tab-button"
                style={{
                  padding: "8px 16px",
                  borderRadius: 30,
                  fontSize: 11.5,
                  cursor: "pointer",
                  border: `1px solid ${active ? colors.accent : "#1f212a"}`,
                  background: active ? colors.bg : "#0d0e12",
                  color: active ? colors.text : "#8a8d9a",
                  fontWeight: 600,
                  letterSpacing: "0.03em",
                  transition: "all 0.2s ease",
                  outline: "none",
                }}
              >
                {label}
              </button>
            );
          })}
        </div>

        {/* Query input */}
        <div style={{ display: "flex", gap: 12, marginBottom: 32 }}>
          <div style={{ flex: 1, position: "relative", display: "flex", alignItems: "center" }}>
            <svg
              style={{ position: "absolute", left: 16, width: 18, height: 18, fill: "#52526b" }}
              viewBox="0 0 24 24"
            >
              <path d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z" />
            </svg>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleQuery()}
              placeholder="Ask anything about database performance, indexing, replication, or architecture..."
              style={{
                width: "100%",
                padding: "14px 16px 14px 48px",
                fontSize: 15,
                background: "#0d0e12",
                color: "#f3f4f6",
                border: "1px solid #1f212a",
                borderRadius: 10,
                outline: "none",
                transition: "border-color 0.2s ease, box-shadow 0.2s ease",
              }}
              onFocus={(e) => {
                e.target.style.borderColor = "#3b82f6";
                e.target.style.boxShadow = "0 0 0 3px rgba(59, 130, 246, 0.15)";
              }}
              onBlur={(e) => {
                e.target.style.borderColor = "#1f212a";
                e.target.style.boxShadow = "none";
              }}
            />
          </div>
          <button
            onClick={() => handleQuery()}
            disabled={loading}
            style={{
              padding: "0 28px",
              background: loading ? "#1d4ed850" : "linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)",
              color: loading ? "#93c5fd" : "white",
              border: "none",
              borderRadius: 10,
              cursor: loading ? "not-allowed" : "pointer",
              fontSize: 15,
              fontWeight: 600,
              transition: "all 0.2s ease",
              boxShadow: "0 4px 12px rgba(37, 99, 235, 0.15)",
            }}
          >
            {loading ? "Searching..." : "Search"}
          </button>
        </div>

        {/* Error */}
        {error && (
          <div style={{ background: "rgba(239, 68, 68, 0.1)", border: "1px solid rgba(239, 68, 68, 0.25)", borderRadius: 10, padding: 16, color: "#f87171", marginBottom: 24, fontSize: 14 }}>
            {error}
          </div>
        )}

        {/* Result */}
        {result && (
          <div className="fade-in">
            {/* Clean Metrics Dashboard - No Emojis */}
            <div style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
              gap: 12,
              marginBottom: 20,
            }}>
              <div style={{ background: "#0d0e12", border: "1px solid #1f212a", borderRadius: 8, padding: "12px 16px" }}>
                <div style={{ fontSize: 11, color: "#6b7280", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 4, fontWeight: 600 }}>
                  Latency
                </div>
                <div style={{ fontSize: 16, fontWeight: 600, color: "#f3f4f6" }}>
                  {result.latency_ms.toFixed(2)} ms
                </div>
              </div>

              <div style={{ background: "#0d0e12", border: "1px solid #1f212a", borderRadius: 8, padding: "12px 16px" }}>
                <div style={{ fontSize: 11, color: "#6b7280", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 4, fontWeight: 600 }}>
                  Document Chunks
                </div>
                <div style={{ fontSize: 16, fontWeight: 600, color: "#f3f4f6" }}>
                  {result.chunks_retrieved} retrieved <span style={{ color: "#374151" }}>/</span> {result.chunks_used} used
                </div>
              </div>

              <div style={{ background: "#0d0e12", border: "1px solid #1f212a", borderRadius: 8, padding: "12px 16px" }}>
                <div style={{ fontSize: 11, color: "#6b7280", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 4, fontWeight: 600 }}>
                  Max Relevance
                </div>
                <div style={{ fontSize: 16, fontWeight: 600, color: "#f3f4f6" }}>
                  {result.top_chunk_score.toFixed(4)}
                </div>
              </div>

              <div style={{ background: "#0d0e12", border: "1px solid #1f212a", borderRadius: 8, padding: "12px 16px" }}>
                <div style={{ fontSize: 11, color: "#6b7280", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 4, fontWeight: 600 }}>
                  Source Routing
                </div>
                <div style={{ 
                  fontSize: 16, 
                  fontWeight: 600, 
                  color: result.detected_db ? DB_COLORS[result.detected_db].text : "#94a3b8" 
                }}>
                  {result.detected_db 
                    ? `${result.detected_db.toUpperCase()} (Auto)` 
                    : "Multi-DB Search"
                  }
                </div>
              </div>
            </div>

            {/* Refusal Alert (Clean) */}
            {result.refused && (
              <div style={{
                background: "rgba(239, 68, 68, 0.08)",
                border: "1px solid rgba(239, 68, 68, 0.25)",
                borderRadius: 8,
                padding: "14px 16px",
                color: "#f87171",
                fontSize: 14,
                marginBottom: 16,
                fontWeight: 500,
              }}>
                Security Shield Alert: The retrieved context does not contain sufficient documentation evidence to generate a reliable response.
              </div>
            )}

            {/* Answer Display */}
            <div style={{
              padding: "24px 28px",
              background: "#0d0e12",
              border: `1px solid ${result.refused ? "rgba(239, 68, 68, 0.25)" : "#1f212a"}`,
              borderRadius: 12,
              marginBottom: 20,
            }}>
              <div className="markdown-content" style={{ fontSize: 15, lineHeight: "1.8", color: "#e2e8f0" }}>
                <ReactMarkdown>{result.answer}</ReactMarkdown>
              </div>
            </div>

            {/* Sources List */}
            {result.sources.length > 0 && (
              <div style={{ marginBottom: 32 }}>
                <div style={{ fontSize: 11, color: "#6b7280", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 10 }}>
                  Retrieved References
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                  {result.sources.map((s) => {
                    const db = result.dbs_used.find((d) => s.toLowerCase().includes(d)) || "ambiguous";
                    const c = DB_COLORS[db] || DB_COLORS.ambiguous;
                    return (
                      <span key={s} style={{
                        padding: "4px 12px",
                        borderRadius: 20,
                        fontSize: 12,
                        background: c.bg,
                        color: c.text,
                        border: `1px solid ${c.border}`,
                        fontWeight: 500,
                      }}>
                        {s}
                      </span>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Sample queries */}
        <div style={{ marginTop: 44 }}>
          <p style={{ fontSize: 11, color: "#4b5563", marginBottom: 16, fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase" }}>
            Suggested Queries
          </p>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            {SAMPLE_QUERIES.map(({ q, db }) => {
              const c = DB_COLORS[db || "ambiguous"];
              return (
                <button
                  key={q}
                  onClick={() => handleQuery(q, db)}
                  className="query-card"
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "flex-start",
                    gap: 10,
                    padding: "16px 18px",
                    background: "#0d0e12",
                    border: "1px solid #1f212a",
                    borderRadius: 10,
                    cursor: "pointer",
                    textAlign: "left",
                    fontSize: 14,
                    color: "#d1d5db",
                    outline: "none",
                  }}
                >
                  <span style={{
                    padding: "2px 8px",
                    borderRadius: 12,
                    fontSize: 10,
                    fontWeight: 700,
                    background: c.bg,
                    color: c.text,
                    border: `1px solid ${c.border}`,
                    textTransform: "uppercase",
                    letterSpacing: "0.03em",
                  }}>
                    {db ?? "cross-db"}
                  </span>
                  <span style={{ fontWeight: 500, lineHeight: "1.4" }}>{q}</span>
                </button>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
