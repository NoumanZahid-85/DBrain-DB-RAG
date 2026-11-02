import React, { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./App.css";
import { api, errorMessage, Engine, QueryResponse, Status } from "./api";

const ENGINES: Record<Engine, { label: string; color: string }> = {
  postgresql: { label: "PostgreSQL", color: "var(--pg)" },
  mysql: { label: "MySQL", color: "var(--mysql)" },
  mongodb: { label: "MongoDB", color: "var(--mongo)" },
};

const STARTERS: { engine: Engine | null; q: string }[] = [
  { engine: "postgresql", q: "When should I run VACUUM manually, and what does autovacuum already handle?" },
  { engine: "postgresql", q: "How do I read the output of EXPLAIN ANALYZE?" },
  { engine: "mysql", q: "How should I size the InnoDB buffer pool?" },
  { engine: "mongodb", q: "How do I find slow operations with the database profiler?" },
  { engine: null, q: "Compare how PostgreSQL, MySQL and MongoDB build indexes." },
];

interface Turn {
  question: string;
  engine: Engine | null;
  result?: QueryResponse;
  answer?: string; // restored history has text only
  error?: string;
  pending?: boolean;
}

interface ChatMeta { id: string; title: string; at: number; }

const CHATS_KEY = "dbrain.chats.v1";
const ACTIVE_KEY = "dbrain.active.v1";

function load<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function save(key: string, value: unknown) {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* private mode: chats just won't persist */
  }
}

const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const seconds = (ms: number) => `${(ms / 1000).toFixed(2)} s`;

// "[Source: pg_vacuum (PostgreSQL)]" → inline code tagged as a citation, styled by <Markdown>.
// gpt-oss often uses full-width brackets: 【Source: …】.
const tagCitations = (md: string) =>
  md.replace(/[[【]Source:\s*([^\]】]+)[\]】]/g, (_m, s: string) =>
    "`cite:" + s.replace(/^Chunk \d+\s*\|\s*/, "").trim() + "`");

function Markdown({ text }: { text: string }) {
  return (
    <div className="answer">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          code({ children, className }) {
            const value = String(children);
            if (!className && value.startsWith("cite:")) {
              return <cite className="cite">{value.slice(5)}</cite>;
            }
            return <code className={className}>{children}</code>;
          },
          a({ children, href }) {
            return <a href={href} target="_blank" rel="noreferrer">{children}</a>;
          },
        }}
      >
        {tagCitations(text)}
      </ReactMarkdown>
    </div>
  );
}

interface PlanRow { depth: number; node: string; detail: string; time?: number; }

/** The answer's provenance, laid out like EXPLAIN ANALYZE output. */
function Plan({ r, chosen, status }: { r: QueryResponse; chosen: Engine | null; status: Status | null }) {
  const route = r.detected_db ? ENGINES[r.detected_db].label : "all engines";
  const how = chosen ? "chosen" : r.detected_db ? "detected" : "cross-engine";
  const minSim = status?.min_similarity ?? 0.35;
  const sim = r.top_chunk_score.toFixed(2);
  const retrieve: PlanRow = {
    depth: 0, node: "Hybrid search",
    detail: `${r.chunks_retrieved} passages, BM25 + vectors`, time: r.timings.retrieval_ms,
  };
  const routeRow: PlanRow = { depth: 0, node: "Route", detail: `${route} (${how}), best similarity ${sim}` };

  let summary: string;
  let rows: PlanRow[];
  if (r.guardrail_blocked) {
    summary = "Stopped by a safety check before any search";
    rows = [
      { depth: 0, node: "Refused", detail: "", time: r.latency_ms },
      { depth: 1, node: "Guardrail", detail: "request blocked; no search, no model call" },
    ];
  } else if (r.chunks_used === 0) {
    summary = `No answer: best match ${sim} is below the ${minSim} evidence bar`;
    rows = [
      { depth: 0, node: "Refused", detail: "", time: r.latency_ms },
      { depth: 1, node: "Evidence gate", detail: `best similarity ${sim} < ${minSim}` },
      { ...retrieve, depth: 2 },
      { ...routeRow, depth: 3 },
    ];
  } else {
    summary = r.refused
      ? `The model found no answer in ${r.chunks_used} passages (${seconds(r.latency_ms)})`
      : `Built from ${r.chunks_used} of ${r.chunks_retrieved} passages in ${seconds(r.latency_ms)}`;
    rows = [
      { depth: 0, node: r.refused ? "Refused" : "Answer", detail: "", time: r.latency_ms },
      { depth: 1, node: "Generate", detail: status?.llm ?? "LLM", time: r.timings.generation_ms },
      {
        depth: 2, node: "Rerank",
        detail: status?.reranker.enabled ? `cross-encoder, kept ${r.chunks_used}` : "skipped, fusion order kept",
        time: r.timings.rerank_ms,
      },
      { ...retrieve, depth: 3 },
      { ...routeRow, depth: 4 },
    ];
  }

  const color = r.detected_db ? ENGINES[r.detected_db].color : "var(--all)";
  return (
    <details className="plan">
      <summary>{summary}</summary>
      <div className="plan-body" style={{ ["--engine" as string]: color }}>
        {rows.map((row, i) => (
          <div className="plan-row" key={i}>
            <span className="plan-node">
              {"      ".repeat(Math.max(0, row.depth - 1))}
              {row.depth > 0 ? "  ->  " : ""}
              <b>{row.node}</b>
              {row.detail && <span className="plan-detail">{"  " + row.detail}</span>}
            </span>
            <span className="plan-time">{row.time !== undefined ? seconds(row.time) : ""}</span>
          </div>
        ))}
      </div>
    </details>
  );
}

function TurnView({ turn, sessionId, status }: { turn: Turn; sessionId: string | null; status: Status | null }) {
  const [note, setNote] = useState("");
  const r = turn.result;
  const text = r?.answer ?? turn.answer ?? "";

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setNote("Copied");
    } catch {
      setNote("Copy isn't available in this browser");
    }
  };

  const sendForApproval = async () => {
    if (!sessionId) return;
    try {
      await api.sendForApproval(sessionId, turn.question, text);
      setNote("Sent for approval");
    } catch (err) {
      setNote(errorMessage(err));
    }
  };

  return (
    <article className="turn">
      <h2 className="question">{turn.question}</h2>
      {turn.pending && <p className="pending" role="status">Searching the docs</p>}
      {turn.error && <p className="error" role="alert">{turn.error}</p>}

      {r?.refused ? (
        <div className={r.guardrail_blocked ? "refusal blocked" : "refusal"}>
          <p>
            <strong>{r.guardrail_blocked ? "Blocked. " : "Not in the docs. "}</strong>
            {r.answer}
          </p>
        </div>
      ) : (
        text && <Markdown text={text} />
      )}

      {r && r.citations.length > 0 && (
        <ul className="sources" aria-label="Sources">
          {r.citations.map((c) => (
            <li className="source" key={c.doc_name}>
              <span className="source-engine" style={{ color: ENGINES[c.db]?.color }}>{c.db_label}</span>
              {c.url ? <a href={c.url} target="_blank" rel="noreferrer">{c.doc_name}</a> : <span className="doc">{c.doc_name}</span>}
            </li>
          ))}
        </ul>
      )}

      {r && <Plan r={r} chosen={turn.engine} status={status} />}

      {text && !r?.refused && !turn.pending && (
        <div className="turn-actions">
          <button className="link-button" onClick={copy}>Copy answer</button>
          {sessionId && <button className="link-button" onClick={sendForApproval}>Send for approval</button>}
          {note && <span className="action-note" role="status">{note}</span>}
        </div>
      )}
    </article>
  );
}

export default function App() {
  const [chats, setChats] = useState<ChatMeta[]>(() => load(CHATS_KEY, []));
  const [sessionId, setSessionId] = useState<string | null>(() => load(ACTIVE_KEY, null));
  const [turns, setTurns] = useState<Turn[]>([]);
  const [engine, setEngine] = useState<Engine | null>(null);
  const [draft, setDraft] = useState("");
  const [status, setStatus] = useState<Status | null>(null);
  const [chatsOpen, setChatsOpen] = useState(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const lastTurnRef = useRef<HTMLDivElement>(null);
  const busy = turns.some((t) => t.pending);

  useEffect(() => { api.status().then(setStatus).catch(() => setStatus(null)); }, []);
  useEffect(() => save(CHATS_KEY, chats), [chats]);
  useEffect(() => save(ACTIVE_KEY, sessionId), [sessionId]);

  // Restore the open chat from the server on first load.
  useEffect(() => {
    if (sessionId) openChat(sessionId);
  }, []); // mount only

  useEffect(() => {
    lastTurnRef.current?.scrollIntoView({ block: "start", behavior: reducedMotion() ? "auto" : "smooth" });
  }, [turns.length]);

  async function openChat(id: string) {
    setChatsOpen(false);
    setSessionId(id);
    setTurns([]);
    try {
      const msgs = await api.messages(id);
      const restored: Turn[] = [];
      msgs.forEach((m) => {
        if (m.role === "user") restored.push({ question: m.content, engine: null });
        else if (restored.length) restored[restored.length - 1].answer = m.content;
      });
      setTurns(restored);
    } catch {
      // Expired or unknown on this server (e.g. after a redeploy): forget it.
      setChats((c) => c.filter((x) => x.id !== id));
      setSessionId(null);
    }
  }

  function newChat() {
    setSessionId(null);
    setTurns([]);
    setChatsOpen(false);
    inputRef.current?.focus();
  }

  async function ask(question: string, chosen: Engine | null) {
    const q = question.trim();
    if (!q || busy) return;
    setDraft("");
    if (inputRef.current) inputRef.current.style.height = "";
    const index = turns.length;
    setTurns((t) => [...t, { question: q, engine: chosen, pending: true }]);

    const update = (patch: Partial<Turn>) =>
      setTurns((t) => t.map((turn, i) => (i === index ? { ...turn, ...patch, pending: false } : turn)));

    try {
      const result = await api.ask(q, chosen, sessionId);
      update({ result });
      if (result.session_id !== sessionId) {
        setSessionId(result.session_id);
        setChats((c) => [{ id: result.session_id, title: q, at: Date.now() }, ...c].slice(0, 30));
      } else {
        setChats((c) => c.map((x) => (x.id === sessionId ? { ...x, at: Date.now() } : x)));
      }
    } catch (err) {
      update({ error: errorMessage(err) });
    }
  }

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      ask(draft, engine);
    }
  };

  const autosize = (el: HTMLTextAreaElement) => {
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  };

  const sortedChats = [...chats].sort((a, b) => b.at - a.at);

  return (
    <div className="shell">
      <aside className="sidebar">
        <div>
          <h1 className="brand">DBrain</h1>
          <p className="brand-sub">Database docs, answered with sources</p>
        </div>

        <button className="new-chat" onClick={newChat}>New chat</button>

        <nav className="chats" aria-label="Your chats" data-open={chatsOpen}>
          <h2 className="side-heading">Your chats</h2>
          <button
            className="link-button chats-toggle"
            aria-expanded={chatsOpen}
            onClick={() => setChatsOpen((o) => !o)}
          >
            Your chats ({chats.length})
          </button>
          {sortedChats.length === 0 ? (
            <p className="chat-empty chat-list">Your questions will be listed here.</p>
          ) : (
            <ul className="chat-list">
              {sortedChats.map((c) => (
                <li key={c.id}>
                  <button
                    className="chat-link"
                    aria-current={c.id === sessionId}
                    title={c.title}
                    onClick={() => openChat(c.id)}
                  >
                    {c.title}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </nav>

        {status && status.index.total > 0 && (
          <p className="index-note">
            Searches {status.index.total.toLocaleString()} passages from the official PostgreSQL,
            MySQL and MongoDB manuals.
          </p>
        )}
      </aside>

      <main className="main">
        <div className="thread">
          {turns.length === 0 ? (
            <section aria-labelledby="intro">
              <h2 id="intro" className="intro-title">Ask the PostgreSQL, MySQL and MongoDB manuals.</h2>
              <p className="intro-body">
                Every answer quotes the official documentation and names its sources. When the docs
                don't cover a question, DBrain says so instead of guessing.
              </p>
              <ul className="starters" aria-label="Example questions">
                {STARTERS.map(({ engine: e, q }) => (
                  <li key={q}>
                    <button className="starter" onClick={() => { setEngine(e); ask(q, e); }}>
                      <span className="starter-engine" style={{ color: e ? ENGINES[e].color : "var(--muted)" }}>
                        {e ? ENGINES[e].label : "All engines"}
                      </span>
                      <span className="starter-q">{q}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          ) : (
            turns.map((t, i) => (
              <div key={i} ref={i === turns.length - 1 ? lastTurnRef : undefined}>
                <TurnView turn={t} sessionId={sessionId} status={status} />
              </div>
            ))
          )}
        </div>

        <div className="composer-wrap">
          <form className="composer" onSubmit={(e) => { e.preventDefault(); ask(draft, engine); }}>
            <fieldset className="engines">
              <legend>Search in</legend>
              {([null, "postgresql", "mysql", "mongodb"] as (Engine | null)[]).map((e) => (
                <label className="engine-option" key={e ?? "all"}>
                  <input
                    type="radio"
                    name="engine"
                    checked={engine === e}
                    onChange={() => setEngine(e)}
                  />
                  {e ? ENGINES[e].label : "All engines"}
                </label>
              ))}
            </fieldset>
            <div className="ask-row">
              <label htmlFor="ask" className="visually-hidden">Your question</label>
              <textarea
                id="ask"
                ref={inputRef}
                className="ask-input"
                rows={1}
                maxLength={1000}
                value={draft}
                placeholder="Ask about indexes, locking, VACUUM, replication…"
                onChange={(e) => { setDraft(e.target.value); autosize(e.target); }}
                onKeyDown={onKeyDown}
              />
              <button className="ask-button" type="submit" disabled={busy || !draft.trim()}>
                {busy ? "Asking" : "Ask"}
              </button>
            </div>
            <p className="composer-hint">Enter to ask, Shift+Enter for a new line.</p>
          </form>
        </div>
      </main>
    </div>
  );
}
