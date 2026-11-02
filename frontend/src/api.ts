import axios from "axios";

// Empty in production: the API is served from the same origin as the UI.
const API_URL = process.env.REACT_APP_API_URL ?? "http://localhost:8000";

export type Engine = "postgresql" | "mysql" | "mongodb";

export interface Citation { doc_name: string; db: Engine; db_label: string; url: string; }

export interface QueryResponse {
  query: string;
  answer: string;
  citations: Citation[];
  detected_db: Engine | null;
  chunks_retrieved: number;
  chunks_used: number;
  refused: boolean;
  guardrail_blocked: boolean;
  top_chunk_score: number;
  timings: { retrieval_ms: number; rerank_ms: number; generation_ms: number };
  latency_ms: number;
  session_id: string;
}

export interface Status {
  index: { total: number } & Partial<Record<Engine, number>>;
  llm: string;
  reranker: { enabled: boolean; model: string };
  min_similarity: number;
}

const http = axios.create({ baseURL: API_URL, timeout: 90_000 });

export function errorMessage(err: unknown): string {
  if (axios.isAxiosError(err)) {
    const detail = err.response?.data?.detail;
    if (typeof detail === "string") return detail;
    if (err.response?.status === 422) return "That question couldn't be read. Keep it under 1,000 characters.";
    if (err.code === "ECONNABORTED") return "The answer took too long. Try again, or ask a narrower question.";
    if (!err.response) return "Can't reach the DBrain server. Check that the backend is running.";
  }
  return "Something went wrong. Try again.";
}

export const api = {
  ask: (query: string, db_filter: Engine | null, session_id: string | null) =>
    http.post<QueryResponse>("/api/v1/query", {
      query, db_filter, session_id: session_id ?? undefined,
    }).then((r) => r.data),

  messages: (sessionId: string) =>
    http.get<{ messages: { role: "user" | "assistant"; content: string }[] }>(
      `/api/v1/sessions/${sessionId}/messages`,
    ).then((r) => r.data.messages),

  status: () => http.get<Status>("/api/v1/status").then((r) => r.data),

  sendForApproval: (session_id: string, query: string, action: string) =>
    http.post("/api/v1/actions", { session_id, query, action, risk_level: "medium" }),
};
