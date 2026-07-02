const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

export function health() {
  return request("/api/health");
}

export function runQuery(payload) {
  return request("/api/query", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getHistory(limit = 50) {
  return request(`/api/history?limit=${limit}`);
}

export function getHistoryItem(queryId) {
  return request(`/api/history/${queryId}`);
}

export function deleteHistoryItem(queryId) {
  return request(`/api/history/${queryId}`, { method: "DELETE" });
}

export function getKgStats() {
  return request("/api/knowledge-graph/stats");
}

export function getKgSubgraph(disease, maxHops = 2) {
  return request(`/api/knowledge-graph/${encodeURIComponent(disease)}?max_hops=${maxHops}`);
}

export function getMetrics() {
  return request("/api/metrics");
}

export function getComparativeEvaluation() {
  return request("/api/comparative-evaluation");
}

export function getLinkPredictions(disease, topK = 10) {
  return request(`/api/link-prediction/${encodeURIComponent(disease)}?top_k=${topK}`);
}

export function getLinkPredictionStatus() {
  return request("/api/link-prediction-status");
}

export function trainEmbeddings(epochs = 150, dim = 64) {
  return request(`/api/train-embeddings?epochs=${epochs}&dim=${dim}`, { method: "POST" });
}

export function getCaseStudies() {
  return request("/api/case-studies");
}

export function getPredictBenchmark() {
  return request("/api/predict-benchmark");
}
