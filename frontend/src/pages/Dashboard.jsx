import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { getHistory, getKgStats, health } from "../api/client.js";
import LoadingState from "../components/LoadingState.jsx";

export default function Dashboard() {
  const [history, setHistory] = useState([]);
  const [kgStats, setKgStats] = useState(null);
  const [apiStatus, setApiStatus] = useState("checking");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        await health();
        setApiStatus("online");
      } catch {
        setApiStatus("offline");
      }
      try {
        const [h, kg] = await Promise.all([getHistory(8), getKgStats()]);
        setHistory(h);
        setKgStats(kg);
      } catch {
        // non-fatal
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Dashboard</h1>
          <p className="page-subtitle">
            Agentic AI System for Accelerated Drug Repurposing — Planner, Retrieval, NER, Reasoning,
            Validation, Knowledge Graph, and Aggregation agents working together.
          </p>
        </div>
        <Link to="/query" className="btn btn-primary">+ New Query</Link>
      </div>

      <div className="grid grid-4" style={{ marginBottom: 24 }}>
        <div className="stat-card">
          <div className="stat-value" style={{ color: apiStatus === "online" ? "var(--success)" : "var(--danger)" }}>
            {apiStatus === "checking" ? "…" : apiStatus === "online" ? "Online" : "Offline"}
          </div>
          <div className="stat-label">API Status</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{kgStats?.total_nodes ?? "–"}</div>
          <div className="stat-label">Knowledge Graph Nodes</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{kgStats?.drugs ?? "–"}</div>
          <div className="stat-label">Drugs in KG</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{history.length}</div>
          <div className="stat-label">Recent queries</div>
        </div>
      </div>

      <div className="grid grid-2" style={{ marginBottom: 24 }}>
        <div className="card">
          <div className="card-title">Pipeline Architecture</div>
          <AgentPipeline />
        </div>
        <div className="card">
          <div className="card-title">Knowledge Graph Composition</div>
          {kgStats ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 12, marginTop: 4 }}>
              <KgBar label="Drugs" value={kgStats.drugs} total={kgStats.total_nodes} color="var(--accent)" />
              <KgBar label="Targets" value={kgStats.targets} total={kgStats.total_nodes} color="var(--success)" />
              <KgBar label="Diseases" value={kgStats.diseases} total={kgStats.total_nodes} color="var(--warn)" />
              <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 6 }}>
                {kgStats.total_edges} edges (drug→target, target→disease, drug→disease)
              </div>
            </div>
          ) : (
            <LoadingState label="Loading…" />
          )}
        </div>
      </div>

      <div className="card">
        <div className="card-title">Recent Queries</div>
        {loading ? (
          <LoadingState label="Loading history…" />
        ) : history.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state-icon">⌕</div>
            <div>No queries yet. Start with your first disease query.</div>
            <Link to="/query" className="btn btn-primary" style={{ marginTop: 14 }}>Run a Query</Link>
          </div>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>Query</th>
                <th>Candidates</th>
                <th>Recommended</th>
                <th>Date</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {history.map((h) => (
                <tr key={h.query_id}>
                  <td style={{ fontWeight: 500 }}>{h.query}</td>
                  <td>{h.candidate_count}</td>
                  <td style={{ color: "var(--success)" }}>{h.recommended_count}</td>
                  <td style={{ color: "var(--text-muted)" }}>{new Date(h.timestamp).toLocaleString()}</td>
                  <td>
                    <Link to={`/results/${h.query_id}`} className="btn btn-ghost" style={{ fontSize: 12, padding: "5px 12px" }}>
                      View →
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

function KgBar({ label, value, total, color }) {
  const pct = total ? Math.round((value / total) * 100) : 0;
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12.5, marginBottom: 4 }}>
        <span>{label}</span>
        <span style={{ color: "var(--text-muted)" }}>{value}</span>
      </div>
      <div className="progress-bar">
        <div className="progress-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

const AGENTS = [
  { name: "Planner Agent", desc: "Decomposes multi-disease queries" },
  { name: "Retrieval Agent", desc: "PubMed + semantic vector search" },
  { name: "NER Agent", desc: "Biomedical entity extraction" },
  { name: "Reasoning Agent", desc: "LLM / heuristic candidate generation" },
  { name: "Validation Agent", desc: "Plausibility, evidence, safety checks" },
  { name: "Knowledge Graph", desc: "Drug-target-disease path validation" },
  { name: "Master Agent", desc: "Evidence fusion + final ranking" },
];

function AgentPipeline() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
      {AGENTS.map((a, i) => (
        <div key={a.name} style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
            <div
              style={{
                width: 22,
                height: 22,
                borderRadius: "50%",
                background: "var(--accent-dim)",
                color: "var(--accent-hover)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 11,
                fontWeight: 700,
                flexShrink: 0,
              }}
            >
              {i + 1}
            </div>
            {i < AGENTS.length - 1 && <div style={{ width: 2, flex: 1, background: "var(--border)", minHeight: 18 }} />}
          </div>
          <div style={{ paddingBottom: 14 }}>
            <div style={{ fontWeight: 600, fontSize: 13.5 }}>{a.name}</div>
            <div style={{ fontSize: 12, color: "var(--text-muted)" }}>{a.desc}</div>
          </div>
        </div>
      ))}
    </div>
  );
}
