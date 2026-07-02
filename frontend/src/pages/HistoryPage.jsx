import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { deleteHistoryItem, getHistory } from "../api/client.js";
import LoadingState from "../components/LoadingState.jsx";

export default function HistoryPage() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  async function load() {
    setLoading(true);
    try {
      const h = await getHistory(200);
      setHistory(h);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handleDelete(queryId) {
    try {
      await deleteHistoryItem(queryId);
      setHistory((prev) => prev.filter((h) => h.query_id !== queryId));
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Query History</h1>
          <p className="page-subtitle">All past pipeline runs, saved locally.</p>
        </div>
        <Link to="/query" className="btn btn-primary">+ New Query</Link>
      </div>

      {error && <div className="banner banner-error">{error}</div>}

      <div className="card">
        {loading ? (
          <LoadingState label="Loading history…" />
        ) : history.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state-icon">🕘</div>
            <div>No history yet.</div>
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
                  <td style={{ display: "flex", gap: 6, justifyContent: "flex-end" }}>
                    <Link to={`/results/${h.query_id}`} className="btn btn-ghost" style={{ fontSize: 12, padding: "5px 12px" }}>
                      View →
                    </Link>
                    <button
                      className="btn btn-ghost"
                      style={{ fontSize: 12, padding: "5px 12px", color: "var(--danger)" }}
                      onClick={() => handleDelete(h.query_id)}
                    >
                      Delete
                    </button>
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
