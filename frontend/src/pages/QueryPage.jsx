import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { runQuery } from "../api/client.js";

const EXAMPLE_QUERIES = [
  "Find repurposed drug candidates for Alzheimer disease",
  "Find promising repurposed drugs for Parkinson disease",
  "Drugs for Type 2 diabetes",
  "Find repurposed drugs for COVID-19",
  "Drug repurposing candidates for breast cancer and lung cancer",
  "Find drugs for Influenza",
  "Repurposed drugs for depression",
  "Find drug candidates for rheumatoid arthritis",
];

export default function QueryPage() {
  const navigate = useNavigate();
  const [query, setQuery] = useState("");
  const [retmax, setRetmax] = useState(40);
  const [topK, setTopK] = useState(8);
  const [provider, setProvider] = useState("");
  const [model, setModel] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [progressLabel, setProgressLabel] = useState("");

  async function handleSubmit(e) {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setProgressLabel("Retrieving PubMed evidence…");

    const labelTimers = [
      setTimeout(() => setProgressLabel("Running NER + semantic reranking…"), 2500),
      setTimeout(() => setProgressLabel("Reasoning agent generating candidates…"), 6000),
      setTimeout(() => setProgressLabel("Validating against knowledge graph…"), 12000),
      setTimeout(() => setProgressLabel("Master agent fusing evidence…"), 18000),
    ];

    try {
      const payload = {
        query: query.trim(),
        retmax: Number(retmax),
        top_k: Number(topK),
        save_to_history: true,
      };
      if (provider) payload.provider = provider;
      if (model) payload.model = model;

      const res = await runQuery(payload);
      labelTimers.forEach(clearTimeout);

      if (res.query_id) {
        navigate(`/results/${res.query_id}`);
      } else {
        sessionStorage.setItem("drugenix_last_result", JSON.stringify(res.result));
        navigate("/results/last");
      }
    } catch (err) {
      labelTimers.forEach(clearTimeout);
      setError(err.message || "Pipeline execution failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">New Query</h1>
          <p className="page-subtitle">
            Enter a disease or condition. The system retrieves PubMed evidence, extracts entities,
            reasons over candidates, validates against a knowledge graph, and fuses everything into
            ranked, explainable recommendations.
          </p>
        </div>
      </div>

      {error && <div className="banner banner-error">{error}</div>}

      <form onSubmit={handleSubmit} className="card" style={{ maxWidth: 760 }}>
        <label className="card-title" htmlFor="query-input">
          Disease / Condition
        </label>
        <textarea
          id="query-input"
          className="input"
          rows={3}
          placeholder="e.g. Find repurposed drug candidates for Alzheimer disease"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          style={{ resize: "vertical", fontFamily: "inherit", marginBottom: 14 }}
        />

        <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 20 }}>
          {EXAMPLE_QUERIES.map((ex) => (
            <button
              type="button"
              key={ex}
              className="btn btn-ghost"
              style={{ fontSize: 12, padding: "6px 12px" }}
              onClick={() => setQuery(ex)}
            >
              {ex}
            </button>
          ))}
        </div>

        <div className="grid grid-4" style={{ marginBottom: 20 }}>
          <div>
            <label className="card-title" style={{ marginBottom: 6 }}>Retmax</label>
            <input
              type="number"
              className="input"
              value={retmax}
              min={5}
              max={200}
              onChange={(e) => setRetmax(e.target.value)}
            />
          </div>
          <div>
            <label className="card-title" style={{ marginBottom: 6 }}>Top-K</label>
            <input
              type="number"
              className="input"
              value={topK}
              min={3}
              max={30}
              onChange={(e) => setTopK(e.target.value)}
            />
          </div>
          <div>
            <label className="card-title" style={{ marginBottom: 6 }}>LLM Provider</label>
            <select className="select" value={provider} onChange={(e) => setProvider(e.target.value)} style={{ width: "100%" }}>
              <option value="">Auto / heuristic</option>
              <option value="openai">OpenAI</option>
              <option value="gemini">Gemini</option>
              <option value="ollama">Ollama (local)</option>
            </select>
          </div>
          <div>
            <label className="card-title" style={{ marginBottom: 6 }}>Model</label>
            <input
              type="text"
              className="input"
              placeholder="e.g. llama3.2:latest"
              value={model}
              onChange={(e) => setModel(e.target.value)}
            />
          </div>
        </div>

        <button type="submit" className="btn btn-primary" disabled={loading || !query.trim()}>
          {loading ? <span className="spinner" /> : "▶"}
          {loading ? progressLabel : "Run Pipeline"}
        </button>
      </form>

      {loading && (
        <div className="card" style={{ maxWidth: 760, marginTop: 16 }}>
          <div className="card-title">Pipeline Stages</div>
          <ol style={{ margin: 0, paddingLeft: 18, color: "var(--text-secondary)", fontSize: 13, lineHeight: 2 }}>
            <li>Planner Agent — query decomposition</li>
            <li>Retrieval Agent — PubMed E-Utilities + semantic reranking</li>
            <li>NER Agent — biomedical entity extraction</li>
            <li>Reasoning Agent — LLM / heuristic candidate generation</li>
            <li>Validation Agent — plausibility, evidence, safety checks</li>
            <li>Knowledge Graph — drug-target-disease path validation</li>
            <li>Master Agent — evidence fusion, conflict detection, final ranking</li>
          </ol>
        </div>
      )}
    </div>
  );
}
