import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { runQuery, runSymptomQuery } from "../api/client.js";

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

const EXAMPLE_SYMPTOMS = [
  "Persistent dry cough, fever, and shortness of breath for 5 days",
  "Joint pain and swelling in both hands, worse in the morning",
  "Progressive memory loss and confusion over several months",
  "Sudden high fever, rash, and fatigue with no clear cause",
];

export default function QueryPage() {
  const navigate = useNavigate();
  const [mode, setMode] = useState("disease"); // "disease" | "symptoms"

  const [query, setQuery] = useState("");
  const [symptoms, setSymptoms] = useState("");
  const [topConditions, setTopConditions] = useState(2);

  // Thorough-by-default: more PubMed evidence per query means more
  // candidates survive the anti-circularity filter (see validation.py) —
  // a shallow fetch was the actual cause of "only 1 candidate found"
  // results. This is a real research tool, not a speed demo; if you want
  // a faster low-stakes test run, lower these per-query in the form.
  const [retmax, setRetmax] = useState(40);
  const [topK, setTopK] = useState(8);
  const [provider, setProvider] = useState("");
  const [model, setModel] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [progressLabel, setProgressLabel] = useState("");

  const inputText = mode === "disease" ? query : symptoms;

  async function handleSubmit(e) {
    e.preventDefault();
    if (!inputText.trim()) return;

    setLoading(true);
    setError(null);
    setProgressLabel(
      mode === "symptoms" ? "Interpreting symptoms…" : "Retrieving PubMed evidence…"
    );

    const labelTimers =
      mode === "symptoms"
        ? [
            setTimeout(() => setProgressLabel("Grounding candidate conditions in PubMed…"), 3000),
            setTimeout(() => setProgressLabel("Running repurposing pipeline per condition…"), 7000),
            setTimeout(() => setProgressLabel("Validating against knowledge graph…"), 14000),
            setTimeout(() => setProgressLabel("Master agent fusing evidence…"), 20000),
          ]
        : [
            setTimeout(() => setProgressLabel("Running NER + semantic reranking…"), 2500),
            setTimeout(() => setProgressLabel("Reasoning agent generating candidates…"), 6000),
            setTimeout(() => setProgressLabel("Validating against knowledge graph…"), 12000),
            setTimeout(() => setProgressLabel("Master agent fusing evidence…"), 18000),
          ];

    try {
      let res;
      if (mode === "symptoms") {
        const payload = {
          symptoms: symptoms.trim(),
          top_conditions: Number(topConditions),
          retmax: Number(retmax),
          top_k: Number(topK),
          save_to_history: true,
        };
        if (provider) payload.provider = provider;
        if (model) payload.model = model;
        res = await runSymptomQuery(payload);
      } else {
        const payload = {
          query: query.trim(),
          retmax: Number(retmax),
          top_k: Number(topK),
          save_to_history: true,
        };
        if (provider) payload.provider = provider;
        if (model) payload.model = model;
        res = await runQuery(payload);
      }

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
            Enter any disease or condition — not limited to the examples below — or switch to symptom
            mode. The system retrieves live PubMed evidence, extracts entities, reasons over
            candidates, validates against a real-time knowledge graph (Open Targets), and fuses
            everything into ranked, explainable recommendations.
          </p>
        </div>
      </div>

      {error && <div className="banner banner-error">{error}</div>}

      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        <button
          type="button"
          className={mode === "disease" ? "btn btn-primary" : "btn btn-ghost"}
          onClick={() => setMode("disease")}
        >
          Disease / Condition
        </button>
        <button
          type="button"
          className={mode === "symptoms" ? "btn btn-primary" : "btn btn-ghost"}
          onClick={() => setMode("symptoms")}
        >
          Symptoms
        </button>
      </div>

      <form onSubmit={handleSubmit} className="card" style={{ maxWidth: 760 }}>
        {mode === "symptoms" ? (
          <>
            <div className="banner" style={{ marginBottom: 14, fontSize: 13 }}>
              Research hypothesis only — not a medical diagnosis, and results are NOT symptom relief
              recommendations. Describe symptoms in plain language; the AI proposes candidate conditions
              (grounded in PubMed where possible) and runs the drug-repurposing pipeline against the top
              matches — surfacing non-obvious existing drugs for that suspected condition, not standard
              first-line treatment. Always consult a qualified clinician for actual diagnosis or treatment.
            </div>

            <label className="card-title" htmlFor="symptom-input">
              Symptoms
            </label>
            <textarea
              id="symptom-input"
              className="input"
              rows={3}
              placeholder="e.g. Persistent dry cough, fever, and shortness of breath for 5 days"
              value={symptoms}
              onChange={(e) => setSymptoms(e.target.value)}
              style={{ resize: "vertical", fontFamily: "inherit", marginBottom: 14 }}
            />

            <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 20 }}>
              {EXAMPLE_SYMPTOMS.map((ex) => (
                <button
                  type="button"
                  key={ex}
                  className="btn btn-ghost"
                  style={{ fontSize: 12, padding: "6px 12px" }}
                  onClick={() => setSymptoms(ex)}
                >
                  {ex}
                </button>
              ))}
            </div>
          </>
        ) : (
          <>
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
          </>
        )}

        <div className="grid grid-4" style={{ marginBottom: 20 }}>
          {mode === "symptoms" && (
            <div>
              <label className="card-title" style={{ marginBottom: 6 }}>Conditions to explore</label>
              <input
                type="number"
                className="input"
                value={topConditions}
                min={1}
                max={5}
                onChange={(e) => setTopConditions(e.target.value)}
              />
            </div>
          )}
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

        <button type="submit" className="btn btn-primary" disabled={loading || !inputText.trim()}>
          {loading ? <span className="spinner" /> : "▶"}
          {loading ? progressLabel : "Run Pipeline"}
        </button>
      </form>

      {loading && (
        <div className="card" style={{ maxWidth: 760, marginTop: 16 }}>
          <div className="card-title">Pipeline Stages</div>
          <ol style={{ margin: 0, paddingLeft: 18, color: "var(--text-secondary)", fontSize: 13, lineHeight: 2 }}>
            {mode === "symptoms" && (
              <li>Symptom Agent — LLM interprets symptoms into candidate conditions, grounded in live PubMed evidence</li>
            )}
            <li>Planner Agent — query decomposition</li>
            <li>Retrieval Agent — PubMed E-Utilities + semantic reranking</li>
            <li>NER Agent — biomedical entity extraction</li>
            <li>Reasoning Agent — LLM / heuristic candidate generation</li>
            <li>Validation Agent — plausibility, evidence, safety checks</li>
            <li>Knowledge Graph — drug-target-disease path validation (Open Targets, live)</li>
            <li>Master Agent — evidence fusion, conflict detection, final ranking</li>
          </ol>
        </div>
      )}
    </div>
  );
}
