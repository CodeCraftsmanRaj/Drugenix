import { useEffect, useState } from "react";
import {
  BarChart, Bar,
  RadarChart, PolarGrid, PolarAngleAxis, Radar,
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from "recharts";
import {
  getMetrics, getComparativeEvaluation,
  getCaseStudies, getPredictBenchmark,
  getLinkPredictionStatus, getLinkPredictions, trainEmbeddings,
  runBenchmark,
} from "../api/client.js";
import LoadingState from "../components/LoadingState.jsx";

const Ks = [1, 3, 5, 10];

const tooltipStyle = {
  background: "#ffffff",
  border: "1px solid #e2e8f0",
  borderRadius: 8,
  fontSize: 12.5,
  color: "#0f172a",
  boxShadow: "0 4px 16px rgba(15,23,42,0.08)",
};

const VARIANT_LABELS = {
  pure_llm: "Pure LLM",
  pure_graph: "Pure Graph",
  hybrid_recommended_only: "Hybrid (Recommended)",
  hybrid_all_candidates: "Hybrid (All)",
};
const VARIANT_COLORS = {
  pure_llm: "#f5b84e",
  pure_graph: "#34c77b",
  hybrid_recommended_only: "#4f8cff",
  hybrid_all_candidates: "#a78bfa",
};

// ── Case studies — known successful repurposings ──────────────────────────
const CASE_STUDIES = [
  {
    id: "metformin_cancer",
    drug: "Metformin",
    from: "Type 2 Diabetes",
    to: "Cancer (Breast, Colorectal)",
    mechanism: "AMPK activation → mTOR inhibition → reduced cancer cell proliferation",
    evidence: "50+ RCTs and observational studies; meta-analyses show 30% cancer risk reduction",
    status: "Phase III trials ongoing",
    color: "#2563eb",
  },
  {
    id: "thalidomide_myeloma",
    drug: "Thalidomide",
    from: "Morning sickness (withdrawn)",
    to: "Multiple Myeloma",
    mechanism: "Anti-angiogenic + immunomodulatory via cereblon E3 ligase complex",
    evidence: "FDA approved for MM in 2006; MRC Myeloma VII trial",
    status: "FDA Approved (2006)",
    color: "#34c77b",
  },
  {
    id: "sildenafil_pah",
    drug: "Sildenafil",
    from: "Hypertension / Angina",
    to: "Pulmonary Arterial Hypertension + Erectile Dysfunction",
    mechanism: "PDE5 inhibition → elevated cGMP → smooth muscle relaxation in pulmonary vessels",
    evidence: "SUPER-1 trial; FDA approved PAH indication 2005",
    status: "FDA Approved (2005)",
    color: "#34c77b",
  },
  {
    drug: "Ketamine",
    from: "General Anaesthetic",
    to: "Treatment-Resistant Depression",
    mechanism: "NMDA receptor antagonism → rapid glutamate surge → BDNF-mediated synaptogenesis",
    evidence: "Esketamine (Spravato) approved 2019; response within hours vs weeks for SSRIs",
    status: "FDA Approved (2019)",
    color: "#34c77b",
  },
  {
    drug: "Dexamethasone",
    from: "Anti-inflammatory / Immunosuppressant",
    to: "COVID-19 (severe)",
    mechanism: "Glucocorticoid suppression of cytokine storm → reduced ARDS mortality",
    evidence: "RECOVERY trial: 35% mortality reduction in ventilated patients",
    status: "WHO recommended (2020)",
    color: "#f5b84e",
  },
  {
    drug: "Rapamycin",
    from: "Organ Transplant Rejection",
    to: "Alzheimer disease / Cancer / Ageing",
    mechanism: "mTOR inhibition → autophagy induction → Aβ/tau clearance; anti-proliferative",
    evidence: "ITP ageing study; multiple AD trials; NCI cancer studies",
    status: "Phase II/III trials",
    color: "#a78bfa",
  },
];

function ChartCard({ title, children, height = 240 }) {
  return (
    <div className="card">
      <div className="card-title">{title}</div>
      <ResponsiveContainer width="100%" height={height}>
        {children}
      </ResponsiveContainer>
    </div>
  );
}

function ComparisonBar({ data }) {
  const max = Math.max(...data.map((d) => d.value), 0.001);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {data.map((row) => {
        const pct = (row.value / max) * 100;
        return (
          <div key={row.name}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12.5, marginBottom: 4 }}>
              <span style={{ fontWeight: 600 }}>{row.name}</span>
              <span style={{ color: "var(--text-muted)", fontVariantNumeric: "tabular-nums" }}>
                {(row.value * 100).toFixed(1)}%
              </span>
            </div>
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${pct}%`, background: row.fill }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function MetricBadge({ label, value, unit = "%", highlight = false }) {
  return (
    <div className="stat-card" style={{ textAlign: "center" }}>
      <div style={{
        fontSize: 28, fontWeight: 800,
        color: highlight ? "var(--primary)" : "var(--text-primary)",
      }}>
        {typeof value === "number" ? (value * 100).toFixed(1) : "—"}{unit}
      </div>
      <div className="stat-label" style={{ marginTop: 4 }}>{label}</div>
    </div>
  );
}

function RunBenchmarkBanner({ onRun, running, error }) {
  return (
    <div className="card" style={{ marginBottom: 20, background: "var(--bg-elevated)", borderLeft: "4px solid var(--primary)" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
        <div>
          <div style={{ fontWeight: 700, marginBottom: 4 }}>Benchmark data may be stale</div>
          <div style={{ fontSize: 13, color: "var(--text-secondary)" }}>
            The numbers below are from the last benchmark run, not live from your current queries. Click
            "Run Benchmark" to re-evaluate the pipeline against the gold-standard drug-disease dataset
            (FDA-approved + known repurposed drugs) and refresh these numbers. Runs 3 variants across 5
            diseases. Takes ~10 min with Ollama.
          </div>
        </div>
        <button
          className="btn btn-primary"
          onClick={onRun}
          disabled={running}
          style={{ whiteSpace: "nowrap" }}
        >
          {running ? <><span className="spinner" /> Running…</> : "▶ Run Benchmark"}
        </button>
      </div>
      {running && (
        <div className="banner banner-info" style={{ marginTop: 12 }}>
          Benchmark started in background. This page will auto-refresh every 30s.
          Make sure Ollama is running (ollama serve).
        </div>
      )}
      {error && (
        <div className="banner banner-error" style={{ marginTop: 12 }}>
          Could not start the benchmark: {error}
        </div>
      )}
    </div>
  );
}

function ComparativeSection({ data }) {
  const MAIN_VARIANTS = ["pure_llm", "pure_graph", "hybrid_recommended_only"];

  const buildRows = (fn) =>
    MAIN_VARIANTS.map((v) => ({
      name: VARIANT_LABELS[v] || v,
      value: fn(data[v]),
      fill: VARIANT_COLORS[v],
    }));

  const hitData = buildRows((v) => v?.hit_at_k?.["Hit@5"] ?? 0);
  const precData = buildRows((v) => v?.precision_at_k?.["P@5"] ?? 0);
  const mrrData = buildRows((v) => v?.mrr ?? 0);
  const aurData = buildRows((v) => v?.auroc ?? 0);
  const groundData = buildRows((v) => v?.mean_groundedness ?? 0);
  const hallucData = buildRows((v) => v?.hallucination_rate ?? 0);

  const groupedHit = ["Hit@1", "Hit@3", "Hit@5", "Hit@10"].map((k) => ({
    k,
    "Pure LLM": data.pure_llm?.hit_at_k?.[k] ?? 0,
    "Pure Graph": data.pure_graph?.hit_at_k?.[k] ?? 0,
    "Hybrid": data.hybrid_recommended_only?.hit_at_k?.[k] ?? 0,
  }));

  const radarData = ["Hit@5", "P@5", "MRR", "AUROC", "Groundedness"].map((metric) => {
    const entry = { metric };
    const get = (v, m) => {
      if (m === "Hit@5") return (v?.hit_at_k?.["Hit@5"] ?? 0) * 100;
      if (m === "P@5") return (v?.precision_at_k?.["P@5"] ?? 0) * 100;
      if (m === "MRR") return (v?.mrr ?? 0) * 100;
      if (m === "AUROC") return (v?.auroc ?? 0) * 100;
      if (m === "Groundedness") return (v?.mean_groundedness ?? 0) * 100;
      return 0;
    };
    MAIN_VARIANTS.forEach((vk) => {
      entry[VARIANT_LABELS[vk]] = get(data[vk], metric);
    });
    return entry;
  });

  const diseases = data.diseases_evaluated || [];
  const hybrid = data.hybrid_recommended_only || {};

  return (
    <>
      <h2 style={{ fontSize: 16, fontWeight: 700, margin: "0 0 8px" }}>
        Comparative Evaluation — Pure LLM vs Pure Graph vs Hybrid
      </h2>
      <p style={{ fontSize: 13, color: "var(--text-secondary)", margin: "0 0 20px" }}>
        Evaluated on {diseases.length} diseases: {diseases.slice(0, 6).join(", ")}{diseases.length > 6 ? ` +${diseases.length - 6} more` : ""}.
        Gold standard: FDA-approved + known repurposed drugs.
      </p>

      {/* Key metric badges for Hybrid (Recommended) */}
      <div className="grid grid-4" style={{ marginBottom: 20 }}>
        <MetricBadge label="Hit@5 (Hybrid)" value={hybrid.hit_at_k?.["Hit@5"]} highlight />
        <MetricBadge label="Precision@5 (Hybrid)" value={hybrid.precision_at_k?.["P@5"]} highlight />
        <MetricBadge label="MRR (Hybrid)" value={hybrid.mrr} highlight />
        <MetricBadge label="AUROC (Hybrid)" value={hybrid.auroc} highlight />
      </div>

      {/* Hit@k grouped bar chart */}
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="card-title">Hit@k — Does a gold-standard drug appear in top-k results?</div>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={groupedHit} barCategoryGap="25%">
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="k" stroke="#64748b" fontSize={12} />
            <YAxis domain={[0, 1]} stroke="#64748b" fontSize={12} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
            <Tooltip contentStyle={tooltipStyle} formatter={(v) => `${(v * 100).toFixed(1)}%`} />
            <Legend wrapperStyle={{ fontSize: 12.5 }} />
            <Bar dataKey="Pure LLM" fill={VARIANT_COLORS.pure_llm} radius={[4, 4, 0, 0]} />
            <Bar dataKey="Pure Graph" fill={VARIANT_COLORS.pure_graph} radius={[4, 4, 0, 0]} />
            <Bar dataKey="Hybrid" fill={VARIANT_COLORS.hybrid_recommended_only} radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Radar chart + metric bars */}
      <div className="grid grid-2" style={{ marginBottom: 16 }}>
        <ChartCard title="Multi-metric Radar — All Variants" height={280}>
          <RadarChart data={radarData}>
            <PolarGrid stroke="#e2e8f0" />
            <PolarAngleAxis dataKey="metric" tick={{ fill: "#64748b", fontSize: 11 }} />
            {MAIN_VARIANTS.map((v) => (
              <Radar
                key={v}
                name={VARIANT_LABELS[v]}
                dataKey={VARIANT_LABELS[v]}
                stroke={VARIANT_COLORS[v]}
                fill={VARIANT_COLORS[v]}
                fillOpacity={0.15}
              />
            ))}
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Tooltip contentStyle={tooltipStyle} formatter={(v) => `${v.toFixed(1)}%`} />
          </RadarChart>
        </ChartCard>

        <div className="card">
          <div className="card-title" style={{ marginBottom: 14 }}>AUROC — Ranking Quality</div>
          <ComparisonBar data={aurData} />
          <div className="card-title" style={{ marginTop: 18, marginBottom: 14 }}>MRR — First Correct Hit</div>
          <ComparisonBar data={mrrData} />
        </div>
      </div>

      <div className="grid grid-2" style={{ marginBottom: 16 }}>
        <div className="card">
          <div className="card-title" style={{ marginBottom: 14 }}>Hit@5</div>
          <ComparisonBar data={hitData} />
          <div className="card-title" style={{ marginTop: 18, marginBottom: 14 }}>Precision@5</div>
          <ComparisonBar data={precData} />
        </div>
        <div className="card">
          <div className="card-title" style={{ marginBottom: 14 }}>Mean Groundedness</div>
          <ComparisonBar data={groundData} />
          <div className="card-title" style={{ marginTop: 18, marginBottom: 14 }}>Hallucination Rate (lower = better)</div>
          <ComparisonBar data={hallucData} />
        </div>
      </div>

      {/* Full results table */}
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="card-title">Full Results Table</div>
        <table className="table">
          <thead>
            <tr>
              <th>Variant</th>
              <th>Hit@1</th><th>Hit@5</th>
              <th>P@5</th>
              <th>MRR</th>
              <th>AUROC</th>
              <th>Groundedness</th>
              <th>Halluc. Rate</th>
              <th>N candidates</th>
            </tr>
          </thead>
          <tbody>
            {["pure_llm", "pure_graph", "hybrid_recommended_only", "hybrid_all_candidates"].map((v) => {
              const r = data[v];
              if (!r) return null;
              const isHybrid = v === "hybrid_recommended_only";
              return (
                <tr key={v} style={isHybrid ? { background: "rgba(79,140,255,0.06)" } : {}}>
                  <td style={{ fontWeight: isHybrid ? 700 : 400 }}>
                    {VARIANT_LABELS[v]} {isHybrid && "★"}
                  </td>
                  <td>{((r.hit_at_k?.["Hit@1"] ?? 0) * 100).toFixed(0)}%</td>
                  <td style={{ fontWeight: 600 }}>{((r.hit_at_k?.["Hit@5"] ?? 0) * 100).toFixed(0)}%</td>
                  <td>{((r.precision_at_k?.["P@5"] ?? 0) * 100).toFixed(0)}%</td>
                  <td>{((r.mrr ?? 0) * 100).toFixed(1)}%</td>
                  <td style={{ color: (r.auroc ?? 0.5) > 0.6 ? "var(--success)" : "var(--text-muted)" }}>
                    {((r.auroc ?? 0) * 100).toFixed(1)}%
                  </td>
                  <td>{((r.mean_groundedness ?? 0) * 100).toFixed(1)}%</td>
                  <td style={{ color: (r.hallucination_rate ?? 0) > 0.3 ? "var(--danger)" : "var(--success)" }}>
                    {((r.hallucination_rate ?? 0) * 100).toFixed(1)}%
                  </td>
                  <td style={{ color: "var(--text-muted)" }}>{r.total_candidates ?? "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <p style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 10 }}>
          ★ <strong>Hybrid (Recommended)</strong> = the final Master Agent output — what the user acts on.
          Gold standard: FDA-approved drugs + known repurposed candidates from literature.
        </p>
      </div>
    </>
  );
}

function StatusBadge({ status }) {
  const approved = status?.includes("Approved") || status?.includes("approved");
  const phase = status?.includes("Phase");
  return (
    <span style={{
      fontSize: 11, fontWeight: 600, padding: "2px 8px", borderRadius: 12,
      background: approved ? "rgba(52,199,123,0.15)" : phase ? "rgba(79,140,255,0.15)" : "rgba(245,184,78,0.15)",
      color: approved ? "var(--success)" : phase ? "var(--primary)" : "var(--warn)",
    }}>
      {status}
    </span>
  );
}

function CaseStudiesSection({ realResults }) {
  // Merge static metadata with real pipeline results
  const merged = CASE_STUDIES.map((cs) => {
    const real = realResults?.find((r) => r.id === cs.id);
    return { ...cs, real };
  });

  return (
    <>
      <h2 style={{ fontSize: 16, fontWeight: 700, margin: "0 0 8px" }}>
        Validated Case Studies — Known Repurposing Successes
      </h2>
      <p style={{ fontSize: 13, color: "var(--text-secondary)", margin: "0 0 16px" }}>
        Three landmark repurposing cases run through the actual Drugenix pipeline.
        Pipeline results show whether the system surfaces each drug from PubMed evidence.
      </p>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(340px, 1fr))", gap: 14, marginBottom: 24 }}>
        {merged.map((cs) => {
          const pr = cs.real?.pipeline_result;
          const kg = cs.real?.kg_validation;
          return (
            <div key={cs.drug} className="card" style={{ borderLeft: `4px solid ${cs.color}` }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 8 }}>
                <span style={{ fontWeight: 700, fontSize: 15 }}>{cs.drug}</span>
                <StatusBadge status={cs.status} />
              </div>
              <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 6 }}>
                <span style={{ color: "var(--warn)", fontWeight: 600 }}>{cs.from}</span>
                {" → "}
                <span style={{ color: "var(--primary)", fontWeight: 600 }}>{cs.to}</span>
              </div>
              <div style={{ fontSize: 12.5, color: "var(--text-secondary)", marginBottom: 6, lineHeight: 1.5 }}>
                <strong>Mechanism:</strong> {cs.mechanism}
              </div>
              <div style={{ fontSize: 12, color: "var(--text-muted)", lineHeight: 1.5, marginBottom: pr ? 10 : 0 }}>
                <strong>Evidence:</strong> {cs.evidence}
              </div>

              {pr && (
                <div style={{ borderTop: "1px solid var(--border)", paddingTop: 10, marginTop: 6 }}>
                  <div style={{ fontSize: 11, fontWeight: 700, color: "var(--text-muted)", marginBottom: 6, letterSpacing: "0.05em" }}>
                    PIPELINE RESULT
                  </div>
                  <div style={{ display: "flex", gap: 16, flexWrap: "wrap", fontSize: 12 }}>
                    <div>
                      <span style={{ color: "var(--text-muted)" }}>Candidates: </span>
                      <span style={{ fontWeight: 600 }}>{pr.total_candidates}</span>
                    </div>
                    <div>
                      <span style={{ color: "var(--text-muted)" }}>Drug found: </span>
                      <span style={{ fontWeight: 600, color: pr.target_drug_found ? "var(--success)" : "var(--warn)" }}>
                        {pr.target_drug_found ? `Yes (rank #${pr.target_drug_rank})` : "Not in top results"}
                      </span>
                    </div>
                    {pr.target_drug_confidence != null && (
                      <div>
                        <span style={{ color: "var(--text-muted)" }}>Confidence: </span>
                        <span style={{ fontWeight: 600 }}>{(pr.target_drug_confidence * 100).toFixed(0)}%</span>
                      </div>
                    )}
                  </div>
                  {kg && (
                    <div style={{ fontSize: 11.5, color: "var(--text-secondary)", marginTop: 6 }}>
                      <span style={{ color: "var(--text-muted)" }}>KG: </span>
                      {kg.has_direct_path ? (
                        <span style={{ color: "var(--success)" }}>Direct path found — {kg.known_mechanism}</span>
                      ) : kg.has_indirect_path ? (
                        <span style={{ color: "var(--primary)" }}>Indirect path ({kg.path_length} hops)</span>
                      ) : (
                        <span style={{ color: "var(--text-muted)" }}>{kg.explanation}</span>
                      )}
                    </div>
                  )}
                  {pr.top_5_candidates?.length > 0 && (
                    <div style={{ marginTop: 8 }}>
                      <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 4 }}>Top candidates:</div>
                      <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                        {pr.top_5_candidates.map((c, i) => (
                          <span key={i} style={{
                            fontSize: 11, padding: "2px 7px", borderRadius: 8,
                            background: c.kg_validated ? "rgba(52,199,123,0.12)" : "rgba(154,165,189,0.1)",
                            color: c.kg_validated ? "var(--success)" : "var(--text-secondary)",
                            border: `1px solid ${c.kg_validated ? "rgba(52,199,123,0.3)" : "var(--border)"}`,
                          }}>
                            {c.drug}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
              {!pr && cs.real?.status === "error" && (
                <div style={{ fontSize: 11, color: "var(--danger)", marginTop: 6 }}>
                  Pipeline error: {cs.real.error}
                </div>
              )}
              {!cs.real && (
                <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 8, fontStyle: "italic" }}>
                  Run scripts/run_case_studies.py to see pipeline results
                </div>
              )}
            </div>
          );
        })}
      </div>
    </>
  );
}

function TransESection({ status, benchmark }) {
  const [training, setTraining] = useState(false);
  const [predictions, setPredictions] = useState({});
  const [loadingPreds, setLoadingPreds] = useState(false);
  const diseases = ["alzheimer disease", "type 2 diabetes", "breast cancer", "rheumatoid arthritis"];

  async function handleTrain() {
    setTraining(true);
    try { await trainEmbeddings(100, 64); } catch {}
  }

  async function loadPredictions() {
    if (!status?.trained) return;
    setLoadingPreds(true);
    try {
      const results = {};
      for (const d of diseases) {
        try {
          const r = await getLinkPredictions(d, 5);
          results[d] = r.predictions || [];
        } catch {}
      }
      setPredictions(results);
    } finally { setLoadingPreds(false); }
  }

  useEffect(() => { loadPredictions(); }, [status?.trained]);

  return (
    <>
      <h2 style={{ fontSize: 16, fontWeight: 700, margin: "0 0 8px" }}>
        TransE Link Prediction — Novel Drug-Disease Hypotheses
      </h2>
      <p style={{ fontSize: 13, color: "var(--text-secondary)", margin: "0 0 16px" }}>
        TransE (Bordes et al., 2013) learns entity/relation embeddings where
        <em> h + r ≈ t</em> for known triples. After training on our KG, it scores unseen
        drug-disease pairs — surfacing repurposing hypotheses beyond what's in the graph.
      </p>

      {!status?.trained ? (
        <div className="card" style={{ marginBottom: 16, borderLeft: "4px solid var(--primary)" }}>
          <div style={{ fontWeight: 700, marginBottom: 6 }}>TransE model not trained yet</div>
          <div style={{ fontSize: 13, color: "var(--text-secondary)", marginBottom: 12 }}>
            Training takes 3–5 minutes on CPU. The model will learn from {" "}
            <strong>2,300+ KG triples</strong> (drug→target→disease relationships).
          </div>
          <button className="btn btn-primary" onClick={handleTrain} disabled={training}>
            {training ? <><span className="spinner" /> Training… (check back in 5 min)</> : "▶ Train TransE Model"}
          </button>
          {training && (
            <div className="banner banner-info" style={{ marginTop: 10 }}>
              Training started in background. Refresh this page after ~5 minutes.
            </div>
          )}
        </div>
      ) : (
        <>
          {benchmark && (
            <div className="card" style={{ marginBottom: 16 }}>
              <div className="card-title">PREDICT-style Benchmark — TransE on FDA-approved pairs</div>
              <p style={{ fontSize: 12.5, color: "var(--text-secondary)", margin: "0 0 14px" }}>
                Evaluated on held-out 20% of ChEMBL Phase 4 (FDA-approved) drug-disease pairs.
                {benchmark.benchmark_split && (
                  <> Train: {benchmark.benchmark_split.train_pairs} pairs · Test: {benchmark.benchmark_split.test_pos_pairs} pos / {benchmark.benchmark_split.test_neg_pairs} neg.</>
                )}
              </p>
              <div className="grid grid-4">
                {[
                  { label: "AUROC", value: benchmark.metrics?.auroc },
                  { label: "Hit@1", value: benchmark.metrics?.hit_at_1 },
                  { label: "Hit@5", value: benchmark.metrics?.hit_at_5 },
                  { label: "Hit@10", value: benchmark.metrics?.hit_at_10 },
                ].map(({ label, value }) => (
                  <div key={label} className="stat-card" style={{ textAlign: "center" }}>
                    <div style={{ fontSize: 26, fontWeight: 800, color: "var(--primary)" }}>
                      {value != null ? `${(value * 100).toFixed(1)}%` : "—"}
                    </div>
                    <div className="stat-label">{label}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="card" style={{ marginBottom: 16 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
              <div className="card-title" style={{ margin: 0 }}>Novel Predictions by Disease</div>
              {loadingPreds && <span style={{ fontSize: 12, color: "var(--text-muted)" }}>Loading…</span>}
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))", gap: 14 }}>
              {diseases.map((disease) => (
                <div key={disease} style={{ background: "var(--bg-elevated)", borderRadius: 8, padding: 14 }}>
                  <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 10, color: "var(--primary)" }}>
                    {disease.split(" ").map(w => w[0].toUpperCase() + w.slice(1)).join(" ")}
                  </div>
                  {(predictions[disease] || []).length === 0 ? (
                    <div style={{ fontSize: 12, color: "var(--text-muted)" }}>No predictions loaded</div>
                  ) : (
                    <ol style={{ margin: 0, paddingLeft: 18 }}>
                      {(predictions[disease] || []).map((p, i) => (
                        <li key={i} style={{ fontSize: 12.5, marginBottom: 5 }}>
                          <span style={{ fontWeight: 600 }}>{p.drug}</span>
                          <span style={{
                            marginLeft: 8, fontSize: 11, color: "var(--text-muted)",
                            fontVariantNumeric: "tabular-nums",
                          }}>
                            {(p.confidence * 100).toFixed(0)}%
                          </span>
                        </li>
                      ))}
                    </ol>
                  )}
                </div>
              ))}
            </div>
            <p style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 12 }}>
              Confidence = normalized inverse TransE distance. Higher = stronger embedding-space association.
              These are <em>novel</em> predictions, not just KG retrieval.
            </p>
          </div>
        </>
      )}
    </>
  );
}

export default function MetricsPage() {
  const [metrics, setMetrics] = useState(null);
  const [compEval, setCompEval] = useState(null);
  const [caseStudies, setCaseStudies] = useState(null);
  const [predictBenchmark, setPredictBenchmark] = useState(null);
  const [transeStatus, setTranseStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [compLoading, setCompLoading] = useState(true);
  const [error, setError] = useState(null);
  const [benchRunning, setBenchRunning] = useState(false);
  const [benchError, setBenchError] = useState(null);

  function loadData() {
    getMetrics()
      .then(setMetrics)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));

    getComparativeEvaluation()
      .then(setCompEval)
      .catch(() => setCompEval(null))
      .finally(() => setCompLoading(false));

    getCaseStudies().then(setCaseStudies).catch(() => setCaseStudies(null));
    getPredictBenchmark().then(setPredictBenchmark).catch(() => setPredictBenchmark(null));
    getLinkPredictionStatus().then(setTranseStatus).catch(() => setTranseStatus(null));
  }

  useEffect(() => {
    loadData();
  }, []);

  useEffect(() => {
    if (!benchRunning) return;
    const id = setInterval(() => {
      getComparativeEvaluation()
        .then((data) => {
          setCompEval(data);
          setBenchRunning(false);
          clearInterval(id);
          // run_benchmark.py also regenerates metrics_drugenix_mvp.json
          // (backs the "Retrieval & Generation Benchmark" section below),
          // but this poll previously only re-fetched the comparative-
          // evaluation data — that section was silently left showing
          // whatever was loaded when the page first opened, even after a
          // successful fresh run finished. Re-pull everything the
          // benchmark script can change once it's done.
          getMetrics().then(setMetrics).catch(() => {});
        })
        .catch(() => {});
    }, 30000);
    return () => clearInterval(id);
  }, [benchRunning]);

  async function handleRunBenchmark() {
    setBenchRunning(true);
    setBenchError(null);
    try {
      await runBenchmark();
    } catch (e) {
      // Previously this silently swallowed any failure (missing await'd
      // error, wrong base URL hitting the frontend's own dev server
      // instead of the backend) — the button would appear to do nothing
      // with zero feedback. Surface it instead.
      setBenchRunning(false);
      setBenchError(e.message || "Failed to start benchmark run.");
    }
  }

  const retrieval = metrics?.retrieval || {};
  const generation = metrics?.generation || {};
  const ablation = metrics?.ablation || {};

  const precisionData = Ks.map((k) => ({ k: `P@${k}`, value: retrieval[`P@${k}`] ?? 0 }));
  const topKData = Ks.map((k) => ({ k: `Top-${k}`, value: generation[`Top-${k} Accuracy`] ?? 0 }));
  const ablationData = [
    { name: "Without RAG", correctness: ablation.without_rag_correctness ?? 0, groundedness: ablation.without_rag_groundedness ?? 0 },
    { name: "With RAG", correctness: ablation.with_rag_correctness ?? 0, groundedness: ablation.with_rag_groundedness ?? 0 },
  ];
  const hasMetrics = Object.keys(retrieval).length > 0 || Object.keys(generation).length > 0;

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Evaluation Metrics</h1>
          <p className="page-subtitle">
            Real benchmark evaluation — Pure LLM vs Pure Graph vs Hybrid Pipeline, measured against
            FDA-approved + repurposed drug gold standard. Metrics: Hit@k, Precision@k, MRR, AUROC.
          </p>
        </div>
        <button className="btn btn-ghost" onClick={loadData}>↻ Refresh</button>
      </div>

      {/* Comparative evaluation */}
      {compLoading ? (
        <LoadingState label="Loading evaluation data…" />
      ) : (
        <>
          {/* Previously this banner (and the only way to trigger a fresh
              benchmark run) only rendered when compEval was completely
              empty — since a benchmark file already exists from a prior
              run, the refresh option was permanently hidden behind
              already-stale data with no way to update it from the UI. */}
          <RunBenchmarkBanner onRun={handleRunBenchmark} running={benchRunning} error={benchError} />
          {compEval && <ComparativeSection data={compEval} />}
        </>
      )}

      {/* Retrieval / generation benchmark charts */}
      <h2 style={{ fontSize: 16, fontWeight: 700, margin: "0 0 16px" }}>
        Retrieval &amp; Generation Benchmark
      </h2>

      {loading ? (
        <LoadingState label="Loading metrics…" />
      ) : error ? (
        <div className="banner banner-error">{error}</div>
      ) : (
        <>
          {!hasMetrics && (
            <div className="banner banner-info" style={{ marginBottom: 16 }}>
              Run the benchmark above to populate these charts with real evaluation data.
            </div>
          )}

          {hasMetrics && (
            <>
              <div className="grid grid-4" style={{ marginBottom: 20 }}>
                <MetricBadge label="Hit@5" value={generation["Top-5 Accuracy"]} highlight />
                <MetricBadge label="MRR" value={generation.mrr} highlight />
                <MetricBadge label="AUROC" value={generation.auroc} highlight />
                <MetricBadge label="Groundedness" value={generation.mean_groundedness} highlight />
              </div>

              <div className="grid grid-2" style={{ marginBottom: 20 }}>
                <ChartCard title="Precision@k (Hybrid Pipeline)">
                  <BarChart data={precisionData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="k" stroke="#64748b" fontSize={12} />
                    <YAxis domain={[0, 1]} stroke="#64748b" fontSize={12} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                    <Tooltip contentStyle={tooltipStyle} formatter={(v) => `${(v * 100).toFixed(1)}%`} />
                    <Bar dataKey="value" fill="#4f8cff" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ChartCard>

                <ChartCard title="Hit@k Accuracy (Hybrid Pipeline)">
                  <BarChart data={topKData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="k" stroke="#64748b" fontSize={12} />
                    <YAxis domain={[0, 1]} stroke="#64748b" fontSize={12} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                    <Tooltip contentStyle={tooltipStyle} formatter={(v) => `${(v * 100).toFixed(1)}%`} />
                    <Bar dataKey="value" fill="#f5b84e" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ChartCard>

                <ChartCard title="RAG Ablation — Without vs With Evidence Retrieval">
                  <BarChart data={ablationData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="name" stroke="#64748b" fontSize={12} />
                    <YAxis domain={[0, 1]} stroke="#64748b" fontSize={12} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                    <Tooltip contentStyle={tooltipStyle} formatter={(v) => `${(v * 100).toFixed(1)}%`} />
                    <Legend />
                    <Bar dataKey="correctness" name="Hit@5 Accuracy" fill="#4f8cff" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="groundedness" name="Groundedness" fill="#34c77b" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ChartCard>

                <div className="card">
                  <div className="card-title">Mean Groundedness</div>
                  <div style={{ fontSize: 48, fontWeight: 800, color: "var(--primary)" }}>
                    {generation.mean_groundedness != null
                      ? `${(generation.mean_groundedness * 100).toFixed(1)}%`
                      : "—"}
                  </div>
                  <div style={{ fontSize: 13, color: "var(--text-muted)", marginTop: 6, lineHeight: 1.6 }}>
                    Fraction of recommended drug mechanism claims directly supported by PubMed abstract text.
                    A high groundedness score means the system explains <em>why</em> based on real literature,
                    not hallucination.
                  </div>
                </div>
              </div>
            </>
          )}
        </>
      )}

      {/* TransE link prediction */}
      <TransESection status={transeStatus} benchmark={predictBenchmark} />

      {/* Case studies */}
      <CaseStudiesSection realResults={caseStudies} />
    </div>
  );
}
