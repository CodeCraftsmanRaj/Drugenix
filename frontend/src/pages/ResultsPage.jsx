import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { getHistoryItem } from "../api/client.js";
import LoadingState from "../components/LoadingState.jsx";
import CandidateCard from "../components/CandidateCard.jsx";
import StatusBadge from "../components/StatusBadge.jsx";

const TABS = ["Recommended", "Suggested Combinations", "Review Needed", "Rejected", "Clinical Trials", "Retrieved Papers", "Knowledge Graph", "NER Entities"];

export default function ResultsPage() {
  const { queryId } = useParams();
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState("Recommended");
  const [showSymptomDetail, setShowSymptomDetail] = useState(false);

  useEffect(() => {
    async function load() {
      setLoading(true);
      setError(null);
      try {
        if (queryId === "last") {
          const cached = sessionStorage.getItem("drugenix_last_result");
          if (cached) {
            setResult(JSON.parse(cached));
          } else {
            setError("No cached result found.");
          }
        } else {
          const data = await getHistoryItem(queryId);
          setResult(data.result);
        }
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [queryId]);

  if (loading) return <LoadingState label="Loading result…" />;
  if (error) return <div className="banner banner-error">{error}</div>;
  if (!result) return null;

  const agg = result.aggregation || {};
  const planner = result.planner || {};
  const validation = result.validation || {};
  const kgResults = result.kg_validation || [];
  const ner = result.ner || {};
  const admetData = result.admet || {};
  const trialsData = result.clinical_trials || {};
  const combinations = result.combinations || [];

  const isSymptomMode = result.mode === "symptom";
  const recommendedCount = agg.recommended?.length ?? 0;
  const headlineSubtitle = isSymptomMode
    ? `${recommendedCount} drug candidate${recommendedCount === 1 ? "" : "s"} found for these symptoms` +
      (combinations.length ? ` · ${combinations.length} suggested combination${combinations.length === 1 ? "" : "s"}` : "")
    : agg.evidence_summary;

  // Enrich candidates with ADMET + clinical trial data by drug name
  function enrich(candidates) {
    if (!candidates) return [];
    return candidates.map((c) => ({
      ...c,
      admet: admetData[c.drug] || null,
      clinical_trials: trialsData[c.drug] || null,
    }));
  }

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">
            {isSymptomMode ? "Drugs found for your symptoms" : result.query}
          </h1>
          <p className="page-subtitle">{headlineSubtitle}</p>
        </div>
        <Link to="/query" className="btn btn-ghost">+ New Query</Link>
      </div>

      {result.data_quality_warnings?.length > 0 && (
        <div className="banner banner-warn" style={{ marginBottom: 16 }}>
          <div style={{ fontWeight: 700, marginBottom: 6 }}>
            Data quality notice — this result may be incomplete
          </div>
          <ul style={{ margin: 0, paddingLeft: 18 }}>
            {result.data_quality_warnings.map((w, i) => (
              <li key={i} style={{ fontSize: 13, marginBottom: 4 }}>{w}</li>
            ))}
          </ul>
        </div>
      )}

      {isSymptomMode && result.symptom_analysis && (
        <div className="card" style={{ marginBottom: 20, borderColor: "var(--warn)" }}>
          <div
            style={{ display: "flex", justifyContent: "space-between", alignItems: "center", cursor: "pointer" }}
            onClick={() => setShowSymptomDetail(!showSymptomDetail)}
          >
            <div className="card-title" style={{ marginBottom: 0 }}>
              How we interpreted your symptoms
              <span style={{ fontWeight: 400, fontSize: 12.5, color: "var(--text-muted)", marginLeft: 10 }}>
                {(result.symptom_analysis.candidate_conditions || []).map((c) => c.condition).join(", ") || "no candidates derived"}
              </span>
            </div>
            <span style={{ fontSize: 13, color: "var(--text-muted)" }}>{showSymptomDetail ? "▲ Hide" : "▼ Show detail"}</span>
          </div>

          {!result.symptom_analysis.llm_available && (
            <div className="banner banner-warn" style={{ marginTop: 12, fontSize: 13 }}>
              No LLM provider was available to interpret the symptoms — configure an LLM provider
              (OpenAI / Gemini / Ollama) and retry.
              {result.symptom_analysis.llm_error && (
                <div style={{ marginTop: 6, fontFamily: "monospace", fontSize: 12, opacity: 0.85 }}>
                  Reason: {result.symptom_analysis.llm_error}
                </div>
              )}
            </div>
          )}

          {showSymptomDetail && (
            <>
              <div className="banner banner-info" style={{ margin: "12px 0", fontSize: 13 }}>
                {result.symptom_analysis.disclaimer}
              </div>
              <table className="table">
                <thead>
                  <tr>
                    <th>Candidate condition</th>
                    <th>LLM confidence</th>
                    <th>PubMed hits</th>
                    <th>Grounded confidence</th>
                    <th>Rationale</th>
                  </tr>
                </thead>
                <tbody>
                  {(result.symptom_analysis.candidate_conditions || []).map((c) => (
                    <tr key={c.condition}>
                      <td style={{ fontWeight: 600 }}>
                        {c.condition}
                        {c.is_novel_presentation && (
                          <span className="badge badge-warn" style={{ marginLeft: 6 }}>possible novel</span>
                        )}
                      </td>
                      <td>{c.llm_confidence?.toFixed(2)}</td>
                      <td>{c.pubmed_hits}</td>
                      <td>{c.grounded_confidence?.toFixed(2)}</td>
                      <td style={{ fontSize: 12.5, color: "var(--text-muted)" }}>{c.rationale}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </div>
      )}

      {isSymptomMode && (result.standard_treatment_reference?.length > 0) && (
        <div className="card" style={{ marginBottom: 20 }}>
          <div className="card-title">Standard / First-Line Treatment</div>
          <p style={{ fontSize: 12.5, color: "var(--text-muted)", marginTop: 0, marginBottom: 12 }}>
            The obvious, textbook answer for these symptoms — shown for reference. This is separate from
            the repurposing candidates below, which deliberately exclude drugs already used for the
            matched condition (that's the point of a repurposing tool — finding non-obvious options).
          </p>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {result.standard_treatment_reference.map((t) => (
              <div key={t.drug} style={{ display: "flex", gap: 10, alignItems: "flex-start", fontSize: 13.5 }}>
                <span style={{ fontWeight: 700, minWidth: 200 }}>{t.drug}</span>
                <span style={{ color: "var(--text-secondary)" }}>{t.note}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {planner.is_multi_disease && (
        <div className="banner banner-info">
          Planner Agent decomposed this into {planner.total_tasks} sub-queries:{" "}
          {planner.sub_tasks?.map((t) => t.disease_query).join(", ")}
        </div>
      )}

      <div className="grid grid-4" style={{ marginBottom: 24 }}>
        <div className="stat-card">
          <div className="stat-value">{agg.total_candidates ?? 0}</div>
          <div className="stat-label">Total candidates</div>
        </div>
        <div className="stat-card">
          <div className="stat-value" style={{ color: "var(--success)" }}>{agg.recommended?.length ?? 0}</div>
          <div className="stat-label">Recommended</div>
        </div>
        <div className="stat-card">
          <div className="stat-value" style={{ color: "var(--warn)" }}>{agg.review_needed?.length ?? 0}</div>
          <div className="stat-label">Needs review</div>
        </div>
        <div className="stat-card">
          <div className="stat-value" style={{ color: "var(--danger)" }}>{agg.hallucinations_flagged ?? 0}</div>
          <div className="stat-label">Hallucination flags</div>
        </div>
      </div>

      <div className="tabs">
        {TABS.map((t) => (
          <div key={t} className={`tab ${activeTab === t ? "active" : ""}`} onClick={() => setActiveTab(t)}>
            {t}
          </div>
        ))}
      </div>

      {activeTab === "Recommended" && (
        <CandidateList candidates={enrich(agg.recommended)} emptyMsg="No candidates met the recommendation threshold." />
      )}

      {activeTab === "Suggested Combinations" && (
        <div>
          <div className="card" style={{ marginBottom: 16, fontSize: 13, color: "var(--text-secondary)" }}>
            Pairs of the top candidates worth testing together, ranked by how complementary their mechanisms
            look — different targets/pathways score higher than two drugs that just duplicate the same
            mechanism. Deterministic, based on Knowledge Graph target data (or mechanism text when KG data
            isn't available) — not an LLM guess, and not a substitute for pharmacological review of real
            drug-drug interactions before testing.
          </div>
          {combinations.length === 0 ? (
            <div className="empty-state">
              <div className="empty-state-icon">∅</div>
              <div>No combination suggestions — need at least 2 reasonably confident candidates to pair.</div>
            </div>
          ) : (
            combinations.map((c) => (
              <div key={`${c.drug_a}+${c.drug_b}`} className="accordion-item" style={{ marginBottom: 12, padding: "14px 16px" }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 10 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <span style={{ fontWeight: 700, fontSize: 15 }}>{c.drug_a} + {c.drug_b}</span>
                    <span className={`badge ${c.complementary_mechanisms ? "badge-pass" : "badge-warn"}`}>
                      {c.complementary_mechanisms ? "Complementary" : "Overlapping"}
                    </span>
                    <span className="badge badge-neutral" title="How the complementarity was determined">
                      {c.confidence_basis === "kg_targets" ? "KG target data" : "mechanism text"}
                    </span>
                  </div>
                  <span style={{ fontSize: 13, color: "var(--text-muted)" }}>
                    combined score {(c.combined_score * 100).toFixed(0)}%
                  </span>
                </div>
                <div style={{ fontSize: 13.5, color: "var(--text-primary)", marginTop: 10, lineHeight: 1.55 }}>
                  {c.rationale}
                </div>
                {(c.targets_a?.length > 0 || c.targets_b?.length > 0) && (
                  <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 8 }}>
                    {c.drug_a} targets: {c.targets_a?.join(", ") || "—"} · {c.drug_b} targets: {c.targets_b?.join(", ") || "—"}
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      )}

      {activeTab === "Review Needed" && (
        <CandidateList candidates={enrich(agg.review_needed)} emptyMsg="No candidates require review." />
      )}
      {activeTab === "Rejected" && (
        <CandidateList candidates={enrich(agg.rejected)} emptyMsg="No candidates were rejected." />
      )}

      {activeTab === "Clinical Trials" && (
        <div className="card">
          <div className="card-title">ClinicalTrials.gov Cross-Reference</div>
          <p style={{ fontSize: 13, color: "var(--text-muted)", marginTop: 0 }}>
            Active and completed trials for top candidates matched to this disease query.
          </p>
          {Object.keys(trialsData).length === 0 ? (
            <div className="empty-state"><div>No trial data yet — run a new query to populate.</div></div>
          ) : (
            Object.entries(trialsData).map(([drug, td]) => (
              <div key={drug} style={{ marginBottom: 20 }}>
                <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 8, display: "flex", alignItems: "center", gap: 8 }}>
                  {drug}
                  {td.lookup_failed
                    ? <span className="badge" style={{ background: "rgba(255,159,10,0.15)", color: "var(--warn)" }}>⚠ Lookup failed</span>
                    : td.has_trials
                      ? <span className="badge" style={{ background: "rgba(52,199,123,0.15)", color: "#34c77b" }}>⚗ {td.trial_count} trial{td.trial_count !== 1 ? "s" : ""}</span>
                      : <span className="badge badge-neutral">No trials found</span>
                  }
                  {td.highest_phase && <span className="badge badge-neutral">{td.highest_phase}</span>}
                </div>
                {td.trials?.map((t) => (
                  <div key={t.nct_id} style={{ padding: "10px 14px", background: "var(--bg-elevated)", borderRadius: 8, marginBottom: 6, fontSize: 13 }}>
                    <div style={{ fontWeight: 600 }}>{t.title}</div>
                    <div style={{ color: "var(--text-muted)", marginTop: 4, display: "flex", gap: 12, flexWrap: "wrap" }}>
                      <span>{t.nct_id}</span>
                      <span>{t.phase}</span>
                      <span style={{ color: t.status === "COMPLETED" ? "var(--success)" : t.status === "RECRUITING" ? "var(--warn)" : "var(--text-muted)" }}>{t.status}</span>
                      {t.enrollment && <span>{t.enrollment.toLocaleString()} enrolled</span>}
                      {t.start_date && <span>Started {t.start_date}</span>}
                    </div>
                    <a href={t.url} target="_blank" rel="noreferrer" style={{ fontSize: 12, color: "var(--accent)", marginTop: 4, display: "inline-block" }}>
                      View on ClinicalTrials.gov ↗
                    </a>
                  </div>
                ))}
                {td.lookup_failed ? (
                  <div style={{ fontSize: 12, color: "var(--warn)", padding: "6px 0" }}>
                    ClinicalTrials.gov could not be reached for this drug — this is NOT a confirmed
                    absence of trials, just a failed lookup. Retry the query.
                  </div>
                ) : !td.has_trials && (
                  <div style={{ fontSize: 12, color: "var(--text-muted)", padding: "6px 0" }}>
                    No matching trials found — this may be a novel repurposing hypothesis.
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      )}

      {activeTab === "Retrieved Papers" && (
        <div className="card">
          <div className="card-title">{result.retrieved?.length ?? 0} PubMed papers retrieved</div>
          {(result.retrieved || []).map((doc) => (
            <div key={doc.pmid} className="accordion-item">
              <div className="accordion-header" style={{ cursor: "default" }}>
                <div>
                  <div style={{ fontWeight: 600, fontSize: 13.5 }}>{doc.title}</div>
                  <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 4 }}>
                    {doc.journal} · {doc.year} · PMID:{doc.pmid} · score={doc.retrieval_score?.toFixed(2)}
                  </div>
                </div>
                <a
                  className="btn btn-ghost"
                  style={{ fontSize: 11, padding: "4px 10px" }}
                  href={`https://pubmed.ncbi.nlm.nih.gov/${doc.pmid}/`}
                  target="_blank"
                  rel="noreferrer"
                >
                  View ↗
                </a>
              </div>
              <div className="accordion-body" style={{ paddingTop: 12 }}>
                {doc.abstract?.slice(0, 500)}
                {doc.abstract?.length > 500 ? "…" : ""}
              </div>
            </div>
          ))}
        </div>
      )}

      {activeTab === "Knowledge Graph" && (
        <div className="card">
          <div className="card-title">Knowledge Graph Validation ({kgResults.length} candidates checked)</div>
          <table className="table">
            <thead>
              <tr>
                <th>Drug</th>
                <th>Path Type</th>
                <th>Path</th>
                <th>Targets</th>
                <th>Mechanism</th>
              </tr>
            </thead>
            <tbody>
              {kgResults.map((kr) => (
                <tr key={kr.drug}>
                  <td style={{ fontWeight: 600 }}>{kr.drug}</td>
                  <td>
                    {kr.has_direct_path ? (
                      <span className="badge badge-pass">DIRECT</span>
                    ) : kr.has_indirect_path ? (
                      <span className="badge badge-warn">{kr.path_length}-HOP</span>
                    ) : (
                      <span className="badge badge-fail">NONE</span>
                    )}
                  </td>
                  <td style={{ fontSize: 12, color: "var(--text-muted)" }}>
                    {kr.path_nodes?.join(" → ") || "—"}
                  </td>
                  <td>{kr.known_targets?.join(", ") || "—"}</td>
                  <td style={{ fontSize: 12.5 }}>{kr.known_mechanism || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {activeTab === "NER Entities" && (
        <div className="grid grid-2">
          <div className="card">
            <div className="card-title">Drugs detected via NER ({ner.drugs_found?.length ?? 0})</div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {(ner.drugs_found || []).map((d) => (
                <span key={d} className="badge badge-neutral">{d}</span>
              ))}
            </div>
          </div>
          <div className="card">
            <div className="card-title">Targets detected via NER ({ner.targets_found?.length ?? 0})</div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {(ner.targets_found || []).map((t) => (
                <span key={t} className="badge badge-neutral">{t}</span>
              ))}
            </div>
          </div>
          <div className="card" style={{ gridColumn: "span 2" }}>
            <div className="card-title">Drug–Target pairs ({ner.drug_target_pairs?.length ?? 0})</div>
            <table className="table">
              <thead>
                <tr><th>Drug</th><th>Target</th></tr>
              </thead>
              <tbody>
                {(ner.drug_target_pairs || []).map(([drug, target], i) => (
                  <tr key={i}><td>{drug}</td><td>{target}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function CandidateList({ candidates, emptyMsg }) {
  if (!candidates || candidates.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">∅</div>
        <div>{emptyMsg}</div>
      </div>
    );
  }
  return (
    <div>
      {candidates.map((c) => (
        <CandidateCard key={c.drug} candidate={c} />
      ))}
    </div>
  );
}
