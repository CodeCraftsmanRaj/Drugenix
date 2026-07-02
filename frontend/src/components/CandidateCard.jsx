import { useState } from "react";
import StatusBadge from "./StatusBadge.jsx";
import ConfidenceBar from "./ConfidenceBar.jsx";
import ReasoningTrace from "./ReasoningTrace.jsx";

export default function CandidateCard({ candidate }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="accordion-item" style={{ marginBottom: 12 }}>
      {/* ── Header (always visible) ── */}
      <div className="accordion-header" onClick={() => setOpen(!open)}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, flex: 1, minWidth: 0 }}>
          <span style={{ fontWeight: 700, fontSize: 13, color: "var(--text-muted)", width: 26, flexShrink: 0 }}>
            #{candidate.rank}
          </span>
          <span style={{ fontWeight: 700, fontSize: 15 }}>{candidate.drug}</span>
          <StatusBadge status={candidate.validation_status} />
          {candidate.kg_validated && (
            <span className="badge badge-neutral" title="Confirmed by Knowledge Graph">◈ KG</span>
          )}
          {candidate.ner_confirmed && (
            <span className="badge badge-neutral" title="Confirmed by NER">✓ NER</span>
          )}
          {candidate.admet && (
            <span
              className={`badge badge-${candidate.admet.verdict === "pass" ? "pass" : candidate.admet.verdict === "borderline" ? "warn" : "fail"}`}
              title={`MW:${candidate.admet.molecular_weight} LogP:${candidate.admet.logp} HBD:${candidate.admet.hbd} HBA:${candidate.admet.hba} | ${candidate.admet.verdict_label}`}
            >
              {candidate.admet.verdict === "pass" ? "✓" : candidate.admet.verdict === "borderline" ? "~" : "✗"} Ro5
            </span>
          )}
          {candidate.clinical_trials?.has_trials && (
            <span
              className="badge"
              style={{ background: "rgba(52,199,123,0.15)", color: "#34c77b", border: "1px solid rgba(52,199,123,0.3)" }}
              title={`${candidate.clinical_trials.trial_count} clinical trial(s) found on ClinicalTrials.gov`}
            >
              ⚗ {candidate.clinical_trials.highest_phase || "In Trial"}
            </span>
          )}
          {candidate.hallucination_flags?.length > 0 && (
            <span className="badge badge-fail" title={candidate.hallucination_flags.join(", ")}>
              ⚠ {candidate.hallucination_flags.length}
            </span>
          )}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 16, flexShrink: 0 }}>
          <ConfidenceBar value={candidate.final_confidence} />
          <span style={{ fontSize: 13, color: "var(--text-muted)" }}>{open ? "▲" : "▼"}</span>
        </div>
      </div>

      {/* ── Repurposing summary (always visible below header) ── */}
      {(candidate.original_use || candidate.repurposing_rationale) && (
        <div style={{
          padding: "12px 16px",
          background: "var(--bg-elevated)",
          borderTop: "1px solid var(--border)",
          display: "flex",
          flexDirection: "column",
          gap: 8,
        }}>
          {candidate.original_use && (
            <div style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
              <span style={{
                flexShrink: 0,
                fontSize: 11,
                fontWeight: 700,
                color: "var(--text-muted)",
                textTransform: "uppercase",
                letterSpacing: "0.05em",
                paddingTop: 2,
                width: 120,
              }}>
                Currently used for
              </span>
              <span style={{ fontSize: 13.5, color: "var(--warn)", fontWeight: 600 }}>
                {candidate.original_use}
              </span>
            </div>
          )}
          {candidate.repurposing_rationale && (
            <div style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
              <span style={{
                flexShrink: 0,
                fontSize: 11,
                fontWeight: 700,
                color: "var(--text-muted)",
                textTransform: "uppercase",
                letterSpacing: "0.05em",
                paddingTop: 2,
                width: 120,
              }}>
                Why repurpose
              </span>
              <span style={{ fontSize: 13.5, color: "var(--text-primary)", lineHeight: 1.55 }}>
                {candidate.repurposing_rationale}
              </span>
            </div>
          )}
        </div>
      )}

      {/* ── Expanded detail ── */}
      {open && (
        <div className="accordion-body" style={{ paddingTop: 14 }}>
          {candidate.known_mechanism && (
            <p style={{ marginTop: 0 }}>
              <strong>Molecular mechanism:</strong> {candidate.known_mechanism}
            </p>
          )}

          {candidate.explanation && !candidate.repurposing_rationale && (
            <p>{candidate.explanation}</p>
          )}

          {candidate.known_targets?.length > 0 && (
            <p>
              <strong>Known targets:</strong> {candidate.known_targets.join(", ")}
            </p>
          )}

          {candidate.evidence_pmids?.length > 0 && (
            <div style={{ marginBottom: 12 }}>
              <strong style={{ display: "block", marginBottom: 6 }}>
                Supporting PubMed evidence ({candidate.evidence_pmids.length} papers):
              </strong>
              {candidate.evidence_pmids.map((pmid) => (
                <a
                  key={pmid}
                  className="pmid-link"
                  href={`https://pubmed.ncbi.nlm.nih.gov/${pmid}/`}
                  target="_blank"
                  rel="noreferrer"
                >
                  PMID:{pmid}
                </a>
              ))}
            </div>
          )}

          {candidate.admet && (
            <div style={{ background: "var(--bg-elevated)", borderRadius: 8, padding: "10px 14px", marginBottom: 12 }}>
              <strong style={{ display: "block", marginBottom: 6, fontSize: 12, textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--text-muted)" }}>
                ADMET / Drug-Likeness (Lipinski Ro5)
              </strong>
              <div style={{ display: "flex", gap: 16, flexWrap: "wrap", fontSize: 13 }}>
                <span>MW: <strong>{candidate.admet.molecular_weight}</strong></span>
                <span>LogP: <strong>{candidate.admet.logp}</strong></span>
                <span>HBD: <strong>{candidate.admet.hbd}</strong></span>
                <span>HBA: <strong>{candidate.admet.hba}</strong></span>
                <span>RotB: <strong>{candidate.admet.rotatable_bonds}</strong></span>
                <span>TPSA: <strong>{candidate.admet.tpsa}</strong></span>
                <span>Violations: <strong style={{ color: candidate.admet.ro5_violations === 0 ? "var(--success)" : "var(--danger)" }}>{candidate.admet.ro5_violations}</strong></span>
              </div>
              <a href={candidate.admet.pubchem_url} target="_blank" rel="noreferrer" style={{ fontSize: 11, color: "var(--accent)", marginTop: 6, display: "inline-block" }}>
                View on PubChem ↗
              </a>
            </div>
          )}

          {candidate.clinical_trials?.has_trials && (
            <div style={{ background: "var(--bg-elevated)", borderRadius: 8, padding: "10px 14px", marginBottom: 12 }}>
              <strong style={{ display: "block", marginBottom: 6, fontSize: 12, textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--text-muted)" }}>
                Clinical Trials ({candidate.clinical_trials.trial_count} found · Highest: {candidate.clinical_trials.highest_phase})
              </strong>
              {candidate.clinical_trials.trials?.slice(0, 2).map((t) => (
                <div key={t.nct_id} style={{ fontSize: 12.5, marginBottom: 4 }}>
                  <a href={t.url} target="_blank" rel="noreferrer" style={{ color: "var(--accent)" }}>{t.nct_id}</a>
                  {" · "}{t.phase}{" · "}<span style={{ color: "var(--text-muted)" }}>{t.status}</span>
                </div>
              ))}
            </div>
          )}

          {candidate.conflicts?.length > 0 && (
            <div className="banner banner-error" style={{ marginTop: 10 }}>
              {candidate.conflicts.join(" · ")}
            </div>
          )}

          <strong style={{ display: "block", margin: "14px 0 6px" }}>
            Agent reasoning trace:
          </strong>
          <ReasoningTrace steps={candidate.reasoning_trace} />
        </div>
      )}
    </div>
  );
}
