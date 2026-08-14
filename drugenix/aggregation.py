"""
Drugenix - Aggregation / Explanation Agent (Master Agent)
==========================================================
Aggregates cross-source evidence (literature + KG + NER), detects conflicts,
assigns final confidence scores, flags hallucinations, and produces
human-readable reasoning traces.

This is the "Master Agent" described in the project proposal (Slide 4).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from drugenix.knowledge_graph import KGValidationResult
from drugenix.models import DrugCandidate, EvidenceDoc, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class EvidenceSource:
    source_type: str  # "literature", "knowledge_graph", "ner"
    confidence: float
    details: str
    pmids: list[str] = field(default_factory=list)


@dataclass
class ReasoningTrace:
    drug: str
    step: int
    action: str
    detail: str
    source: str = ""


@dataclass
class AggregatedCandidate:
    rank: int
    drug: str
    final_confidence: float
    evidence_sources: list[EvidenceSource] = field(default_factory=list)
    reasoning_trace: list[ReasoningTrace] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    hallucination_flags: list[str] = field(default_factory=list)
    known_targets: list[str] = field(default_factory=list)
    known_mechanism: str = ""
    pmid_count: int = 0
    evidence_pmids: list[str] = field(default_factory=list)
    kg_validated: bool = False
    ner_confirmed: bool = False
    validation_status: str = "UNKNOWN"
    summary: str = ""
    explanation: str = ""
    original_use: str = ""           # what the drug is currently approved for
    repurposing_rationale: str = ""  # why it may work for the queried disease
    evidence_tier: str = "literature"  # "literature" | "mechanistic" (see models.DrugCandidate)


@dataclass
class AggregationReport:
    query: str
    total_candidates: int
    recommended: list[AggregatedCandidate] = field(default_factory=list)
    review_needed: list[AggregatedCandidate] = field(default_factory=list)
    rejected: list[AggregatedCandidate] = field(default_factory=list)
    conflicts_detected: int = 0
    hallucinations_flagged: int = 0
    evidence_summary: str = ""
    methodology: str = ""


# Weights for final confidence scoring
_WEIGHTS = {
    "literature": 0.40,
    "validation": 0.25,
    "knowledge_graph": 0.20,
    "ner": 0.15,
}


def aggregate(
    disease_query: str,
    candidates: list[DrugCandidate],
    validation_results: list[ValidationResult],
    kg_results: list[KGValidationResult],
    ner_drug_names: set[str],
    docs: list[EvidenceDoc],
) -> AggregationReport:
    """
    Fuse evidence from all sources and produce final ranked recommendations.
    """
    val_map = {r.drug.lower(): r for r in validation_results}
    kg_map = {r.drug.lower(): r for r in kg_results}

    aggregated: list[AggregatedCandidate] = []

    for candidate in candidates:
        drug_lower = candidate.drug.lower()
        is_mechanistic = getattr(candidate, "evidence_tier", "literature") == "mechanistic"
        traces: list[ReasoningTrace] = []
        evidence_sources: list[EvidenceSource] = []
        conflicts: list[str] = []
        hallucination_flags: list[str] = []

        # --- Step 1: Literature evidence ---
        lit_confidence = candidate.confidence
        pmids = candidate.evidence_pmids
        traces.append(ReasoningTrace(
            drug=candidate.drug,
            step=1,
            action="Literature retrieval",
            detail=f"Found in {len(pmids)} PubMed papers with confidence {lit_confidence:.2f}",
            source="Reasoning Agent",
        ))
        evidence_sources.append(EvidenceSource(
            source_type="literature",
            confidence=lit_confidence,
            details=f"{len(pmids)} supporting papers",
            pmids=pmids,
        ))

        # --- Step 2: Validation Agent evidence ---
        val_result = val_map.get(drug_lower)
        val_confidence = 0.0
        val_status = "UNKNOWN"
        if val_result:
            val_confidence = val_result.adjusted_confidence
            val_status = val_result.status
            traces.append(ReasoningTrace(
                drug=candidate.drug,
                step=2,
                action="Validation check",
                detail=f"Status={val_status}, adjusted_confidence={val_confidence:.2f}",
                source="Validation Agent",
            ))
            evidence_sources.append(EvidenceSource(
                source_type="validation",
                confidence=val_confidence,
                details=f"Validation: {val_status}",
            ))

            if val_result.warnings:
                for w in val_result.warnings:
                    if "safety" in w.lower():
                        conflicts.append(f"Safety concern from validation: {w}")
        else:
            traces.append(ReasoningTrace(
                drug=candidate.drug,
                step=2,
                action="Validation check",
                detail="No validation result available",
                source="Validation Agent",
            ))

        # --- Step 3: Knowledge Graph evidence ---
        kg_result = kg_map.get(drug_lower)
        kg_confidence = 0.0
        kg_validated = False
        known_targets: list[str] = []
        known_mechanism = ""

        if kg_result:
            if kg_result.has_direct_path:
                kg_confidence = 0.9
                kg_validated = True
                traces.append(ReasoningTrace(
                    drug=candidate.drug,
                    step=3,
                    action="Knowledge Graph validation",
                    detail=f"DIRECT path found: {kg_result.explanation}",
                    source="Knowledge Graph",
                ))
            elif kg_result.has_indirect_path:
                kg_confidence = 0.6 + (0.1 * max(0, 4 - kg_result.path_length))
                kg_validated = True
                traces.append(ReasoningTrace(
                    drug=candidate.drug,
                    step=3,
                    action="Knowledge Graph validation",
                    detail=f"Indirect path ({kg_result.path_length} hops): {kg_result.explanation}",
                    source="Knowledge Graph",
                ))
            else:
                kg_confidence = 0.2
                traces.append(ReasoningTrace(
                    drug=candidate.drug,
                    step=3,
                    action="Knowledge Graph validation",
                    detail=f"No path found: {kg_result.explanation}",
                    source="Knowledge Graph",
                ))

            known_targets = kg_result.known_targets
            known_mechanism = kg_result.known_mechanism

            evidence_sources.append(EvidenceSource(
                source_type="knowledge_graph",
                confidence=kg_confidence,
                details=kg_result.explanation,
            ))

            # Conflict detection: KG says no link but LLM suggests it
            if not kg_result.has_direct_path and not kg_result.has_indirect_path and lit_confidence > 0.7:
                conflicts.append(
                    f"LLM suggests {candidate.drug} (confidence={lit_confidence:.2f}) "
                    f"but no KG path exists — potential hallucination"
                )
        else:
            traces.append(ReasoningTrace(
                drug=candidate.drug,
                step=3,
                action="Knowledge Graph validation",
                detail="KG validation not available",
                source="Knowledge Graph",
            ))

        # --- Step 4: NER confirmation ---
        ner_confirmed = drug_lower in {n.lower() for n in ner_drug_names}
        ner_confidence = 0.8 if ner_confirmed else 0.3

        traces.append(ReasoningTrace(
            drug=candidate.drug,
            step=4,
            action="NER entity confirmation",
            detail=f"{'Confirmed' if ner_confirmed else 'Not found'} in NER extraction from abstracts",
            source="NER Agent",
        ))
        evidence_sources.append(EvidenceSource(
            source_type="ner",
            confidence=ner_confidence,
            details=f"NER {'confirmed' if ner_confirmed else 'not detected'} in abstracts",
        ))

        # --- Step 5: Hallucination detection ---
        # Only flag as hallucinated when there is ZERO PubMed evidence.
        # Absence from our small curated KG (~30 entries) or the general-purpose
        # spaCy NER model is NOT sufficient evidence of hallucination — many
        # legitimate repurposing candidates are simply not yet curated.
        # Mechanistic-tier candidates are EXPECTED to have zero PMIDs by
        # design (they exist specifically because literature is thin/absent
        # for this disease) — flagging them as "likely hallucinated" would
        # be wrong, so they're excluded from this check entirely and instead
        # carry their own explicit "mechanistic/indirect" label everywhere.
        if len(pmids) == 0 and not is_mechanistic:
            hallucination_flags.append("No PMID evidence — likely hallucinated")

        # --- Weighted confidence fusion ---
        scores = {
            "literature": lit_confidence,
            "validation": val_confidence,
            "knowledge_graph": kg_confidence,
            "ner": ner_confidence,
        }
        final_confidence = sum(
            scores[src] * _WEIGHTS[src] for src in _WEIGHTS
        )

        # Boost/penalty adjustments
        if kg_validated and ner_confirmed:
            final_confidence = min(1.0, final_confidence * 1.10)
        elif kg_validated or ner_confirmed:
            final_confidence = min(1.0, final_confidence * 1.05)
        if hallucination_flags:
            final_confidence *= 0.60  # only fires when PMIDs == 0 and not mechanistic
        if val_status == "FAIL":
            final_confidence *= 0.70  # softer penalty — validation FAIL ≠ wrong drug
        if is_mechanistic:
            # Hard ceiling: mechanistic/indirect candidates must never
            # outrank a literature-backed candidate, no matter how strong
            # the target-overlap signal looks numerically.
            final_confidence = min(final_confidence, 0.50)

        traces.append(ReasoningTrace(
            drug=candidate.drug,
            step=5,
            action="Evidence fusion",
            detail=(
                f"Weighted scores: lit={scores['literature']:.2f}*{_WEIGHTS['literature']}, "
                f"val={scores['validation']:.2f}*{_WEIGHTS['validation']}, "
                f"kg={scores['knowledge_graph']:.2f}*{_WEIGHTS['knowledge_graph']}, "
                f"ner={scores['ner']:.2f}*{_WEIGHTS['ner']} "
                f"=> final={final_confidence:.2f}"
            ),
            source="Master Agent",
        ))

        # Build explanation
        explanation_parts = []
        if is_mechanistic:
            explanation_parts.append(
                "MECHANISTIC/INDIRECT candidate — proposed via target-overlap reasoning, "
                "not direct PubMed literature for this disease"
            )
        if known_mechanism:
            explanation_parts.append(f"Mechanism: {known_mechanism}")
        if known_targets:
            explanation_parts.append(f"Known targets: {', '.join(known_targets)}")
        explanation_parts.append(f"Supported by {len(pmids)} PubMed papers")
        if kg_validated:
            explanation_parts.append("Validated by Knowledge Graph")
        if ner_confirmed:
            explanation_parts.append("Confirmed by NER extraction")
        if conflicts:
            explanation_parts.append(f"Conflicts: {len(conflicts)}")

        summary_status = val_status if val_result else ("PASS" if final_confidence >= 0.5 else "WARN" if final_confidence >= 0.3 else "FAIL")

        agg = AggregatedCandidate(
            rank=0,
            drug=candidate.drug,
            final_confidence=round(final_confidence, 3),
            evidence_sources=evidence_sources,
            reasoning_trace=traces,
            conflicts=conflicts,
            hallucination_flags=hallucination_flags,
            known_targets=known_targets,
            known_mechanism=known_mechanism,
            pmid_count=len(pmids),
            evidence_pmids=pmids,
            kg_validated=kg_validated,
            ner_confirmed=ner_confirmed,
            validation_status=summary_status,
            summary=f"[{summary_status}] {candidate.drug} | confidence={final_confidence:.2f} | {len(pmids)} PMIDs",
            explanation=". ".join(explanation_parts),
            original_use=candidate.original_use,
            repurposing_rationale=candidate.repurposing_rationale,
            evidence_tier=getattr(candidate, "evidence_tier", "literature"),
        )
        aggregated.append(agg)

    # Sort by final confidence descending
    aggregated.sort(key=lambda a: a.final_confidence, reverse=True)
    for i, agg in enumerate(aggregated):
        agg.rank = i + 1

    # Categorize
    # PASS with confidence >= 0.40 → recommended
    # WARN with confidence >= 0.45 → also recommended (safety note doesn't disqualify)
    # PASS/WARN below those thresholds → review needed
    # FAIL → rejected
    recommended = [
        a for a in aggregated
        if (a.validation_status == "PASS" and a.final_confidence >= 0.40)
        or (a.validation_status == "WARN" and a.final_confidence >= 0.45)
    ]
    review_needed = [
        a for a in aggregated
        if a not in recommended and a.validation_status != "FAIL"
    ]
    rejected = [a for a in aggregated if a.validation_status == "FAIL"]

    report = AggregationReport(
        query=disease_query,
        total_candidates=len(aggregated),
        recommended=recommended,
        review_needed=review_needed,
        rejected=rejected,
        conflicts_detected=sum(len(a.conflicts) for a in aggregated),
        hallucinations_flagged=sum(len(a.hallucination_flags) for a in aggregated),
        evidence_summary=(
            f"{len(recommended)} recommended, {len(review_needed)} need review, "
            f"{len(rejected)} rejected out of {len(aggregated)} total candidates"
        ),
        methodology=(
            "Multi-source evidence fusion: Literature (40%) + Validation (25%) + "
            "Knowledge Graph (20%) + NER (15%). Conflict detection and hallucination "
            "flagging applied post-fusion."
        ),
    )

    logger.info(
        "Aggregation complete: %d recommended, %d review, %d rejected",
        len(recommended), len(review_needed), len(rejected),
    )

    return report


def report_to_dict(report: AggregationReport) -> dict:
    """Serialize aggregation report to JSON-friendly dict."""
    def _candidate_to_dict(c: AggregatedCandidate) -> dict:
        return {
            "rank": c.rank,
            "drug": c.drug,
            "final_confidence": c.final_confidence,
            "validation_status": c.validation_status,
            "evidence_tier": c.evidence_tier,
            "original_use": c.original_use,
            "repurposing_rationale": c.repurposing_rationale,
            "pmid_count": c.pmid_count,
            "evidence_pmids": c.evidence_pmids,
            "kg_validated": c.kg_validated,
            "ner_confirmed": c.ner_confirmed,
            "known_targets": c.known_targets,
            "known_mechanism": c.known_mechanism,
            "conflicts": c.conflicts,
            "hallucination_flags": c.hallucination_flags,
            "summary": c.summary,
            "explanation": c.explanation,
            "evidence_sources": [
                {
                    "source_type": es.source_type,
                    "confidence": es.confidence,
                    "details": es.details,
                    "pmids": es.pmids,
                }
                for es in c.evidence_sources
            ],
            "reasoning_trace": [
                {
                    "step": rt.step,
                    "action": rt.action,
                    "detail": rt.detail,
                    "source": rt.source,
                }
                for rt in c.reasoning_trace
            ],
        }

    return {
        "query": report.query,
        "total_candidates": report.total_candidates,
        "recommended": [_candidate_to_dict(c) for c in report.recommended],
        "review_needed": [_candidate_to_dict(c) for c in report.review_needed],
        "rejected": [_candidate_to_dict(c) for c in report.rejected],
        "conflicts_detected": report.conflicts_detected,
        "hallucinations_flagged": report.hallucinations_flagged,
        "evidence_summary": report.evidence_summary,
        "methodology": report.methodology,
    }
