from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class EvidenceDoc:
    pmid: str
    title: str
    abstract: str
    journal: str
    year: str
    retrieval_score: float = 0.0


@dataclass(slots=True)
class DrugCandidate:
    drug: str
    mechanism: str
    confidence: float
    evidence_pmids: list[str]
    original_use: str = ""          # disease/condition it is currently approved for
    repurposing_rationale: str = "" # scientific reason it may work for the queried disease
    # "literature"  - normal path: grounded in retrieved PubMed abstracts.
    # "mechanistic" - fallback path used when direct literature is thin/absent
    #                 (e.g. a brand-new or very rare disease): proposed via
    #                 target/pathway overlap with related diseases instead of
    #                 direct citations. Always ranked below literature-backed
    #                 candidates and never silently mixed in without this tag.
    evidence_tier: str = "literature"


@dataclass
class ValidationResult:
    """Result produced by the Validation Agent for a single drug candidate."""

    drug: str
    status: str          # "PASS" | "WARN" | "FAIL"
    confidence: float    # original confidence from reasoning stage
    adjusted_confidence: float  # confidence after validation adjustments
    checks: dict         # individual check outcomes
    warnings: list[str]  # human-readable warning messages
    summary: str         # one-line verdict for display
