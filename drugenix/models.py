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
