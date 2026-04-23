from __future__ import annotations

from dataclasses import dataclass


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
