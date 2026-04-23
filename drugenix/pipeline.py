from __future__ import annotations

from dataclasses import asdict

from drugenix.reasoning import generate_candidates
from drugenix.retrieval import retrieve_pubmed


def run_pipeline(disease_query: str, retmax: int = 40, top_k: int = 8) -> dict:
    retrieved_docs = retrieve_pubmed(disease_query=disease_query, retmax=retmax, top_k=top_k)
    candidates = generate_candidates(disease_query=disease_query, docs=retrieved_docs)

    return {
        "query": disease_query,
        "retrieved": [asdict(d) for d in retrieved_docs],
        "candidates": [asdict(c) for c in candidates],
    }
