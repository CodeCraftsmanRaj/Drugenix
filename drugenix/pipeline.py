from __future__ import annotations

import os
from dataclasses import asdict

from drugenix.reasoning import generate_candidates
from drugenix.retrieval import retrieve_pubmed
from drugenix.validation import validate_candidates, validation_summary_stats


def run_pipeline(disease_query: str, retmax: int = 40, top_k: int = 8) -> dict:
    """
    Full Drugenix pipeline:
      1. Retrieval  – fetch & rerank PubMed abstracts
      2. Reasoning  – generate drug candidates (LLM or heuristic)
      3. Validation – cross-check candidates for plausibility, evidence & safety
    """
    # Stage 1 – Retrieval
    retrieved_docs = retrieve_pubmed(disease_query=disease_query, retmax=retmax, top_k=top_k)

    # Stage 2 – Reasoning
    candidates = generate_candidates(disease_query=disease_query, docs=retrieved_docs)

    # Stage 3 – Validation Agent
    validation_results = validate_candidates(candidates=candidates, docs=retrieved_docs)
    val_stats = validation_summary_stats(validation_results)

    return {
        "query": disease_query,
        "runtime": {
            "llm_provider": os.getenv("LLM_PROVIDER", "openai"),
            "openai_model": os.getenv("OPENAI_MODEL", ""),
            "gemini_model": os.getenv("GEMINI_MODEL", ""),
            "ollama_model": os.getenv("OLLAMA_MODEL", ""),
            "custom_model": os.getenv("LLM_MODEL", ""),
        },
        "retrieved": [asdict(d) for d in retrieved_docs],
        "candidates": [asdict(c) for c in candidates],
        "validation": {
            "results": [
                {
                    "drug": r.drug,
                    "status": r.status,
                    "confidence": r.confidence,
                    "adjusted_confidence": r.adjusted_confidence,
                    "checks": r.checks,
                    "warnings": r.warnings,
                    "summary": r.summary,
                }
                for r in validation_results
            ],
            "stats": val_stats,
        },
    }
