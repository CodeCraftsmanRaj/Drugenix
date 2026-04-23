from __future__ import annotations

import os
from dataclasses import asdict

from drugenix.reasoning import generate_candidates
from drugenix.retrieval import retrieve_pubmed


def run_pipeline(disease_query: str, retmax: int = 40, top_k: int = 8) -> dict:
    retrieved_docs = retrieve_pubmed(disease_query=disease_query, retmax=retmax, top_k=top_k)
    candidates = generate_candidates(disease_query=disease_query, docs=retrieved_docs)

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
    }
