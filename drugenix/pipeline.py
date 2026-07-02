from __future__ import annotations

import logging
import os
from dataclasses import asdict

from drugenix.aggregation import aggregate, report_to_dict
from drugenix.knowledge_graph import get_knowledge_graph
from drugenix.ner_extraction import extract_entities_from_docs
from drugenix.reasoning import generate_candidates
from drugenix.retrieval import retrieve_pubmed
from drugenix.validation import validate_candidates, validation_summary_stats

logger = logging.getLogger(__name__)


def run_pipeline(
    disease_query: str,
    retmax: int = 40,
    top_k: int = 8,
    use_semantic_retrieval: bool = True,
    use_ner: bool = True,
    use_kg: bool = True,
) -> dict:
    """
    Full Drugenix agentic pipeline:
      1. Retrieval        - fetch PubMed abstracts (+ optional semantic reranking)
      2. NER Extraction   - extract drug/target entities from abstracts
      3. Reasoning        - generate drug candidates (LLM or heuristic)
      4. Validation       - rule-based plausibility/evidence/safety checks
      5. Knowledge Graph  - validate candidates against drug-target-disease graph
      6. Aggregation      - Master Agent fuses all evidence into final ranking
    """
    # Stage 1 - Retrieval
    retrieved_docs = retrieve_pubmed(disease_query=disease_query, retmax=retmax, top_k=top_k)

    if use_semantic_retrieval and retrieved_docs:
        try:
            from drugenix.semantic_retrieval import semantic_rerank
            retrieved_docs = semantic_rerank(disease_query, retrieved_docs, top_k=top_k)
        except Exception as e:
            logger.warning("Semantic retrieval unavailable, using token-overlap ranking: %s", e)

    # Stage 2 - NER Extraction
    ner_drug_names: set[str] = set()
    ner_result = None
    if use_ner and retrieved_docs:
        try:
            ner_result = extract_entities_from_docs(retrieved_docs)
            ner_drug_names = {d.normalized for d in ner_result.drugs}
        except Exception as e:
            logger.warning("NER extraction unavailable: %s", e)

    # Stage 3 - Reasoning
    candidates = generate_candidates(disease_query=disease_query, docs=retrieved_docs)

    # Stage 4 - Validation Agent
    validation_results = validate_candidates(candidates=candidates, docs=retrieved_docs)
    val_stats = validation_summary_stats(validation_results)

    # Stage 5 - Knowledge Graph validation
    kg_results = []
    if use_kg and candidates:
        try:
            kg = get_knowledge_graph()
            kg_results = kg.validate_candidates(candidates, disease_query)
        except Exception as e:
            logger.warning("Knowledge graph validation unavailable: %s", e)

    # Stage 6 - Aggregation / Master Agent
    agg_report = aggregate(
        disease_query=disease_query,
        candidates=candidates,
        validation_results=validation_results,
        kg_results=kg_results,
        ner_drug_names=ner_drug_names,
        docs=retrieved_docs,
    )

    # Stage 7 - ADMET + ClinicalTrials enrichment (top candidates only)
    top_candidate_names = [c.drug for c in candidates[:top_k]]
    admet_scores: dict = {}
    clinical_trials_data: dict = {}

    try:
        from drugenix.admet import score_candidates
        admet_scores = score_candidates(top_candidate_names)
        logger.info("ADMET scored %d/%d candidates", len(admet_scores), len(top_candidate_names))
    except Exception as e:
        logger.warning("ADMET scoring unavailable: %s", e)

    try:
        from drugenix.clinical_trials import search_trials, get_highest_phase
        for name in top_candidate_names:
            trials = search_trials(name, disease_query)
            clinical_trials_data[name] = {
                "trials": trials,
                "trial_count": len(trials),
                "highest_phase": get_highest_phase(trials),
                "has_trials": len(trials) > 0,
            }
        logger.info("ClinicalTrials enriched %d candidates", len(clinical_trials_data))
    except Exception as e:
        logger.warning("ClinicalTrials enrichment unavailable: %s", e)

    result = {
        "query": disease_query,
        "runtime": {
            "llm_provider": os.getenv("LLM_PROVIDER", "openai"),
            "openai_model": os.getenv("OPENAI_MODEL", ""),
            "gemini_model": os.getenv("GEMINI_MODEL", ""),
            "ollama_model": os.getenv("OLLAMA_MODEL", ""),
            "custom_model": os.getenv("LLM_MODEL", ""),
            "semantic_retrieval": use_semantic_retrieval,
            "ner_enabled": use_ner,
            "kg_enabled": use_kg,
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
        "kg_validation": [
            {
                "drug": kr.drug,
                "disease": kr.disease,
                "has_direct_path": kr.has_direct_path,
                "has_indirect_path": kr.has_indirect_path,
                "path_length": kr.path_length,
                "path_nodes": kr.path_nodes,
                "known_targets": kr.known_targets,
                "known_mechanism": kr.known_mechanism,
                "kg_confidence_boost": kr.kg_confidence_boost,
                "explanation": kr.explanation,
            }
            for kr in kg_results
        ],
        "ner": {
            "drugs_found": [d.normalized for d in ner_result.drugs] if ner_result else [],
            "targets_found": [t.normalized for t in ner_result.targets] if ner_result else [],
            "drug_target_pairs": ner_result.drug_target_pairs if ner_result else [],
        },
        "aggregation": report_to_dict(agg_report),
        "admet": admet_scores,
        "clinical_trials": clinical_trials_data,
    }

    return result
