"""
Drugenix - Mechanism-Similarity Fallback
============================================
Handles the "brand-new / very rare disease with no PubMed history yet"
case: literature retrieval (drugenix.retrieval) returns few or zero
papers, so the normal literature-grounded Reasoning Agent has nothing to
work with.

Instead of returning nothing, this module proposes drugs via
TARGET-OVERLAP reasoning:
  1. Ask Open Targets which genes/proteins are associated with the queried
     disease (works from genetic/clinical evidence even with zero
     published literature — this is exactly the kind of evidence that
     exists before papers get written).
  2. Search the knowledge graph for drugs already known to act on any of
     those same targets (approved for a DIFFERENT disease).
  3. Propose those drugs as repurposing candidates, explicitly tagged
     evidence_tier="mechanistic" so they are never confused with
     literature-backed candidates downstream (aggregation.py caps their
     confidence and skips the "zero PMID = hallucination" penalty that
     would otherwise unfairly wreck them).

If Open Targets/the KG have nothing either (fully offline, or a disease
with zero recorded target associations), falls back to a pure-LLM
mechanistic hypothesis, still tagged the same way, so a network/API
outage degrades to "no candidates" rather than a crash.
"""

from __future__ import annotations

import logging
import re

from drugenix import opentargets
from drugenix.models import DrugCandidate

logger = logging.getLogger(__name__)

MIN_DOCS_FOR_LITERATURE_MODE = 3  # below this, mechanism fallback kicks in

_LLM_SYSTEM_PROMPT = (
    "You are a biomedical reasoning assistant proposing MECHANISTIC "
    "(indirect, target/pathway-overlap based) drug repurposing hypotheses "
    "for a disease with little or no existing PubMed literature — e.g. a "
    "newly emerging or very rare disease. These are explicitly speculative, "
    "not literature-confirmed. Always return valid JSON only."
)


def _targets_for_drug(kg, drug_node: str) -> set[str]:
    targets = set()
    if drug_node not in kg.graph:
        return targets
    for _, v, data in kg.graph.out_edges(drug_node, data=True):
        if data.get("relation") == "targets":
            targets.add(v)
    return targets


def _other_indication(kg, drug_node: str, exclude_disease: str) -> str:
    """Best-effort: find a disease this drug is already linked to in the KG, other than the queried one."""
    exclude = exclude_disease.lower().strip()
    for _, v, data in kg.graph.out_edges(drug_node, data=True):
        node_data = kg.graph.nodes.get(v, {})
        if node_data.get("type") == "disease" and data.get("relation") == "treats" and exclude not in v and v not in exclude:
            return v.title()
    return "Investigational / other indication"


def _kg_target_overlap_candidates(
    disease_query: str, max_candidates: int
) -> list[DrugCandidate]:
    from drugenix.knowledge_graph import get_knowledge_graph

    kg = get_knowledge_graph()
    kg.build()

    resolved = opentargets.resolve_disease(disease_query)
    if not resolved:
        return []

    associated = opentargets.get_associated_targets(resolved["id"], size=15)
    if not associated:
        return []

    disease_name = resolved["name"]
    candidates_by_drug: dict[str, DrugCandidate] = {}

    for assoc in associated:
        target_symbol = assoc["target"].upper()
        score = float(assoc.get("score", 0.0) or 0.0)
        target_node = target_symbol[:20]

        # Find drugs in the KG that target this same gene/protein for ANY disease.
        if target_node not in kg.graph:
            continue
        for u, _, data in kg.graph.in_edges(target_node, data=True):
            if data.get("relation") != "targets":
                continue
            node_data = kg.graph.nodes.get(u, {})
            if node_data.get("type") != "drug":
                continue

            drug_display = u.title()
            if drug_display.lower() in candidates_by_drug:
                continue

            confidence = round(min(0.55, 0.30 + 0.20 * score), 3)
            original_use = _other_indication(kg, u, disease_query)

            candidates_by_drug[drug_display.lower()] = DrugCandidate(
                drug=drug_display,
                mechanism=f"Acts on {target_symbol}, a target genetically/clinically "
                          f"associated with {disease_name} (Open Targets association score={score:.2f}).",
                confidence=confidence,
                evidence_pmids=[],
                original_use=original_use,
                repurposing_rationale=(
                    f"{drug_display} targets {target_symbol}, which shares mechanistic overlap "
                    f"with {disease_name}. Proposed via target-similarity reasoning because "
                    f"direct PubMed literature for this disease is currently thin or absent — "
                    f"treat as a speculative starting point for further investigation, not a "
                    f"literature-confirmed finding."
                ),
                evidence_tier="mechanistic",
            )

            if len(candidates_by_drug) >= max_candidates:
                break
        if len(candidates_by_drug) >= max_candidates:
            break

    return list(candidates_by_drug.values())


def _llm_mechanistic_candidates(disease_query: str, max_candidates: int) -> list[DrugCandidate]:
    """Pure-LLM fallback when the KG/Open Targets path finds nothing (still tagged mechanistic)."""
    from drugenix.reasoning import chat_completion_json

    prompt = f"""
Disease/condition query: {disease_query}

There is little or no PubMed literature available for this specific query
(it may be a newly emerging, very rare, or unusually described disease).

Task:
Propose up to {max_candidates} existing, already-approved drugs that could
plausibly be repurposed based on MECHANISTIC reasoning — shared molecular
targets, pathways, or symptom-relevant physiology with related/similar
conditions. This is explicitly speculative reasoning, not a literature
claim.

Return a JSON object with key "candidates", each item with fields:
- drug (string): one specific approved drug name.
- original_use (string): what it's currently approved for.
- mechanism (string): its known mechanism of action.
- repurposing_rationale (string): why this mechanism could plausibly help
  with "{disease_query}", based on biological reasoning (not literature).
- confidence (number 0-0.5): keep this LOW (max 0.5) since this is
  speculative mechanistic reasoning, not evidence-backed.
""".strip()

    try:
        parsed = chat_completion_json(prompt, system_prompt=_LLM_SYSTEM_PROMPT)
    except Exception as e:
        logger.info("Mechanistic LLM fallback failed for '%s': %s", disease_query, e)
        return []

    out: list[DrugCandidate] = []
    for item in parsed.get("candidates", [])[:max_candidates]:
        raw_name = str(item.get("drug", "")).strip()
        clean_name = re.sub(r"\s*\(.*?\)\s*$", "", raw_name).strip()
        if not clean_name:
            continue
        out.append(DrugCandidate(
            drug=clean_name,
            mechanism=str(item.get("mechanism", "")).strip(),
            confidence=min(0.5, float(item.get("confidence", 0.3) or 0.3)),
            evidence_pmids=[],
            original_use=str(item.get("original_use", "")).strip() or "Investigational / other indication",
            repurposing_rationale=str(item.get("repurposing_rationale", "")).strip(),
            evidence_tier="mechanistic",
        ))
    return out


def generate_mechanistic_candidates(
    disease_query: str, max_candidates: int = 6
) -> list[DrugCandidate]:
    """
    Entry point used by the pipeline when literature retrieval came back
    thin. Tries structured target-overlap reasoning first (deterministic,
    no LLM required); falls back to an LLM-only mechanistic hypothesis if
    that finds nothing. Every candidate returned has evidence_tier=
    "mechanistic" — callers must not treat these as literature-backed.
    """
    try:
        kg_candidates = _kg_target_overlap_candidates(disease_query, max_candidates)
    except Exception as e:
        logger.info("KG target-overlap fallback failed for '%s': %s", disease_query, e)
        kg_candidates = []

    if kg_candidates:
        return kg_candidates[:max_candidates]

    return _llm_mechanistic_candidates(disease_query, max_candidates)
