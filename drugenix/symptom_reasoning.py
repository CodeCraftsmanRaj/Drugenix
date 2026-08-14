"""
Drugenix - Symptom Interpretation Agent
==========================================
Lets a user type symptoms instead of a disease name and still get
drug-repurposing candidates.

Pipeline:
  1. LLM reads the symptom description and proposes ranked candidate
     conditions (diagnoses are hypotheses, never asserted as fact).
  2. Each candidate condition is grounded against live PubMed literature
     that co-occurs with those symptom terms, so a candidate with zero
     supporting literature is clearly distinguished from one with real
     evidence, instead of trusting the LLM's guess blindly.
  3. The caller (drugenix.pipeline.run_symptom_pipeline) takes the
     top-ranked, evidence-grounded condition(s) and runs them through the
     normal disease -> drug-repurposing pipeline.

This is explicitly NOT a diagnostic tool. Every result carries a
disclaimer field the frontend must surface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from drugenix.reasoning import chat_completion_json
from drugenix.retrieval import retrieve_pubmed

logger = logging.getLogger(__name__)

MEDICAL_DISCLAIMER = (
    "This is an AI-generated hypothesis for research purposes only, not a medical "
    "diagnosis. The drugs below are repurposing candidates for the condition the AI "
    "guessed from your symptoms — not symptom relief, and not a treatment recommendation. "
    "Some may be inappropriate or unsafe without a confirmed diagnosis (e.g. drugs that "
    "suppress the immune system can mask or worsen an undiagnosed infection). Always "
    "consult a qualified clinician for actual diagnosis or treatment decisions."
)

_SYSTEM_PROMPT = (
    "You are a biomedical research assistant helping identify plausible "
    "disease/condition candidates from a symptom description, for a drug-"
    "repurposing research tool. You are NOT diagnosing a patient. "
    "Always return valid JSON only."
)


@dataclass
class SymptomCandidateCondition:
    condition: str
    llm_confidence: float
    rationale: str
    pubmed_hits: int = 0
    grounded_confidence: float = 0.0
    top_pmids: list[str] = field(default_factory=list)


def _build_prompt(symptom_text: str, max_candidates: int) -> str:
    return f"""
Symptom description (from a researcher, not a patient consultation):
"{symptom_text}"

Task:
Propose up to {max_candidates} plausible disease/condition candidates that
could present with these symptoms. This includes common conditions AND,
if the symptom pattern is unusual or doesn't cleanly match a known
disease, candidate conditions that resemble an emerging/novel disease
profile (label those clearly as "possible novel/atypical presentation").

Return a JSON object with key "conditions", a list of up to {max_candidates}
objects, each with:
- condition (string): a specific, searchable disease/condition name
  (e.g. "Rheumatoid arthritis", "Community-acquired pneumonia"), OR, if
  none fit well, a short descriptive label prefixed with "Novel/atypical:"
  (e.g. "Novel/atypical: acute multi-organ inflammatory syndrome").
- confidence (number 0-1): your confidence this condition explains the
  symptoms.
- rationale (string): 1-2 sentences connecting the symptoms to this
  condition.

Order by confidence, descending.
""".strip()


def interpret_symptoms(
    symptom_text: str, max_candidates: int = 5
) -> list[SymptomCandidateCondition]:
    """Back-compat wrapper. See interpret_symptoms_with_meta."""
    candidates, _meta = interpret_symptoms_with_meta(symptom_text, max_candidates)
    return candidates


def interpret_symptoms_with_meta(
    symptom_text: str, max_candidates: int = 5
) -> tuple[list[SymptomCandidateCondition], dict]:
    """
    Same as interpret_symptoms, but also returns {"llm_error": str | None} so
    a caller can tell "no LLM configured" apart from "the LLM call failed
    for a specific reason" (bad API key, network error, wrong model name,
    provider outage, ...) instead of both collapsing into the same silent
    empty list. A researcher staring at "0 candidates, no LLM available"
    with no further detail has no way to self-diagnose a misconfigured key
    vs. a real outage vs. Google rejecting the request outright — this is
    the same failure mode reasoning.generate_candidates_with_meta already
    fixed for the disease-query path.
    """
    prompt = _build_prompt(symptom_text, max_candidates)
    try:
        parsed = chat_completion_json(prompt, system_prompt=_SYSTEM_PROMPT)
    except Exception as e:
        llm_error = f"{type(e).__name__}: {e}"
        logger.warning("Symptom interpretation LLM call failed: %s", llm_error)
        return [], {"llm_error": llm_error}

    out: list[SymptomCandidateCondition] = []
    for item in parsed.get("conditions", []):
        name = str(item.get("condition", "")).strip()
        if not name:
            continue
        out.append(
            SymptomCandidateCondition(
                condition=name,
                llm_confidence=float(item.get("confidence", 0.0) or 0.0),
                rationale=str(item.get("rationale", "")).strip(),
            )
        )

    if not out:
        return [], {"llm_error": "LLM returned no usable candidate conditions — not configured, no API key, or empty/malformed response."}

    return out[:max_candidates], {"llm_error": None}


def ground_with_pubmed(
    symptom_text: str, candidates: list[SymptomCandidateCondition], retmax: int = 8
) -> list[SymptomCandidateCondition]:
    """
    For each candidate condition, run a live PubMed search combining the
    symptom text and the condition name, so we know whether real
    literature backs the LLM's guess — not just LLM confidence in
    isolation. grounded_confidence blends LLM confidence with literature
    support; a candidate with 0 PubMed hits gets pulled down hard.

    Runs all candidates' searches concurrently (same pattern as
    drugenix.admet / drugenix.clinical_trials) instead of one at a time —
    this step alone was adding several sequential PubMed round trips
    before the main pipeline even started.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _search(cand: SymptomCandidateCondition):
        query = f"{cand.condition} {symptom_text}"
        try:
            return cand, retrieve_pubmed(query, retmax=retmax, top_k=5)
        except Exception as e:
            logger.info("PubMed grounding failed for '%s': %s", cand.condition, e)
            return cand, []

    if not candidates:
        return candidates

    with ThreadPoolExecutor(max_workers=min(5, len(candidates))) as pool:
        futures = [pool.submit(_search, c) for c in candidates]
        for future in as_completed(futures):
            cand, docs = future.result()
            cand.pubmed_hits = len(docs)
            cand.top_pmids = [d.pmid for d in docs]

            if cand.pubmed_hits == 0:
                lit_factor = 0.15  # heavy penalty, but don't zero out — could be a genuinely novel disease
            elif cand.pubmed_hits <= 2:
                lit_factor = 0.6
            else:
                lit_factor = 1.0

            cand.grounded_confidence = round(min(1.0, cand.llm_confidence * lit_factor + 0.05 * min(cand.pubmed_hits, 5)), 3)

    candidates.sort(key=lambda c: c.grounded_confidence, reverse=True)
    return candidates


def analyze_symptoms(symptom_text: str, max_candidates: int = 5) -> dict:
    """
    Full symptom-analysis step: interpret -> ground -> return a
    JSON-serializable report. Does NOT run the drug-repurposing pipeline
    itself; drugenix.pipeline.run_symptom_pipeline does that with the
    top-ranked condition(s) from this report.
    """
    candidates, meta = interpret_symptoms_with_meta(symptom_text, max_candidates=max_candidates)
    llm_available = bool(candidates)

    candidates = ground_with_pubmed(symptom_text, candidates)

    return {
        "symptom_text": symptom_text,
        "disclaimer": MEDICAL_DISCLAIMER,
        "llm_available": llm_available,
        "llm_error": meta.get("llm_error"),
        "candidate_conditions": [
            {
                "condition": c.condition,
                "llm_confidence": round(c.llm_confidence, 3),
                "pubmed_hits": c.pubmed_hits,
                "grounded_confidence": c.grounded_confidence,
                "rationale": c.rationale,
                "top_pmids": c.top_pmids,
                "is_novel_presentation": c.condition.lower().startswith("novel/atypical"),
            }
            for c in candidates
        ],
    }
