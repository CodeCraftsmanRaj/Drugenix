"""
Drugenix - ClinicalTrials.gov Cross-Reference
===============================================
Queries the ClinicalTrials.gov v2 REST API to check whether predicted
drug-disease pairs have active or completed clinical trials.

API docs: https://clinicaltrials.gov/data-api/api
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

CT_BASE = "https://clinicaltrials.gov/api/v2/studies"
_cache: dict[str, list] = {}


def search_trials(drug: str, disease: str, max_results: int = 3, timeout: int = 8) -> list[dict]:
    """
    Search ClinicalTrials.gov for trials involving this drug + disease pair.
    Returns list of trial summaries (NCT ID, title, phase, status).
    """
    cache_key = f"{drug.lower()}::{disease.lower()}"
    if cache_key in _cache:
        return _cache[cache_key]

    params = urllib.parse.urlencode({
        "query.intr": drug,
        "query.cond": disease,
        "pageSize": max_results,
        "fields": "NCTId,BriefTitle,OverallStatus,Phase,EnrollmentCount,StartDate,CompletionDate",
        "format": "json",
    })
    url = f"{CT_BASE}?{params}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Drugenix/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        logger.debug("ClinicalTrials lookup failed for %s / %s: %s", drug, disease, e)
        return []

    studies = data.get("studies", [])
    results = []
    for study in studies:
        proto = study.get("protocolSection", {})
        id_mod = proto.get("identificationModule", {})
        status_mod = proto.get("statusModule", {})
        design_mod = proto.get("designModule", {})
        enroll = design_mod.get("enrollmentInfo", {})

        nct_id = id_mod.get("nctId", "")
        title = id_mod.get("briefTitle", "")
        status = status_mod.get("overallStatus", "")
        phase_list = design_mod.get("phases", [])
        phase = phase_list[0] if phase_list else "N/A"
        enrollment = enroll.get("count", None)
        start = status_mod.get("startDateStruct", {}).get("date", "")
        completion = status_mod.get("completionDateStruct", {}).get("date", "")

        # Normalize phase string
        phase_display = phase.replace("PHASE", "Phase ").strip()
        if phase_display == "N/A" or not phase_display:
            phase_display = "Phase N/A"

        results.append({
            "nct_id": nct_id,
            "title": title,
            "status": status,
            "phase": phase_display,
            "enrollment": enrollment,
            "start_date": start,
            "completion_date": completion,
            "url": f"https://clinicaltrials.gov/study/{nct_id}",
        })

    _cache[cache_key] = results
    return results


def get_highest_phase(trials: list[dict]) -> Optional[str]:
    """Return the highest clinical trial phase found among a list of trials."""
    phase_order = {
        "Phase 4": 4, "Phase 3": 3, "Phase 2/Phase 3": 3,
        "Phase 2": 2, "Phase 1/Phase 2": 2, "Phase 1": 1,
        "Early Phase 1": 0,
    }
    best = None
    best_rank = -1
    for t in trials:
        phase = t.get("phase", "")
        rank = phase_order.get(phase, -1)
        if rank > best_rank:
            best_rank = rank
            best = phase
    return best


def enrich_candidates_with_trials(
    candidates: list[dict],
    disease: str,
    max_workers: int = 5,
) -> list[dict]:
    """
    Add clinical trial data to each candidate dict in-place.
    candidates: list of dicts with at least a "drug" key.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _fetch(candidate: dict) -> dict:
        drug = candidate.get("drug", "")
        trials = search_trials(drug, disease)
        highest = get_highest_phase(trials)
        return {
            "drug": drug,
            "trials": trials,
            "trial_count": len(trials),
            "highest_phase": highest,
            "has_trials": len(trials) > 0,
        }

    drug_to_trials = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch, c): c.get("drug", "") for c in candidates}
        for future in as_completed(futures):
            try:
                result = future.result()
                drug_to_trials[result["drug"]] = result
            except Exception as e:
                logger.debug("Trial enrichment error: %s", e)

    for candidate in candidates:
        drug = candidate.get("drug", "")
        trial_data = drug_to_trials.get(drug, {"trials": [], "trial_count": 0, "highest_phase": None, "has_trials": False})
        candidate["clinical_trials"] = trial_data

    return candidates
