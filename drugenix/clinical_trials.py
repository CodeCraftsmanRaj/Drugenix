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


def search_trials(drug: str, disease: str, max_results: int = 3, timeout: int = 8) -> Optional[list[dict]]:
    """
    Search ClinicalTrials.gov for trials involving this drug + disease pair.
    Returns list of trial summaries (NCT ID, title, phase, status), or an
    empty list if the query succeeded but genuinely found none.

    Returns None (not []) when the API call itself failed (network error,
    timeout, malformed response) — this is a distinct outcome from "zero
    trials exist" and callers should not treat the two the same way. For a
    clinical-trial-shortlisting tool, silently rendering "no trials" for a
    drug that actually has Phase 3 data — just because ClinicalTrials.gov
    timed out — is a correctness bug, not a cosmetic one.
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
        return None

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


def get_trial_result_outcome(nct_id: str, timeout: int = 10) -> Optional[dict]:
    """
    Best-effort fetch of a completed trial's PRIMARY outcome measure result,
    to distinguish "trial completed and worked" from "trial completed and
    failed" — ClinicalTrials.gov's overallStatus alone can't tell the two
    apart; both are simply "COMPLETED" (this is the case that slipped past
    the status-only check: Riluzole/Huntington's and Zolbetuximab/pancreatic
    cancer both ran their full-length trials and reported negative primary
    endpoints, neither trial was stopped early).

    Returns None if the API call itself fails. Otherwise returns a dict —
    this is a best-effort heuristic reading of messy, inconsistently
    structured real-world trial data, NOT a guaranteed-accurate outcome
    classifier. Always show the NCT ID/link alongside whatever this returns
    so a researcher can verify directly rather than trusting an automated
    significant/non-significant label blindly.

    Returns
    -------
    dict | None
        {
          "has_results": bool,
          "primary_outcome_title": str,
          "primary_outcome_significant": bool | None,  # None = couldn't determine
          "p_value": str | None,
        }
    """
    if not nct_id:
        return None

    url = f"{CT_BASE}/{nct_id}?format=json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Drugenix/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        logger.debug("ClinicalTrials result fetch failed for %s: %s", nct_id, e)
        return None

    has_results = bool(data.get("hasResults", False))
    if not has_results:
        return {
            "has_results": False,
            "primary_outcome_title": "",
            "primary_outcome_significant": None,
            "p_value": None,
        }

    try:
        results_section = data.get("resultsSection", {}) or {}
        outcome_module = results_section.get("outcomeMeasuresModule", {}) or {}
        outcomes = outcome_module.get("outcomeMeasures", []) or []
        primary = next((o for o in outcomes if o.get("type") == "PRIMARY"), None)
        if primary is None and outcomes:
            primary = outcomes[0]
        if primary is None:
            return {
                "has_results": True,
                "primary_outcome_title": "",
                "primary_outcome_significant": None,
                "p_value": None,
            }

        title = primary.get("title", "")
        significant: Optional[bool] = None
        p_value_str: Optional[str] = None

        for analysis in (primary.get("analyses", []) or []):
            p_val = analysis.get("pValue")
            if not p_val:
                continue
            p_value_str = str(p_val)
            # p-values arrive as strings like "0.045", "<0.001", "P=0.24",
            # ">0.99" — strip comparison/label characters before parsing.
            # If it's an inequality like "<0.001" the true value could be
            # even smaller/larger than what we parse, but the direction
            # (significant vs not) is still reliable across that edge case.
            cleaned = (
                p_value_str.replace("<", "").replace(">", "")
                .replace("=", "").replace("P", "").replace("p", "").strip()
            )
            try:
                significant = float(cleaned) < 0.05
            except ValueError:
                pass
            break  # only the first analysis on the primary outcome

        return {
            "has_results": True,
            "primary_outcome_title": title,
            "primary_outcome_significant": significant,
            "p_value": p_value_str,
        }
    except Exception as e:
        logger.debug("Failed to parse results section for %s: %s", nct_id, e)
        return {
            "has_results": True,
            "primary_outcome_title": "",
            "primary_outcome_significant": None,
            "p_value": None,
        }


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
        lookup_failed = trials is None
        trials = trials or []
        highest = get_highest_phase(trials)
        return {
            "drug": drug,
            "trials": trials,
            "trial_count": len(trials),
            "highest_phase": highest,
            "has_trials": len(trials) > 0,
            "lookup_failed": lookup_failed,
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
