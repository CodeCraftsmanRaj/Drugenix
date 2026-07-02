"""
Drugenix - ADMET Property Scoring
===================================
Uses PubChem REST API to fetch computed molecular properties and evaluate
drug-likeness via Lipinski Rule of 5 + Veber oral bioavailability criteria.

No RDKit required — all properties fetched directly from PubChem.
"""

from __future__ import annotations

import logging
import urllib.parse
import urllib.request
import json
from typing import Optional

logger = logging.getLogger(__name__)

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name"
PROPERTIES = "MolecularWeight,XLogP,HBondDonorCount,HBondAcceptorCount,RotatableBondCount,TPSA"

_cache: dict[str, dict] = {}


def get_admet_properties(drug_name: str, timeout: int = 6) -> Optional[dict]:
    """
    Fetch ADMET-relevant molecular properties from PubChem for a drug name.
    Returns a dict with Lipinski/Veber properties + pass/fail verdict.
    Returns None if drug not found in PubChem.
    """
    key = drug_name.lower().strip()
    if key in _cache:
        return _cache[key]

    encoded = urllib.parse.quote(key)
    url = f"{PUBCHEM_BASE}/{encoded}/property/{PROPERTIES}/JSON"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Drugenix/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        props = data["PropertyTable"]["Properties"][0]
    except Exception as e:
        logger.debug("PubChem lookup failed for %s: %s", drug_name, e)
        return None

    mw = float(props.get("MolecularWeight", 0))
    logp = float(props.get("XLogP", 0))
    hbd = int(props.get("HBondDonorCount", 0))
    hba = int(props.get("HBondAcceptorCount", 0))
    rotb = int(props.get("RotatableBondCount", 0))
    tpsa = float(props.get("TPSA", 0))

    # Lipinski Rule of 5 (oral drug-likeness)
    ro5_violations = sum([
        mw > 500,
        logp > 5,
        hbd > 5,
        hba > 10,
    ])
    ro5_pass = ro5_violations == 0
    ro5_borderline = ro5_violations == 1

    # Veber criteria (oral bioavailability)
    veber_pass = rotb <= 10 and tpsa <= 140

    if ro5_pass and veber_pass:
        verdict = "pass"
        verdict_label = "Drug-like"
        verdict_color = "success"
    elif ro5_borderline:
        verdict = "borderline"
        verdict_label = "Borderline"
        verdict_color = "warn"
    else:
        verdict = "fail"
        verdict_label = "Poor drug-likeness"
        verdict_color = "fail"

    result = {
        "drug": drug_name,
        "molecular_weight": round(mw, 1),
        "logp": round(logp, 2),
        "hbd": hbd,
        "hba": hba,
        "rotatable_bonds": rotb,
        "tpsa": round(tpsa, 1),
        "ro5_violations": ro5_violations,
        "ro5_pass": ro5_pass,
        "veber_pass": veber_pass,
        "verdict": verdict,
        "verdict_label": verdict_label,
        "verdict_color": verdict_color,
        "pubchem_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{urllib.parse.quote(key)}",
    }

    _cache[key] = result
    return result


def score_candidates(drug_names: list[str], max_workers: int = 6) -> dict[str, dict]:
    """
    Fetch ADMET scores for a list of drug names concurrently.
    Returns dict mapping drug_name -> admet_result.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(get_admet_properties, name): name for name in drug_names}
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                if result:
                    results[name] = result
            except Exception as e:
                logger.debug("ADMET scoring failed for %s: %s", name, e)
    return results
