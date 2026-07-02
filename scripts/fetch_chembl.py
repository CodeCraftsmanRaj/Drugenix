"""
Drugenix - ChEMBL Drug-Disease Knowledge Base Builder
=======================================================
Fetches drug-disease associations from ChEMBL's free REST API and
saves them to data/chembl_kg.json for Knowledge Graph integration.

ChEMBL provides:
  - Drug indications (what disease each drug is approved / in trials for)
  - Mechanism of action (drug -> target -> pathway)
  - Phase of evidence (Phase 4 = FDA approved, Phase 3 = late trial, etc.)

No API key required. Rate limit: ~10 requests/sec.

Usage:
  python scripts/fetch_chembl.py              # all 15 diseases
  python scripts/fetch_chembl.py --min-phase 3  # Phase 3+ only (default)
  python scripts/fetch_chembl.py --min-phase 4  # FDA-approved only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urlencode
from urllib.error import URLError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
OUT_PATH = PROJECT_ROOT / "data" / "chembl_kg.json"

# Map our disease names to ChEMBL search terms
DISEASE_SEARCH_TERMS = {
    "Alzheimer disease":                    ["alzheimer disease", "alzheimers disease"],
    "Parkinson disease":                    ["parkinson disease", "parkinsons disease"],
    "Type 2 diabetes":                      ["type 2 diabetes", "diabetes mellitus type 2", "non-insulin-dependent diabetes"],
    "Breast cancer":                        ["breast carcinoma", "breast neoplasm", "breast cancer"],
    "Lung cancer":                          ["lung carcinoma", "lung cancer", "non-small cell lung"],
    "COVID-19":                             ["COVID-19", "SARS-CoV-2"],
    "Rheumatoid arthritis":                 ["rheumatoid arthritis"],
    "Depression":                           ["major depression", "depressive disorder", "depression"],
    "Hypertension":                         ["hypertension", "essential hypertension"],
    "Asthma":                               ["asthma"],
    "Influenza":                            ["influenza", "influenza A"],
    "HIV infection":                        ["HIV infection", "human immunodeficiency virus"],
    "Multiple myeloma":                     ["multiple myeloma"],
    "Colorectal cancer":                    ["colorectal carcinoma", "colorectal cancer", "colon cancer"],
    "Chronic obstructive pulmonary disease":["chronic obstructive pulmonary disease", "COPD"],
}


def _get(url: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            with urlopen(url, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return {}


def fetch_drug_names(chembl_ids: list[str]) -> dict[str, str]:
    """Batch fetch preferred names for a list of ChEMBL molecule IDs."""
    if not chembl_ids:
        return {}
    names = {}
    # ChEMBL allows batch queries with __in filter
    batch_size = 50
    for i in range(0, len(chembl_ids), batch_size):
        batch = chembl_ids[i:i + batch_size]
        ids_str = ",".join(batch)
        url = f"{CHEMBL_BASE}/molecule?molecule_chembl_id__in={ids_str}&format=json&limit={batch_size}"
        try:
            data = _get(url)
            for mol in data.get("molecules", []):
                cid = mol.get("molecule_chembl_id", "")
                name = mol.get("pref_name") or ""
                if not name:
                    synonyms = mol.get("molecule_synonyms") or []
                    name = synonyms[0].get("molecule_synonym", "") if synonyms else ""
                if cid and name:
                    names[cid] = name
        except Exception as e:
            print(f"  Warning: could not fetch names for batch {i}: {e}")
        time.sleep(0.3)
    return names


def fetch_mechanisms(chembl_ids: list[str]) -> dict[str, list[dict]]:
    """Fetch mechanism of action for each drug."""
    if not chembl_ids:
        return {}
    mechs: dict[str, list[dict]] = {}
    batch_size = 50
    for i in range(0, len(chembl_ids), batch_size):
        batch = chembl_ids[i:i + batch_size]
        ids_str = ",".join(batch)
        url = f"{CHEMBL_BASE}/mechanism?molecule_chembl_id__in={ids_str}&format=json&limit=500"
        try:
            data = _get(url)
            for m in data.get("mechanisms", []):
                cid = m.get("molecule_chembl_id", "")
                if cid:
                    mechs.setdefault(cid, []).append({
                        "mechanism": m.get("mechanism_of_action", ""),
                        "target_name": m.get("target_name", ""),
                        "target_chembl_id": m.get("target_chembl_id", ""),
                        "action_type": m.get("action_type", ""),
                    })
        except Exception as e:
            print(f"  Warning: mechanism fetch failed for batch {i}: {e}")
        time.sleep(0.3)
    return mechs


def fetch_indications_for_disease(
    disease_name: str,
    search_terms: list[str],
    min_phase: float,
) -> list[dict]:
    """Fetch all drug indications for a disease from ChEMBL."""
    seen_chembl: set[str] = set()
    results = []

    for term in search_terms:
        encoded = term.replace(" ", "+")
        url = (
            f"{CHEMBL_BASE}/drug_indication"
            f"?efo_term__icontains={encoded}"
            f"&max_phase_for_ind__gte={min_phase}"
            f"&format=json&limit=500"
        )
        try:
            data = _get(url)
            for ind in data.get("drug_indications", []):
                cid = ind.get("molecule_chembl_id", "")
                if cid and cid not in seen_chembl:
                    seen_chembl.add(cid)
                    results.append({
                        "chembl_id": cid,
                        "efo_term": ind.get("efo_term", ""),
                        "mesh_heading": ind.get("mesh_heading", ""),
                        "max_phase": ind.get("max_phase_for_ind", 0),
                    })
        except Exception as e:
            print(f"  Warning: indication fetch failed for '{term}': {e}")
        time.sleep(0.5)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-phase", type=float, default=3.0,
                        help="Minimum clinical phase (3=Phase III, 4=Approved)")
    args = parser.parse_args()

    kg_data: dict[str, list[dict]] = {}
    all_chembl_ids: set[str] = set()

    print(f"Fetching ChEMBL drug indications (Phase >= {args.min_phase})...")
    print("=" * 60)

    for disease, search_terms in DISEASE_SEARCH_TERMS.items():
        print(f"\n{disease}")
        indications = fetch_indications_for_disease(disease, search_terms, args.min_phase)
        print(f"  Found {len(indications)} drug-indication pairs")
        kg_data[disease] = indications
        for ind in indications:
            all_chembl_ids.add(ind["chembl_id"])

    print(f"\nFetching names for {len(all_chembl_ids)} unique molecules...")
    drug_names = fetch_drug_names(list(all_chembl_ids))
    print(f"  Got names for {len(drug_names)} molecules")

    print(f"\nFetching mechanisms for {len(all_chembl_ids)} molecules...")
    mechanisms = fetch_mechanisms(list(all_chembl_ids))
    print(f"  Got mechanisms for {len(mechanisms)} molecules")

    # Build final structured output
    output: dict[str, list[dict]] = {}
    total_pairs = 0
    for disease, indications in kg_data.items():
        disease_drugs = []
        for ind in indications:
            cid = ind["chembl_id"]
            name = drug_names.get(cid, "")
            if not name:
                continue  # skip if we couldn't resolve the name
            mechs = mechanisms.get(cid, [])
            mechanism_str = mechs[0]["mechanism"] if mechs else ""
            target_str = mechs[0]["target_name"] if mechs else ""
            action_str = mechs[0]["action_type"] if mechs else ""

            disease_drugs.append({
                "drug": name,
                "chembl_id": cid,
                "mechanism": mechanism_str,
                "target": target_str,
                "action_type": action_str,
                "max_phase": ind["max_phase"],
                "efo_term": ind["efo_term"],
            })
            total_pairs += 1

        # Sort by phase (highest = most evidence first)
        disease_drugs.sort(key=lambda x: x["max_phase"], reverse=True)
        output[disease] = disease_drugs
        print(f"  {disease}: {len(disease_drugs)} named drugs")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"\nSaved {total_pairs} drug-disease pairs to: {OUT_PATH}")
    print(f"Diseases covered: {len(output)}")
    print(f"Unique drugs (approx): {len(drug_names)}")
    print("\nNow run the backend — the KG will auto-load from this file.")


if __name__ == "__main__":
    main()
