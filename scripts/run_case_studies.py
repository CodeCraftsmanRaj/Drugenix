"""
Drugenix - Real Case Study Validations
========================================
Runs three landmark drug repurposing cases through the actual Drugenix pipeline
and saves real results to data/case_studies.json.

Case studies:
  1. Metformin    -> Cancer       (AMPK pathway repurposing)
  2. Sildenafil   -> Pulmonary hypertension  (PDE5 mechanism)
  3. Thalidomide  -> Multiple myeloma        (immunomodulator repurposing)

Usage:
  python scripts/run_case_studies.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drugenix.knowledge_graph import get_knowledge_graph
from drugenix.pipeline import run_pipeline

CASE_STUDIES = [
    {
        "id": "metformin_cancer",
        "drug": "Metformin",
        "disease": "Cancer",
        "original_use": "Type 2 diabetes",
        "repurposing_rationale": "AMPK activation inhibits mTOR signaling and cancer cell proliferation",
        "key_mechanism": "AMPK → mTOR inhibition → anti-proliferative",
        "evidence_level": "Phase III clinical trials",
        "query": "Cancer",
    },
    {
        "id": "sildenafil_pah",
        "drug": "Sildenafil",
        "disease": "Pulmonary hypertension",
        "original_use": "Erectile dysfunction",
        "repurposing_rationale": "PDE5 inhibition in pulmonary vasculature causes vasodilation, reducing arterial pressure",
        "key_mechanism": "PDE5 inhibition → cGMP elevation → pulmonary vasodilation",
        "evidence_level": "FDA approved (Revatio, 2005)",
        "query": "Pulmonary hypertension",
    },
    {
        "id": "thalidomide_myeloma",
        "drug": "Thalidomide",
        "disease": "Multiple myeloma",
        "original_use": "Morning sickness (withdrawn 1961)",
        "repurposing_rationale": "Immunomodulatory effects (IMiD) and anti-angiogenic properties suppress myeloma growth",
        "key_mechanism": "Cereblon binding → Ikaros/Aiolos degradation → myeloma apoptosis",
        "evidence_level": "FDA approved (2006) + standard of care",
        "query": "Multiple myeloma",
    },
]

OUT_PATH = PROJECT_ROOT / "data" / "case_studies.json"


def run_case_study(case: dict) -> dict:
    """Run a case study through the Drugenix pipeline and return enriched result."""
    print(f"\n{'='*60}")
    print(f"Case Study: {case['drug']} -> {case['disease']}")
    print(f"Query: '{case['query']}'")
    t0 = time.time()

    try:
        result = run_pipeline(
            disease_query=case["query"],
            retmax=20,
            top_k=8,
            use_kg=True,
            use_ner=True,
        )
        elapsed = time.time() - t0

        agg = result.get("aggregation", {})
        recommended = agg.get("recommended", [])
        review_needed = agg.get("review_needed", [])
        all_candidates = recommended + review_needed + agg.get("rejected", [])

        # Find our specific drug in the results
        drug_lower = case["drug"].lower()
        drug_result = None
        drug_rank = None
        for i, c in enumerate(all_candidates):
            if drug_lower in c.get("drug", "").lower() or c.get("drug", "").lower() in drug_lower:
                drug_result = c
                drug_rank = i + 1
                break

        # KG validation for this specific drug
        kg = get_knowledge_graph()
        kg_val = kg.validate_candidate(case["drug"], case["query"])

        output = {
            **case,
            "pipeline_result": {
                "elapsed_seconds": round(elapsed, 1),
                "total_candidates": len(all_candidates),
                "recommended_count": len(recommended),
                "target_drug_found": drug_result is not None,
                "target_drug_rank": drug_rank,
                "target_drug_confidence": drug_result.get("final_confidence") if drug_result else None,
                "target_drug_kg_validated": drug_result.get("kg_validated", False) if drug_result else False,
                "target_drug_pmid_count": drug_result.get("pmid_count", 0) if drug_result else 0,
                "top_5_candidates": [
                    {
                        "drug": c.get("drug"),
                        "confidence": c.get("final_confidence"),
                        "kg_validated": c.get("kg_validated", False),
                    }
                    for c in all_candidates[:5]
                ],
            },
            "kg_validation": {
                "has_direct_path": kg_val.has_direct_path,
                "has_indirect_path": kg_val.has_indirect_path,
                "path_length": kg_val.path_length,
                "known_targets": kg_val.known_targets[:3],
                "known_mechanism": kg_val.known_mechanism,
                "explanation": kg_val.explanation,
            },
            "status": "success",
        }

        # Print summary
        print(f"  Found {len(all_candidates)} candidates in {elapsed:.1f}s")
        if drug_result:
            print(f"  Target drug '{case['drug']}' found at rank #{drug_rank} (confidence={drug_result.get('final_confidence', 0):.3f})")
        else:
            print(f"  Target drug '{case['drug']}' NOT in top results (may be outside retmax scope)")
        print(f"  KG: {kg_val.explanation}")

    except Exception as e:
        print(f"  ERROR: {e}")
        output = {**case, "status": "error", "error": str(e)}

    return output


def main():
    print("Drugenix Case Study Validations")
    print("=" * 60)
    print(f"Running {len(CASE_STUDIES)} landmark repurposing cases through the pipeline...")

    results = []
    for case in CASE_STUDIES:
        result = run_case_study(case)
        results.append(result)

    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n\nSaved case study results to: {OUT_PATH}")

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Case Study':<30} {'Found':<8} {'Rank':<6} {'KG':<8} {'Status'}")
    print("-" * 70)
    for r in results:
        pr = r.get("pipeline_result", {})
        found = "Yes" if pr.get("target_drug_found") else "No"
        rank = str(pr.get("target_drug_rank") or "-")
        kg = "Direct" if r.get("kg_validation", {}).get("has_direct_path") else (
            "Indirect" if r.get("kg_validation", {}).get("has_indirect_path") else "None"
        )
        status = r.get("status", "unknown")
        label = f"{r['drug']} -> {r['disease']}"
        print(f"  {label:<28} {found:<8} {rank:<6} {kg:<8} {status}")


if __name__ == "__main__":
    main()
