"""
Drugenix - Real Benchmark Evaluation
======================================
Runs Pure LLM vs Pure Graph vs Hybrid on a gold-standard drug-disease dataset.
Uses parallel execution and incremental saves so it can resume after interruption.

Gold standard: data/gold_drugs.json (FDA-approved + known repurposed drugs)
Output:       data/comparative_evaluation.json
              data/metrics_drugenix_mvp.json

Usage:
  python scripts/run_benchmark.py                        # all 15 diseases
  python scripts/run_benchmark.py --diseases 5           # first N diseases
  python scripts/run_benchmark.py --fast                 # retmax=10, top_k=5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drugenix.evaluation import (
    auroc_from_rankings,
    candidate_groundedness,
    compute_full_metrics,
    hit_at_k,
    mean_reciprocal_rank,
)
from drugenix.knowledge_graph import get_knowledge_graph
from drugenix.pipeline import run_pipeline
from drugenix.reasoning import generate_candidates
from drugenix.retrieval import retrieve_pubmed

GOLD_PATH = PROJECT_ROOT / "data" / "gold_drugs.json"
COMP_OUT = PROJECT_ROOT / "data" / "comparative_evaluation.json"
MVP_OUT = PROJECT_ROOT / "data" / "metrics_drugenix_mvp.json"
CACHE_DIR = PROJECT_ROOT / "data" / ".benchmark_cache"


def _gold_drugs(entry: dict) -> list[str]:
    """Combine approved + repurposed into one gold set."""
    return entry.get("approved", []) + entry.get("repurposed", [])


def _cache_path(disease: str, variant: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = disease.lower().replace(" ", "_").replace("/", "_")
    return CACHE_DIR / f"{safe}_{variant}.json"


def _load_cached(disease: str, variant: str):
    p = _cache_path(disease, variant)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def _save_cached(disease: str, variant: str, data):
    _cache_path(disease, variant).write_text(json.dumps(data), encoding="utf-8")


# ─── Variant runners ────────────────────────────────────────────────────────

def run_pure_graph(disease: str) -> list[dict]:
    cached = _load_cached(disease, "pure_graph")
    if cached is not None:
        return cached

    kg = get_knowledge_graph()
    drugs = kg.get_known_drugs_for_disease(disease)
    results = []
    for drug in drugs:
        val = kg.validate_candidate(drug, disease)
        conf = 0.90 if val.has_direct_path else 0.65
        results.append({
            "drug": drug.title(),
            "confidence": conf,
            "groundedness": 1.0 if val.has_direct_path else 0.6,
            "hallucinated": False,
        })
    results.sort(key=lambda r: r["confidence"], reverse=True)
    _save_cached(disease, "pure_graph", results)
    return results


def run_pure_llm(disease: str, retmax: int, top_k: int) -> list[dict]:
    cached = _load_cached(disease, "pure_llm")
    if cached is not None:
        return cached

    docs = retrieve_pubmed(disease_query=disease, retmax=retmax, top_k=top_k)
    candidates = generate_candidates(disease_query=disease, docs=docs)
    candidates.sort(key=lambda c: c.confidence, reverse=True)
    docs_dicts = [{"pmid": d.pmid, "title": d.title, "abstract": d.abstract} for d in docs]
    results = [
        {
            "drug": c.drug,
            "confidence": c.confidence,
            "groundedness": candidate_groundedness(
                {"evidence_pmids": c.evidence_pmids, "mechanism": c.mechanism},
                docs_dicts,
            ),
            "hallucinated": len(c.evidence_pmids) == 0,
        }
        for c in candidates
    ]
    _save_cached(disease, "pure_llm", results)
    return results


def run_hybrid(disease: str, retmax: int, top_k: int) -> list[dict]:
    cached = _load_cached(disease, "hybrid")
    if cached is not None:
        return cached

    result = run_pipeline(disease_query=disease, retmax=retmax, top_k=top_k)
    agg = result["aggregation"]
    recommended_names = {r["drug"] for r in agg["recommended"]}
    all_cands = agg["recommended"] + agg["review_needed"] + agg["rejected"]
    all_cands.sort(key=lambda c: c["final_confidence"], reverse=True)
    results = [
        {
            "drug": c["drug"],
            "confidence": c["final_confidence"],
            "groundedness": 1.0 if c.get("kg_validated") else (0.5 if c.get("pmid_count", 0) > 0 else 0.0),
            "hallucinated": len(c.get("hallucination_flags", [])) > 0,
            "recommended": c["drug"] in recommended_names,
        }
        for c in all_cands
    ]
    _save_cached(disease, "hybrid", results)
    return results


# ─── Evaluation aggregation ──────────────────────────────────────────────────

def evaluate_variant(
    preds_by_disease: dict[str, list[dict]],
    gold: dict[str, dict],
) -> dict:
    ks = [1, 3, 5, 10]
    hit_sc = {k: [] for k in ks}
    prec_sc = {k: [] for k in ks}
    conf_sc = []
    ground_sc = []
    halluc = 0
    total = 0
    mrr_sc = []
    auroc_sc = []

    for disease, preds in preds_by_disease.items():
        gold_set = {g.lower() for g in _gold_drugs(gold.get(disease, {}))}
        pred_names = [p["drug"] for p in preds]

        for k in ks:
            hit_sc[k].append(hit_at_k(pred_names, gold_set, k))
            top = pred_names[:k]
            hits = sum(1 for p in top if p.lower() in gold_set)
            prec_sc[k].append(hits / max(1, len(top)))

        mrr_sc.append(mean_reciprocal_rank(pred_names, gold_set))
        auroc_sc.append(auroc_from_rankings(pred_names, gold_set))

        for p in preds:
            conf_sc.append(p["confidence"])
            ground_sc.append(p["groundedness"])
            total += 1
            if p.get("hallucinated"):
                halluc += 1

    def _avg(lst):
        return round(sum(lst) / len(lst), 3) if lst else 0.0

    return {
        "hit_at_k": {f"Hit@{k}": _avg(hit_sc[k]) for k in ks},
        "precision_at_k": {f"P@{k}": _avg(prec_sc[k]) for k in ks},
        "mrr": _avg(mrr_sc),
        "auroc": _avg(auroc_sc),
        "mean_confidence": _avg(conf_sc),
        "mean_groundedness": _avg(ground_sc),
        "hallucination_rate": round(halluc / max(1, total), 3),
        "total_candidates": total,
    }


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diseases", type=int, default=0, help="Limit to first N diseases (0=all)")
    parser.add_argument("--fast", action="store_true", help="Use retmax=10, top_k=5 for speed")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached results")
    args = parser.parse_args()

    if args.no_cache and CACHE_DIR.exists():
        import shutil
        shutil.rmtree(CACHE_DIR)

    retmax = 10 if args.fast else 25
    top_k = 5 if args.fast else 8

    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    diseases = list(gold.keys())
    if args.diseases:
        diseases = diseases[: args.diseases]

    print(f"\nDrugenix Benchmark — {len(diseases)} diseases, retmax={retmax}, top_k={top_k}")
    print("=" * 60)

    pure_graph_preds: dict[str, list[dict]] = {}
    pure_llm_preds: dict[str, list[dict]] = {}
    hybrid_preds: dict[str, list[dict]] = {}

    # Pure Graph is fast — run immediately
    print("\n[1/3] Pure Graph (KG lookup)...")
    for disease in diseases:
        pure_graph_preds[disease] = run_pure_graph(disease)
        n = len(pure_graph_preds[disease])
        print(f"  {disease}: {n} drugs")

    # Pure LLM and Hybrid — sequential to respect PubMed rate limits
    print(f"\n[2/3] Pure LLM + Hybrid (sequential, respecting PubMed rate limits)...")
    for disease in diseases:
        t0 = time.time()
        try:
            llm = run_pure_llm(disease, retmax, top_k)
            pure_llm_preds[disease] = llm
        except Exception as e:
            print(f"  LLM ERROR {disease}: {e}")
            pure_llm_preds[disease] = []

        try:
            hyb = run_hybrid(disease, retmax, top_k)
            hybrid_preds[disease] = hyb
        except Exception as e:
            print(f"  Hybrid ERROR {disease}: {e}")
            hybrid_preds[disease] = []

        elapsed = time.time() - t0
        print(f"  {disease}: LLM={len(pure_llm_preds[disease])} drugs, Hybrid={len(hybrid_preds[disease])} candidates ({elapsed:.0f}s)")
        # Respect PubMed E-utilities rate limit (3 requests/sec without API key)
        time.sleep(1)

    # Hybrid recommended-only view
    hybrid_rec_preds = {
        d: [p for p in preds if p.get("recommended")]
        for d, preds in hybrid_preds.items()
    }

    print("\n[3/3] Computing metrics...")

    results = {
        "diseases_evaluated": diseases,
        "gold_standard_source": "FDA approved drugs + known repurposed candidates (curated)",
        "evaluation_config": {"retmax": retmax, "top_k": top_k},
        "pure_llm": evaluate_variant(pure_llm_preds, gold),
        "pure_graph": evaluate_variant(pure_graph_preds, gold),
        "hybrid_all_candidates": evaluate_variant(hybrid_preds, gold),
        "hybrid_recommended_only": evaluate_variant(hybrid_rec_preds, gold),
        "raw_predictions": {
            "pure_llm": pure_llm_preds,
            "pure_graph": pure_graph_preds,
            "hybrid": hybrid_preds,
        },
    }
    # Alias for backward compatibility with UI
    results["hybrid"] = results["hybrid_all_candidates"]

    COMP_OUT.parent.mkdir(parents=True, exist_ok=True)
    COMP_OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {COMP_OUT}")

    # Also save MVP metrics file (used by /api/metrics)
    mvp = {
        "retrieval": {
            f"P@{k}": results["hybrid_all_candidates"]["precision_at_k"][f"P@{k}"]
            for k in [1, 3, 5, 10]
        },
        "generation": {
            "mean_groundedness": results["hybrid_all_candidates"]["mean_groundedness"],
            "Top-1 Accuracy": results["hybrid_all_candidates"]["hit_at_k"]["Hit@1"],
            "Top-3 Accuracy": results["hybrid_all_candidates"]["hit_at_k"]["Hit@3"],
            "Top-5 Accuracy": results["hybrid_all_candidates"]["hit_at_k"]["Hit@5"],
            "Top-10 Accuracy": results["hybrid_all_candidates"]["hit_at_k"].get("Hit@10", 0),
            "mrr": results["hybrid_all_candidates"]["mrr"],
            "auroc": results["hybrid_all_candidates"]["auroc"],
        },
        "ablation": {
            "without_rag_correctness": results["pure_llm"]["hit_at_k"]["Hit@5"],
            "without_rag_groundedness": results["pure_llm"]["mean_groundedness"],
            "with_rag_correctness": results["hybrid_recommended_only"]["hit_at_k"]["Hit@5"],
            "with_rag_groundedness": results["hybrid_recommended_only"]["mean_groundedness"],
        },
    }
    MVP_OUT.write_text(json.dumps(mvp, indent=2), encoding="utf-8")
    print(f"Saved: {MVP_OUT}")

    # Print summary table
    print("\n" + "=" * 60)
    print(f"{'Variant':<30} {'Hit@5':>6} {'P@5':>6} {'MRR':>6} {'AUROC':>6} {'Hallu':>6}")
    print("-" * 60)
    for key, label in [
        ("pure_llm", "Pure LLM"),
        ("pure_graph", "Pure Graph"),
        ("hybrid_all_candidates", "Hybrid (All)"),
        ("hybrid_recommended_only", "Hybrid (Recommended)"),
    ]:
        r = results[key]
        print(
            f"{label:<30}"
            f" {r['hit_at_k']['Hit@5']:>6.3f}"
            f" {r['precision_at_k']['P@5']:>6.3f}"
            f" {r['mrr']:>6.3f}"
            f" {r['auroc']:>6.3f}"
            f" {r['hallucination_rate']:>6.3f}"
        )


if __name__ == "__main__":
    main()
