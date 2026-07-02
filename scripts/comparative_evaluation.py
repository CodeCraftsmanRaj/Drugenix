"""
Drugenix - Comparative Evaluation: Pure LLM vs Pure Graph vs Hybrid Pipeline
===============================================================================
Fulfills the project proposal's promised experiment (Slide 4):
  "Comparative experiments evaluate: Pure LLM vs Pure Graph vs Hybrid Pipeline."

Three pipeline variants are compared on the same set of disease queries:

  1. Pure LLM    - Retrieval + Reasoning only (no KG, no validation fusion).
  2. Pure Graph  - Knowledge Graph lookup only (no LLM reasoning at all).
  3. Hybrid      - Full agentic pipeline (Retrieval + NER + Reasoning +
                   Validation + Knowledge Graph + Master Agent fusion).

Metrics: Hit@k, Precision@k, Mean Confidence, Groundedness (literature-only checks
are N/A for Pure Graph), and Hallucination Rate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drugenix.evaluation import hit_at_k, candidate_groundedness
from drugenix.knowledge_graph import get_knowledge_graph
from drugenix.pipeline import run_pipeline
from drugenix.reasoning import generate_candidates
from drugenix.retrieval import retrieve_pubmed

GOLD_DRUGS_PATH = PROJECT_ROOT / "data" / "gold_drugs_template.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "comparative_evaluation.json"
PLOT_PATH = PROJECT_ROOT / "plots" / "comparative_evaluation.png"


def run_pure_llm(disease_query: str, retmax: int = 30, top_k: int = 8) -> list[dict]:
    """Retrieval + Reasoning only — no KG, no validation/aggregation fusion."""
    docs = retrieve_pubmed(disease_query=disease_query, retmax=retmax, top_k=top_k)
    candidates = generate_candidates(disease_query=disease_query, docs=docs)
    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return [
        {
            "drug": c.drug,
            "confidence": c.confidence,
            "groundedness": candidate_groundedness(
                {"evidence_pmids": c.evidence_pmids, "mechanism": c.mechanism},
                [{"pmid": d.pmid, "title": d.title, "abstract": d.abstract} for d in docs],
            ),
            "hallucinated": len(c.evidence_pmids) == 0,
        }
        for c in candidates
    ]


def run_pure_graph(disease_query: str) -> list[dict]:
    """Knowledge Graph lookup only — no LLM reasoning involved at all."""
    kg = get_knowledge_graph()
    drugs = kg.get_known_drugs_for_disease(disease_query)

    results = []
    for drug in drugs:
        validation = kg.validate_candidate(drug, disease_query)
        confidence = 0.9 if validation.has_direct_path else 0.6
        results.append({
            "drug": drug.title(),
            "confidence": confidence,
            "groundedness": 1.0 if validation.has_direct_path else 0.6,
            "hallucinated": False,  # KG-only candidates are graph-grounded by definition
        })

    results.sort(key=lambda r: r["confidence"], reverse=True)
    return results


def run_hybrid(disease_query: str, retmax: int = 30, top_k: int = 8) -> list[dict]:
    """Full agentic pipeline with Master Agent evidence fusion.

    Returns ALL scored candidates (recommended + review + rejected) so that
    ranking metrics like Hit@k / P@k see the full ranked list, matching how
    Pure LLM and Pure Graph are evaluated.
    """
    result = run_pipeline(disease_query=disease_query, retmax=retmax, top_k=top_k)
    agg = result["aggregation"]
    all_candidates = agg["recommended"] + agg["review_needed"] + agg["rejected"]
    all_candidates.sort(key=lambda c: c["final_confidence"], reverse=True)
    return [
        {
            "drug": c["drug"],
            "confidence": c["final_confidence"],
            "groundedness": 1.0 if c["kg_validated"] else (0.5 if c["pmid_count"] > 0 else 0.0),
            "hallucinated": len(c["hallucination_flags"]) > 0,
            "recommended": c["drug"] in {r["drug"] for r in agg["recommended"]},
        }
        for c in all_candidates
    ]


def evaluate_variant(predictions_by_disease: dict[str, list[dict]], gold: dict[str, list[str]]) -> dict:
    ks = [1, 3, 5, 8]
    hit_scores = {k: [] for k in ks}
    precision_scores = {k: [] for k in ks}
    mean_confidences = []
    groundedness_scores = []
    hallucination_count = 0
    total_candidates = 0

    for disease, preds in predictions_by_disease.items():
        gold_set = {g.lower() for g in gold.get(disease, [])}
        pred_names = [p["drug"] for p in preds]

        for k in ks:
            hit_scores[k].append(hit_at_k(pred_names, gold_set, k))
            top = pred_names[:k]
            hits = sum(1 for p in top if p.lower() in gold_set)
            precision_scores[k].append(hits / max(1, len(top)))

        for p in preds:
            mean_confidences.append(p["confidence"])
            groundedness_scores.append(p["groundedness"])
            total_candidates += 1
            if p["hallucinated"]:
                hallucination_count += 1

    def _avg(lst):
        return round(sum(lst) / len(lst), 3) if lst else 0.0

    return {
        "hit_at_k": {f"Hit@{k}": _avg(hit_scores[k]) for k in ks},
        "precision_at_k": {f"P@{k}": _avg(precision_scores[k]) for k in ks},
        "mean_confidence": _avg(mean_confidences),
        "mean_groundedness": _avg(groundedness_scores),
        "hallucination_rate": round(hallucination_count / max(1, total_candidates), 3),
        "total_candidates": total_candidates,
    }


def main() -> None:
    gold = json.loads(GOLD_DRUGS_PATH.read_text(encoding="utf-8"))
    diseases = list(gold.keys())

    print(f"Running comparative evaluation on {len(diseases)} diseases: {diseases}\n")

    pure_llm_preds, pure_graph_preds, hybrid_preds = {}, {}, {}

    for disease in diseases:
        print(f"--- {disease} ---")

        print("  Running Pure LLM...")
        pure_llm_preds[disease] = run_pure_llm(disease)

        print("  Running Pure Graph...")
        pure_graph_preds[disease] = run_pure_graph(disease)

        print("  Running Hybrid...")
        hybrid_preds[disease] = run_hybrid(disease)

    # "Recommended-only" view of Hybrid output — this is what a user actually
    # sees and acts on (the Master Agent already filtered out review/rejected
    # candidates), so it is the fairer real-world comparison point.
    hybrid_recommended_preds = {
        disease: [p for p in preds if p.get("recommended")]
        for disease, preds in hybrid_preds.items()
    }

    results = {
        "diseases_evaluated": diseases,
        "pure_llm": evaluate_variant(pure_llm_preds, gold),
        "pure_graph": evaluate_variant(pure_graph_preds, gold),
        "hybrid_all_candidates": evaluate_variant(hybrid_preds, gold),
        "hybrid_recommended_only": evaluate_variant(hybrid_recommended_preds, gold),
        "raw_predictions": {
            "pure_llm": pure_llm_preds,
            "pure_graph": pure_graph_preds,
            "hybrid": hybrid_preds,
        },
    }
    # Backward-compatible alias used by the plotting function.
    results["hybrid"] = results["hybrid_all_candidates"]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved comparative evaluation to: {OUTPUT_PATH}")

    print("\n=== Summary ===")
    for variant in ["pure_llm", "pure_graph", "hybrid_all_candidates", "hybrid_recommended_only"]:
        r = results[variant]
        print(f"\n{variant.upper()}")
        print(f"  Hit@5: {r['hit_at_k']['Hit@5']}  P@5: {r['precision_at_k']['P@5']}")
        print(f"  Mean confidence: {r['mean_confidence']}  Groundedness: {r['mean_groundedness']}")
        print(f"  Hallucination rate: {r['hallucination_rate']}  (n={r['total_candidates']})")

    print(
        "\nNote: 'hybrid_all_candidates' scores every candidate the Master Agent ever "
        "considered (including ones it already rejected) — useful for seeing the raw "
        "ranking. 'hybrid_recommended_only' scores just the final Recommended list, "
        "which is what an actual user of the system sees and acts on."
    )

    _plot_comparison(results)


def _plot_comparison(results: dict) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot.")
        return

    plt.style.use("default")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 11,
    })

    variants = ["pure_llm", "pure_graph", "hybrid"]
    labels = ["Pure LLM", "Pure Graph", "Hybrid Pipeline"]
    colors = ["#ff7f0e", "#2ca02c", "#1f77b4"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    hit5 = [results[v]["hit_at_k"]["Hit@5"] for v in variants]
    axes[0].bar(labels, hit5, color=colors)
    axes[0].set_title("Hit@5")
    axes[0].set_ylim(0, 1)

    groundedness = [results[v]["mean_groundedness"] for v in variants]
    axes[1].bar(labels, groundedness, color=colors)
    axes[1].set_title("Mean Groundedness")
    axes[1].set_ylim(0, 1)

    halluc = [results[v]["hallucination_rate"] for v in variants]
    axes[2].bar(labels, halluc, color=colors)
    axes[2].set_title("Hallucination Rate")
    axes[2].set_ylim(0, 1)

    fig.suptitle("Comparative Evaluation: Pure LLM vs Pure Graph vs Hybrid Pipeline")
    fig.tight_layout()

    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plot to: {PLOT_PATH}")


if __name__ == "__main__":
    main()
