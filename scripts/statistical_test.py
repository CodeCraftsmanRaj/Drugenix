"""
Drugenix - Statistical Significance Verification
===================================================
Calculates paired metric distributions (Reciprocal Rank and Precision@5) 
across all evaluated diseases to compute Wilcoxon signed-rank p-values.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drugenix.evaluation import mean_reciprocal_rank

COMP_JSON = PROJECT_ROOT / "data" / "comparative_evaluation.json"
GOLD_JSON = PROJECT_ROOT / "data" / "gold_drugs.json"

def calculate_ap_at_k(preds: list[str], gold_set: set[str], k: int = 5) -> float:
    top = preds[:k]
    hits = sum(1 for p in top if p.lower() in gold_set)
    return hits / max(1, len(top))

def main():
    if not COMP_JSON.exists() or not GOLD_JSON.exists():
        print("Error: Run scripts/run_benchmark.py first to generate comparative_evaluation.json.")
        return

    comp = json.loads(COMP_JSON.read_text(encoding="utf-8"))
    gold = json.loads(GOLD_JSON.read_text(encoding="utf-8"))
    
    raw = comp.get("raw_predictions", {})
    diseases = comp.get("diseases_evaluated", [])
    
    if not diseases:
        print("Error: No diseases found in comparative evaluation.")
        return

    mrr_llm, mrr_graph, mrr_hybrid = [], [], []
    ap_llm, ap_graph, ap_hybrid = [], [], []

    print(f"Loaded comparative evaluations for {len(diseases)} diseases.")
    
    for dis in diseases:
        gold_entry = gold.get(dis, {})
        gold_set = {g.lower() for g in (gold_entry.get("approved", []) + gold_entry.get("repurposed", []))}
        
        preds_llm = [p["drug"] for p in raw.get("pure_llm", {}).get(dis, [])]
        preds_graph = [p["drug"] for p in raw.get("pure_graph", {}).get(dis, [])]
        # Evaluate against the final recommended Hybrid list
        preds_hybrid = [p["drug"] for p in raw.get("hybrid", {}).get(dis, [])]

        # Reciprocal Rank per disease
        mrr_llm.append(mean_reciprocal_rank(preds_llm, gold_set))
        mrr_graph.append(mean_reciprocal_rank(preds_graph, gold_set))
        mrr_hybrid.append(mean_reciprocal_rank(preds_hybrid, gold_set))

        # Precision@5 per disease
        ap_llm.append(calculate_ap_at_k(preds_llm, gold_set, k=5))
        ap_graph.append(calculate_ap_at_k(preds_graph, gold_set, k=5))
        ap_hybrid.append(calculate_ap_at_k(preds_hybrid, gold_set, k=5))

    print("\n=== Wilcoxon Signed-Rank Test Results (Paired) ===")
    
    def run_test(hybrid_scores, baseline_scores, baseline_label, metric_label):
        # Ties are ignored in Wilcoxon, if all are equal it raises ValueError
        if hybrid_scores == baseline_scores:
            print(f"  {metric_label}: Hybrid vs {baseline_label} -> No difference (exact ties).")
            return
        stat, p_val = wilcoxon(hybrid_scores, baseline_scores, alternative="greater")
        print(f"  {metric_label}: Hybrid vs {baseline_label} -> p-value = {p_val:.5f} (statistic={stat:.1f})")

    run_test(mrr_hybrid, mrr_llm, "Pure LLM", "Reciprocal Rank (RR)")
    run_test(mrr_hybrid, mrr_graph, "Pure Graph", "Reciprocal Rank (RR)")
    run_test(ap_hybrid, ap_llm, "Pure LLM", "Precision@5 (P@5)")
    run_test(ap_hybrid, ap_graph, "Pure Graph", "Precision@5 (P@5)")

if __name__ == "__main__":
    main()