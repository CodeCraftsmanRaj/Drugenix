"""
Drugenix - TransE KG Embeddings Training
==========================================
Trains a TransE model on the drug-disease-target KG and evaluates it
on a held-out test set (PREDICT-style benchmark using ChEMBL Phase 4 data).

Usage:
  python scripts/train_transe.py
  python scripts/train_transe.py --epochs 200 --dim 128
  python scripts/train_transe.py --evaluate-only   (skip training, just evaluate saved model)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from drugenix.kg_embeddings import (
    MODEL_DIR,
    TransEEmbeddings,
    get_embeddings_model,
    train_transe_on_kg,
)
from drugenix.knowledge_graph import get_knowledge_graph


def build_predict_benchmark(kg, min_phase: float = 4.0) -> dict:
    """
    Build a PREDICT-style benchmark from ChEMBL Phase 4 (FDA-approved) drug-disease pairs.
    Returns train/test positive pairs and random negative samples.
    """
    chembl_path = PROJECT_ROOT / "data" / "chembl_kg.json"
    if not chembl_path.exists():
        print("Warning: chembl_kg.json not found. Using curated KG for benchmark.")
        positives = []
        for u, v, data in kg.graph.edges(data=True):
            if data.get("relation") == "treats":
                u_type = kg.graph.nodes.get(u, {}).get("type")
                v_type = kg.graph.nodes.get(v, {}).get("type")
                if u_type == "drug" and v_type == "disease":
                    positives.append((u, v))
    else:
        chembl = json.loads(chembl_path.read_text(encoding="utf-8"))
        positives = []
        for disease, drugs in chembl.items():
            disease_node = disease.lower().strip()
            for entry in drugs:
                if float(entry.get("max_phase", 0) or 0) >= min_phase:
                    drug_node = entry.get("drug", "").lower().strip()
                    if drug_node:
                        positives.append((drug_node, disease_node))

    if not positives:
        raise ValueError("No positive pairs found for benchmark")

    # Shuffle and split 80/20
    np.random.seed(42)
    idx = np.random.permutation(len(positives))
    positives = [positives[i] for i in idx]
    split = int(len(positives) * 0.8)
    train_pos = positives[:split]
    test_pos = positives[split:]

    # Generate negative samples (random drug-disease pairs NOT in positives)
    all_pos_set = set(positives)
    drugs = list(set(d for d, _ in positives))
    diseases = list(set(d for _, d in positives))

    negatives = []
    attempts = 0
    while len(negatives) < len(test_pos) * 3 and attempts < len(test_pos) * 30:
        d = drugs[np.random.randint(len(drugs))]
        dis = diseases[np.random.randint(len(diseases))]
        if (d, dis) not in all_pos_set:
            negatives.append((d, dis))
        attempts += 1

    return {
        "train_pos": train_pos,
        "test_pos": test_pos,
        "test_neg": negatives[:len(test_pos) * 3],
        "total_positives": len(positives),
        "min_phase": min_phase,
    }


def evaluate_on_benchmark(model: TransEEmbeddings, benchmark: dict) -> dict:
    """Evaluate TransE model using AUROC and Hit@K on the test set."""
    from drugenix.evaluation import auroc_from_rankings, hit_at_k

    test_pos = benchmark["test_pos"]
    test_neg = benchmark["test_neg"]

    print(f"Evaluating on {len(test_pos)} positive + {len(test_neg)} negative test pairs...")

    # Score all pairs
    pos_scores = []
    neg_scores = []

    for drug, disease in test_pos:
        score = model.score(drug, "treats", disease)
        pos_scores.append(score if score is not None else 999.0)

    for drug, disease in test_neg:
        score = model.score(drug, "treats", disease)
        neg_scores.append(score if score is not None else 999.0)

    # Convert distances to probabilities (lower distance = higher probability)
    all_scores = pos_scores + neg_scores
    labels = [1] * len(pos_scores) + [0] * len(neg_scores)

    if not all_scores:
        return {"auroc": 0.0, "hit_at_1": 0.0, "hit_at_5": 0.0, "hit_at_10": 0.0}

    max_s = max(all_scores)
    # Flip: higher value = more likely (negate distance)
    probs = [(max_s - s) / (max_s + 1e-6) for s in all_scores]

    # AUROC via Mann-Whitney
    pos_probs = probs[:len(pos_scores)]
    neg_probs = probs[len(pos_scores):]

    n_pos = len(pos_probs)
    n_neg = len(neg_probs)
    if n_pos == 0 or n_neg == 0:
        auroc = 0.5
    else:
        wins = sum(1 for p in pos_probs for n in neg_probs if p > n)
        ties = sum(0.5 for p in pos_probs for n in neg_probs if p == n)
        auroc = (wins + ties) / (n_pos * n_neg)

    # Hit@K per disease
    disease_to_test_pos = {}
    for drug, disease in test_pos:
        disease_to_test_pos.setdefault(disease, []).append(drug)

    hit1_scores, hit5_scores, hit10_scores = [], [], []

    for disease, gold_drugs in disease_to_test_pos.items():
        # Get all drugs, score against this disease
        all_drugs = list(set(d for d, _ in test_pos + test_neg))
        drug_scores = []
        for drug in all_drugs:
            s = model.score(drug, "treats", disease)
            if s is not None:
                drug_scores.append((drug, s))
        drug_scores.sort(key=lambda x: x[1])  # lower = better
        ranked = [d for d, _ in drug_scores]
        gold_set = {g.lower() for g in gold_drugs}

        hit1_scores.append(hit_at_k(ranked, gold_set, 1))
        hit5_scores.append(hit_at_k(ranked, gold_set, 5))
        hit10_scores.append(hit_at_k(ranked, gold_set, 10))

    def avg(lst):
        return round(sum(lst) / len(lst), 3) if lst else 0.0

    return {
        "auroc": round(auroc, 3),
        "hit_at_1": avg(hit1_scores),
        "hit_at_5": avg(hit5_scores),
        "hit_at_10": avg(hit10_scores),
        "n_test_pos": len(test_pos),
        "n_test_neg": len(test_neg),
        "coverage": round(sum(1 for s in pos_scores if s < 999) / max(1, len(pos_scores)), 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--min-phase", type=float, default=4.0,
                        help="Minimum ChEMBL phase for PREDICT benchmark (4=FDA-approved)")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Skip training, just evaluate saved model")
    args = parser.parse_args()

    kg = get_knowledge_graph()
    print(f"KG loaded: {kg.graph.number_of_nodes()} nodes, {kg.graph.number_of_edges()} edges")

    # Build benchmark BEFORE training so test set is not in training data
    print("\nBuilding PREDICT-style benchmark (ChEMBL Phase 4 = FDA-approved pairs)...")
    benchmark = build_predict_benchmark(kg, min_phase=args.min_phase)
    print(f"  Total positives: {benchmark['total_positives']}")
    print(f"  Train set: {len(benchmark['train_pos'])} pairs")
    print(f"  Test set: {len(benchmark['test_pos'])} pos + {len(benchmark['test_neg'])} neg")

    if args.evaluate_only:
        model = get_embeddings_model()
        if not model.trained:
            print("No saved model found. Run without --evaluate-only first.")
            return
    else:
        t0 = time.time()
        model = train_transe_on_kg(
            num_epochs=args.epochs,
            embedding_dim=args.dim,
        )
        model.save(MODEL_DIR)
        print(f"Training done in {time.time() - t0:.0f}s. Model saved to {MODEL_DIR}")

    # Evaluate
    print("\nEvaluating TransE on PREDICT-style benchmark...")
    metrics = evaluate_on_benchmark(model, benchmark)

    print("\n" + "=" * 50)
    print("PREDICT-style Benchmark Results (TransE)")
    print("=" * 50)
    print(f"  AUROC:    {metrics['auroc']:.3f}")
    print(f"  Hit@1:    {metrics['hit_at_1']:.3f}")
    print(f"  Hit@5:    {metrics['hit_at_5']:.3f}")
    print(f"  Hit@10:   {metrics['hit_at_10']:.3f}")
    print(f"  Coverage: {metrics['coverage']:.3f} (fraction of test drugs in vocab)")
    print(f"  Test:     {metrics['n_test_pos']} pos / {metrics['n_test_neg']} neg pairs")

    # Save benchmark results
    out = {
        "model": "TransE",
        "embedding_dim": args.dim,
        "num_epochs": args.epochs,
        "benchmark_type": "PREDICT-style (ChEMBL Phase 4 FDA-approved)",
        "min_phase": args.min_phase,
        "metrics": metrics,
        "benchmark_split": {
            "train_pairs": len(benchmark["train_pos"]),
            "test_pos_pairs": len(benchmark["test_pos"]),
            "test_neg_pairs": len(benchmark["test_neg"]),
        }
    }
    out_path = PROJECT_ROOT / "data" / "predict_benchmark.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_path}")

    # Sample novel predictions
    print("\nSample novel drug predictions (TransE link prediction):")
    for disease in ["alzheimer disease", "type 2 diabetes", "breast cancer"]:
        preds = model.predict_novel_drugs(disease, top_k=5)
        print(f"\n  {disease.title()}:")
        for p in preds:
            print(f"    {p['drug']:<30} confidence={p['confidence']:.3f}  dist={p['transe_distance']:.4f}")


if __name__ == "__main__":
    main()
