from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drugenix.evaluation import (
    groundedness_summary,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    top_k_accuracy,
)


def _load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Drugenix MVP outputs.")
    parser.add_argument("--predictions", required=True, help="Path to prediction JSONL file.")
    parser.add_argument(
        "--qrels",
        default="",
        help=(
            "Optional retrieval labels JSON file: "
            "{query: {relevant_pmids:[...] , retrieved_pmids:[...]}}"
        ),
    )
    parser.add_argument(
        "--gold-drugs",
        default="",
        help="Optional generation labels JSON file: {query: [gold_drug1, ...]}",
    )
    args = parser.parse_args()

    pred_lines = [
        json.loads(line)
        for line in Path(args.predictions).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    print("=== Groundedness summary ===")
    all_grounded = []
    for item in pred_lines:
        g = groundedness_summary(item)
        all_grounded.extend(g["scores"])
    avg_grounded = sum(all_grounded) / max(1, len(all_grounded))
    print(f"mean_groundedness: {avg_grounded:.3f}")
    print(f"num_candidates: {len(all_grounded)}")

    if args.qrels:
        qrels = _load_json(args.qrels)
        p_at = {1: [], 3: [], 5: [], 10: []}
        r_at = {1: [], 3: [], 5: [], 10: []}
        n_at = {1: [], 3: [], 5: [], 10: []}

        for query, payload in qrels.items():
            retrieved = [str(x) for x in payload.get("retrieved_pmids", [])]
            relevant = {str(x) for x in payload.get("relevant_pmids", [])}
            rel_vector = [1 if p in relevant else 0 for p in retrieved]
            total_rel = len(relevant)

            for k in [1, 3, 5, 10]:
                p_at[k].append(precision_at_k(rel_vector, k))
                r_at[k].append(recall_at_k(rel_vector, total_rel, k))
                n_at[k].append(ndcg_at_k(rel_vector, k))

        print("\n=== Retrieval metrics ===")
        for k in [1, 3, 5, 10]:
            print(f"P@{k}: {sum(p_at[k])/max(1,len(p_at[k])):.3f}")
            print(f"R@{k}: {sum(r_at[k])/max(1,len(r_at[k])):.3f}")
            print(f"nDCG@{k}: {sum(n_at[k])/max(1,len(n_at[k])):.3f}")

    if args.gold_drugs:
        gold = _load_json(args.gold_drugs)
        preds_list: list[list[str]] = []
        gold_list: list[set[str]] = []
        for item in pred_lines:
            query = str(item.get("query", ""))
            preds = [str(c.get("drug", "")) for c in item.get("candidates", []) if c.get("drug")]
            preds_list.append(preds)
            gold_list.append(set(str(x) for x in gold.get(query, [])))

        print("\n=== Generation metrics ===")
        for k in [1, 3, 5, 10]:
            print(f"Top-{k} Accuracy: {top_k_accuracy(preds_list, gold_list, k):.3f}")


if __name__ == "__main__":
    main()
