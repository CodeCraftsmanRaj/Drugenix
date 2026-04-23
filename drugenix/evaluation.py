from __future__ import annotations

import math
import re
from statistics import mean


def precision_at_k(relevances: list[int], k: int) -> float:
    top = relevances[:k]
    if not top:
        return 0.0
    return sum(1 for r in top if r > 0) / len(top)


def recall_at_k(relevances: list[int], total_relevant: int, k: int) -> float:
    if total_relevant <= 0:
        return 0.0
    top = relevances[:k]
    hits = sum(1 for r in top if r > 0)
    return hits / total_relevant


def ndcg_at_k(relevances: list[int], k: int) -> float:
    def dcg(vals: list[int]) -> float:
        total = 0.0
        for idx, rel in enumerate(vals, start=1):
            total += (2**rel - 1) / math.log2(idx + 1)
        return total

    top = relevances[:k]
    if not top:
        return 0.0

    ideal = sorted(top, reverse=True)
    denom = dcg(ideal)
    if denom == 0:
        return 0.0
    return dcg(top) / denom


def hit_at_k(predicted_drugs: list[str], gold_drugs: set[str], k: int) -> float:
    top = {d.lower() for d in predicted_drugs[:k]}
    return 1.0 if any(g.lower() in top for g in gold_drugs) else 0.0


def top_k_accuracy(list_of_predictions: list[list[str]], list_of_gold_sets: list[set[str]], k: int) -> float:
    if not list_of_predictions:
        return 0.0
    hits = 0.0
    for preds, gold in zip(list_of_predictions, list_of_gold_sets):
        hits += hit_at_k(preds, gold, k)
    return hits / len(list_of_predictions)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text)]


def candidate_groundedness(candidate: dict, docs: list[dict]) -> float:
    pmids = set(str(p) for p in candidate.get("evidence_pmids", []))
    mechanism = str(candidate.get("mechanism", ""))
    if not pmids or not mechanism:
        return 0.0

    doc_text = {str(d.get("pmid", "")): (str(d.get("title", "")) + " " + str(d.get("abstract", ""))).lower() for d in docs}

    tokens = _tokenize(mechanism)
    if not tokens:
        return 0.0

    supported = 0
    for t in tokens[:25]:
        if any(t in doc_text.get(pmid, "") for pmid in pmids):
            supported += 1

    return supported / max(1, min(25, len(tokens)))


def groundedness_summary(result_json: dict) -> dict:
    docs = result_json.get("retrieved", [])
    cands = result_json.get("candidates", [])
    scores = [candidate_groundedness(c, docs) for c in cands]
    return {
        "count": len(scores),
        "mean": mean(scores) if scores else 0.0,
        "min": min(scores) if scores else 0.0,
        "max": max(scores) if scores else 0.0,
        "scores": scores,
    }
