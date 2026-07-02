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


def top_k_accuracy(
    list_of_predictions: list[list[str]], list_of_gold_sets: list[set[str]], k: int
) -> float:
    if not list_of_predictions:
        return 0.0
    hits = 0.0
    for preds, gold in zip(list_of_predictions, list_of_gold_sets):
        hits += hit_at_k(preds, gold, k)
    return hits / len(list_of_predictions)


def mean_reciprocal_rank(
    predicted_drugs: list[str], gold_drugs: set[str]
) -> float:
    """MRR — reciprocal rank of the first correct hit (0 if none in top 20)."""
    gold_lower = {g.lower() for g in gold_drugs}
    for rank, drug in enumerate(predicted_drugs[:20], start=1):
        if drug.lower() in gold_lower:
            return 1.0 / rank
    return 0.0


def auroc_from_rankings(
    predicted_drugs: list[str], gold_drugs: set[str]
) -> float:
    """
    Approximate AUROC treating predicted list as a ranked score list.
    Gold drugs are positives; all others are negatives.
    Uses the Wilcoxon-Mann-Whitney formula over the rank positions.
    """
    gold_lower = {g.lower() for g in gold_drugs}
    n = len(predicted_drugs)
    if n == 0:
        return 0.5

    labels = [1 if d.lower() in gold_lower else 0 for d in predicted_drugs]
    scores = [n - i for i in range(n)]  # higher rank → higher score

    pos_scores = [scores[i] for i, l in enumerate(labels) if l == 1]
    neg_scores = [scores[i] for i, l in enumerate(labels) if l == 0]

    if not pos_scores or not neg_scores:
        return 0.5

    concordant = sum(1 for p in pos_scores for q in neg_scores if p > q)
    ties = sum(0.5 for p in pos_scores for q in neg_scores if p == q)
    total = len(pos_scores) * len(neg_scores)
    return (concordant + ties) / total


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text)]


def candidate_groundedness(candidate: dict, docs: list[dict]) -> float:
    pmids = set(str(p) for p in candidate.get("evidence_pmids", []))
    mechanism = str(candidate.get("mechanism", ""))
    if not pmids or not mechanism:
        return 0.0

    doc_text = {
        str(d.get("pmid", "")): (
            str(d.get("title", "")) + " " + str(d.get("abstract", ""))
        ).lower()
        for d in docs
    }

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


def compute_full_metrics(
    predictions_by_disease: dict[str, list[str]],
    gold: dict[str, list[str]],
    ks: list[int] | None = None,
) -> dict:
    """
    Compute Hit@k, Precision@k, nDCG@k, MRR, and AUROC across all diseases.

    Parameters
    ----------
    predictions_by_disease : {disease: [drug1, drug2, ...]} ranked list
    gold : {disease: [gold_drug1, ...]}  — combined approved + repurposed
    """
    if ks is None:
        ks = [1, 3, 5, 10]

    hit_scores = {k: [] for k in ks}
    prec_scores = {k: [] for k in ks}
    ndcg_scores = {k: [] for k in ks}
    mrr_scores = []
    auroc_scores = []

    for disease, preds in predictions_by_disease.items():
        gold_set = {g.lower() for g in gold.get(disease, [])}
        if not gold_set:
            continue

        for k in ks:
            hit_scores[k].append(hit_at_k(preds, gold_set, k))
            top = preds[:k]
            hits = sum(1 for p in top if p.lower() in gold_set)
            prec_scores[k].append(hits / max(1, len(top)))
            rel_vec = [1 if p.lower() in gold_set else 0 for p in preds[:k]]
            ndcg_scores[k].append(ndcg_at_k(rel_vec, k))

        mrr_scores.append(mean_reciprocal_rank(preds, gold_set))
        auroc_scores.append(auroc_from_rankings(preds, gold_set))

    def _avg(lst: list[float]) -> float:
        return round(sum(lst) / len(lst), 3) if lst else 0.0

    return {
        "hit_at_k": {f"Hit@{k}": _avg(hit_scores[k]) for k in ks},
        "precision_at_k": {f"P@{k}": _avg(prec_scores[k]) for k in ks},
        "ndcg_at_k": {f"nDCG@{k}": _avg(ndcg_scores[k]) for k in ks},
        "mrr": _avg(mrr_scores),
        "auroc": _avg(auroc_scores),
    }
