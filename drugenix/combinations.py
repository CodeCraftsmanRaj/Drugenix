"""
Drugenix - Combination Suggestion Agent
===========================================
Researchers repurposing drugs don't just want a ranked list of single
drugs — the real time sink is manually testing every possible PAIRING of
promising candidates to see which combinations are worth pursuing. This
module shortlists that for them.

Given the already-ranked candidate list a query produces (disease query,
multi-disease query, or symptom query — all converge on the same
recommended/review_needed shape by the time they reach the API layer),
it proposes the most promising 2-drug combinations: candidates whose
known molecular targets (from the Knowledge Graph validation stage) or
described mechanisms don't overlap are scored higher, since hitting two
different points in the disease's biology is the standard rationale for
combination therapy — versus two drugs that just duplicate the same
mechanism, which isn't a combination worth pursuing.

Deterministic and KG-driven, not LLM-dependent — it reuses `known_targets`
/ `known_mechanism` that validation + KG stages already computed, so it
works even with no LLM provider configured (heuristic-only mode).
"""

from __future__ import annotations

import itertools
import re


def _target_set(candidate: dict) -> set[str]:
    return {t.upper().strip() for t in (candidate.get("known_targets") or []) if t}


def _mechanism_tokens(candidate: dict) -> set[str]:
    text = candidate.get("known_mechanism") or candidate.get("explanation") or candidate.get("repurposing_rationale") or ""
    stop = {"drug", "drugs", "target", "targets", "mechanism", "disease", "receptor", "inhibitor", "agonist"}
    return {t.lower() for t in re.findall(r"[A-Za-z]{4,}", text) if t.lower() not in stop}


def suggest_combinations(
    candidates: list[dict], max_combinations: int = 5, min_confidence: float = 0.35, pool_size: int = 8
) -> list[dict]:
    """
    candidates: the aggregation "recommended" + "review_needed" dicts
    (each has drug, final_confidence, known_targets, known_mechanism,
    explanation/repurposing_rationale) — same shape drugenix.aggregation
    already produces, no extra computation needed upstream.

    Returns up to `max_combinations` pairs, ranked by a combined score
    that rewards mechanistic complementarity, most-confident pairs first.
    """
    # Dedup by drug name (keep highest-confidence entry), drop anything
    # too low-confidence to be worth combining, cap the search space so
    # this stays O(pool_size^2) instead of scaling with total candidates.
    seen: dict[str, dict] = {}
    for c in candidates:
        name = str(c.get("drug", "")).strip()
        if not name or c.get("final_confidence", 0) < min_confidence:
            continue
        key = name.lower()
        if key not in seen or c.get("final_confidence", 0) > seen[key].get("final_confidence", 0):
            seen[key] = c

    pool = sorted(seen.values(), key=lambda c: c.get("final_confidence", 0), reverse=True)[:pool_size]
    if len(pool) < 2:
        return []

    scored: list[dict] = []
    for a, b in itertools.combinations(pool, 2):
        targets_a, targets_b = _target_set(a), _target_set(b)

        if targets_a and targets_b:
            overlap = targets_a & targets_b
            complementary = not overlap
            basis = "kg_targets"
        else:
            mech_a, mech_b = _mechanism_tokens(a), _mechanism_tokens(b)
            shared = mech_a & mech_b
            smaller = max(1, min(len(mech_a), len(mech_b)))
            complementary = (len(shared) / smaller) < 0.5
            basis = "mechanism_text"

        avg_confidence = (a.get("final_confidence", 0) + b.get("final_confidence", 0)) / 2
        combined_score = round(avg_confidence * (1.15 if complementary else 0.75), 3)

        if targets_a and targets_b:
            rationale = (
                f"{a['drug']} targets {', '.join(sorted(targets_a)) or 'an undetermined target'}; "
                f"{b['drug']} targets {', '.join(sorted(targets_b)) or 'an undetermined target'}. "
                + (
                    "Non-overlapping targets — a mechanistically complementary pairing worth testing together."
                    if complementary else
                    "Overlapping target(s) — likely redundant rather than complementary; combining may not add benefit over either drug alone."
                )
            )
        else:
            rationale = (
                f"{a['drug']} and {b['drug']} have "
                + ("largely distinct" if complementary else "overlapping")
                + " described mechanisms based on current evidence "
                "(no shared Knowledge Graph target data available for this pair, so this is a lower-confidence, "
                "text-based estimate — verify mechanisms manually before pursuing)."
            )

        scored.append({
            "drug_a": a["drug"],
            "drug_b": b["drug"],
            "combined_score": combined_score,
            "complementary_mechanisms": complementary,
            "confidence_basis": basis,
            "confidence_a": round(a.get("final_confidence", 0), 3),
            "confidence_b": round(b.get("final_confidence", 0), 3),
            "targets_a": sorted(targets_a),
            "targets_b": sorted(targets_b),
            "rationale": rationale,
        })

    scored.sort(key=lambda s: s["combined_score"], reverse=True)
    return scored[:max_combinations]
