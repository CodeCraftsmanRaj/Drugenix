"""
Drugenix - Planner Agent
==========================
Decomposes complex or multi-disease queries into sub-tasks,
routes each sub-task through the pipeline, and merges results.

Examples:
  "drugs for Parkinson's or Alzheimer's"
    -> [Task("Parkinson disease"), Task("Alzheimer disease")]
  "Find repurposed drugs for Type 2 diabetes"
    -> [Task("Type 2 diabetes")]
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

_CONJUNCTIONS = re.compile(r"\b(?:and|or|as well as|along with|also|,)\b", re.IGNORECASE)
_DISEASE_CLEANERS = re.compile(
    r"\b(?:find|suggest|show|give me|repurposed|drug|drugs|candidates?|promising|for|from pubmed|with evidence|please)\b",
    re.IGNORECASE,
)


@dataclass
class PlannerTask:
    disease_query: str
    priority: int = 0
    retmax: int = 40
    top_k: int = 8
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[dict] = None
    error: Optional[str] = None


@dataclass
class ExecutionPlan:
    original_query: str
    tasks: list[PlannerTask] = field(default_factory=list)
    is_multi_disease: bool = False
    strategy: str = "single"  # single, parallel, sequential


def _clean_disease_term(term: str) -> str:
    cleaned = _DISEASE_CLEANERS.sub("", term).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"^[,.\s]+|[,.\s]+$", "", cleaned)
    return cleaned


def _split_multi_disease(query: str) -> list[str]:
    """Split a query that mentions multiple diseases."""
    parts = _CONJUNCTIONS.split(query)
    diseases = []
    for part in parts:
        cleaned = _clean_disease_term(part)
        if cleaned and len(cleaned) > 2:
            diseases.append(cleaned)
    return diseases


def _detect_multi_disease(query: str) -> bool:
    lower = query.lower()
    if _CONJUNCTIONS.search(lower):
        parts = _CONJUNCTIONS.split(lower)
        valid_parts = [p.strip() for p in parts if _clean_disease_term(p.strip())]
        return len(valid_parts) >= 2
    return False


def create_plan(query: str, retmax: int = 40, top_k: int = 8) -> ExecutionPlan:
    """Analyze the query and create an execution plan."""
    plan = ExecutionPlan(original_query=query)

    if _detect_multi_disease(query):
        diseases = _split_multi_disease(query)
        plan.is_multi_disease = True
        plan.strategy = "parallel"

        for i, disease in enumerate(diseases):
            plan.tasks.append(PlannerTask(
                disease_query=disease,
                priority=i,
                retmax=retmax,
                top_k=top_k,
            ))

        logger.info(
            "Planner: decomposed '%s' into %d sub-tasks: %s",
            query,
            len(plan.tasks),
            [t.disease_query for t in plan.tasks],
        )
    else:
        plan.strategy = "single"
        cleaned = _clean_disease_term(query)
        plan.tasks.append(PlannerTask(
            disease_query=cleaned if cleaned else query,
            priority=0,
            retmax=retmax,
            top_k=top_k,
        ))

    return plan


def execute_plan(plan: ExecutionPlan, pipeline_fn) -> dict:
    """
    Execute a plan by running each task through the pipeline.

    Parameters
    ----------
    plan : ExecutionPlan
    pipeline_fn : callable
        Function with signature (disease_query, retmax, top_k) -> dict
    """
    if plan.strategy == "single" and len(plan.tasks) == 1:
        task = plan.tasks[0]
        task.status = "running"
        try:
            task.result = pipeline_fn(
                disease_query=task.disease_query,
                retmax=task.retmax,
                top_k=task.top_k,
            )
            task.status = "completed"
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            logger.error("Task failed for '%s': %s", task.disease_query, e)

        return _build_merged_result(plan)

    # Parallel execution for multi-disease queries
    with ThreadPoolExecutor(max_workers=min(len(plan.tasks), 4)) as executor:
        future_to_task = {}
        for task in plan.tasks:
            task.status = "running"
            future = executor.submit(
                pipeline_fn,
                disease_query=task.disease_query,
                retmax=task.retmax,
                top_k=task.top_k,
            )
            future_to_task[future] = task

        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                task.result = future.result()
                task.status = "completed"
            except Exception as e:
                task.status = "failed"
                task.error = str(e)
                logger.error("Task failed for '%s': %s", task.disease_query, e)

    return _build_merged_result(plan)


def _build_merged_result(plan: ExecutionPlan) -> dict:
    """Merge results from all completed tasks into a single unified result."""
    all_retrieved = []
    all_candidates = []
    all_validation_results = []
    all_kg_results = []
    all_ner_drugs: list[str] = []
    all_ner_targets: list[str] = []
    all_ner_pairs: list = []
    sub_results = []

    # Merged aggregation accumulators
    agg_recommended = []
    agg_review_needed = []
    agg_rejected = []
    agg_conflicts = 0
    agg_hallucinations = 0
    agg_summaries = []

    # These used to be silently dropped by the merge entirely — a
    # multi-disease or symptom-derived query would come back with no
    # ADMET/ClinicalTrials data and no data-quality warnings at all, even
    # though each individual sub-task pipeline run computed them.
    merged_admet: dict = {}
    merged_clinical_trials: dict = {}
    merged_data_quality_warnings: list[str] = []
    reasoning_modes: set[str] = set()

    for task in plan.tasks:
        # Include all tasks in sub_results so the banner shows the full decomposition
        sub_results.append({
            "disease_query": task.disease_query,
            "status": task.status,
            "candidate_count": len(task.result.get("candidates", [])) if task.result else 0,
            "retrieved_count": len(task.result.get("retrieved", [])) if task.result else 0,
            "error": task.error,
        })

        if task.status != "completed" or task.result is None:
            continue

        result = task.result

        for doc in result.get("retrieved", []):
            doc["source_query"] = task.disease_query
            all_retrieved.append(doc)

        for candidate in result.get("candidates", []):
            candidate["source_query"] = task.disease_query
            all_candidates.append(candidate)

        validation = result.get("validation", {})
        for vr in validation.get("results", []):
            vr["source_query"] = task.disease_query
            all_validation_results.append(vr)

        for kr in result.get("kg_validation", []):
            kr["source_query"] = task.disease_query
            all_kg_results.append(kr)

        # Merge NER
        ner = result.get("ner", {})
        all_ner_drugs.extend(ner.get("drugs_found", []))
        all_ner_targets.extend(ner.get("targets_found", []))
        all_ner_pairs.extend(ner.get("drug_target_pairs", []))

        # Merge aggregation — tag each candidate with source disease
        agg = result.get("aggregation", {})
        for c in agg.get("recommended", []):
            c["source_query"] = task.disease_query
            agg_recommended.append(c)
        for c in agg.get("review_needed", []):
            c["source_query"] = task.disease_query
            agg_review_needed.append(c)
        for c in agg.get("rejected", []):
            c["source_query"] = task.disease_query
            agg_rejected.append(c)
        agg_conflicts += agg.get("conflicts_detected", 0)
        agg_hallucinations += agg.get("hallucinations_flagged", 0)
        if agg.get("evidence_summary"):
            agg_summaries.append(f"[{task.disease_query}] {agg['evidence_summary']}")

        # Merge ADMET / ClinicalTrials — keyed by drug name, so overlapping
        # drug names across sub-tasks just get overwritten with equivalent data.
        merged_admet.update(result.get("admet", {}))
        merged_clinical_trials.update(result.get("clinical_trials", {}))

        for w in result.get("data_quality_warnings", []):
            merged_data_quality_warnings.append(f"[{task.disease_query}] {w}")

        reasoning_modes.add(result.get("runtime", {}).get("reasoning_mode", "unknown"))

    # Deduplicate across sub-tasks: keep the higher-confidence entry per drug name
    def _dedup(candidates: list[dict]) -> list[dict]:
        seen: dict[str, dict] = {}
        for c in candidates:
            key = c.get("drug", "").lower().strip()
            if key not in seen or c.get("final_confidence", 0) > seen[key].get("final_confidence", 0):
                seen[key] = c
        return list(seen.values())

    agg_recommended = _dedup(agg_recommended)
    agg_review_needed = _dedup(
        [c for c in agg_review_needed if c.get("drug", "").lower() not in {x.get("drug", "").lower() for x in agg_recommended}]
    )
    agg_rejected = _dedup(
        [c for c in agg_rejected if c.get("drug", "").lower() not in {x.get("drug", "").lower() for x in agg_recommended + agg_review_needed}]
    )

    # Re-rank merged recommended/review lists by final_confidence
    agg_recommended.sort(key=lambda c: c.get("final_confidence", 0), reverse=True)
    agg_review_needed.sort(key=lambda c: c.get("final_confidence", 0), reverse=True)

    # Fix ranks across merged list
    for i, c in enumerate(agg_recommended + agg_review_needed + agg_rejected):
        c["rank"] = i + 1

    total_agg = len(agg_recommended) + len(agg_review_needed) + len(agg_rejected)

    # Compute merged validation stats
    total_val = len(all_validation_results)
    passed = sum(1 for r in all_validation_results if r.get("status") == "PASS")
    warned = sum(1 for r in all_validation_results if r.get("status") == "WARN")
    failed = sum(1 for r in all_validation_results if r.get("status") == "FAIL")

    return {
        "query": plan.original_query,
        "planner": {
            "is_multi_disease": plan.is_multi_disease,
            "strategy": plan.strategy,
            "sub_tasks": sub_results,
            "total_tasks": len(plan.tasks),
            "completed_tasks": sum(1 for t in plan.tasks if t.status == "completed"),
            "failed_tasks": sum(1 for t in plan.tasks if t.status == "failed"),
        },
        "retrieved": all_retrieved,
        "candidates": all_candidates,
        "admet": merged_admet,
        "clinical_trials": merged_clinical_trials,
        "data_quality_warnings": merged_data_quality_warnings,
        "runtime": {
            "reasoning_mode": "heuristic" if "heuristic" in reasoning_modes else (
                "llm" if reasoning_modes == {"llm"} else "mixed" if reasoning_modes else "unknown"
            ),
        },
        "validation": {
            "results": all_validation_results,
            "stats": {
                "total": total_val,
                "passed": passed,
                "warned": warned,
                "failed": failed,
                "pass_rate": round(passed / total_val, 3) if total_val else 0.0,
            },
        },
        "kg_validation": all_kg_results,
        "ner": {
            "drugs_found": list(dict.fromkeys(all_ner_drugs)),
            "targets_found": list(dict.fromkeys(all_ner_targets)),
            "drug_target_pairs": all_ner_pairs,
        },
        "aggregation": {
            "recommended": agg_recommended,
            "review_needed": agg_review_needed,
            "rejected": agg_rejected,
            "total_candidates": total_agg,
            "conflicts_detected": agg_conflicts,
            "hallucinations_flagged": agg_hallucinations,
            "evidence_summary": " | ".join(agg_summaries) if agg_summaries else f"{len(agg_recommended)} recommended across {len(sub_results)} diseases",
            "methodology": "Multi-disease parallel execution via Planner Agent. Each sub-query runs the full pipeline independently; results are merged and re-ranked by final confidence.",
        },
    }
