"""
Drugenix – Validation Agent
============================
This agent receives the drug candidates produced by the Reasoning Agent and
applies a multi-rule validation pipeline:

  1. Name plausibility check   – is this a recognised drug-like name?
  2. Evidence grounding check  – does the candidate have PMID-backed evidence?
  3. Safety keyword scan       – do the supporting abstracts contain red-flag terms?
  4. Confidence threshold gate – filter candidates with very low confidence.
  5. Cross-candidate dedup     – collapse near-duplicate drug names.

Each candidate gets a status:
  PASS  – all checks OK, confidence maintained or boosted.
  WARN  – minor concern found (safety flag or thin evidence), confidence penalised.
  FAIL  – critical issue (no evidence, extremely low confidence, obviously invalid name).
"""

from __future__ import annotations

import re
from dataclasses import asdict

from drugenix.models import DrugCandidate, EvidenceDoc, ValidationResult

# ---------------------------------------------------------------------------
# Reference data (lightweight, no external downloads required)
# ---------------------------------------------------------------------------

# Common approved drug name suffixes (WHO INN stems)
_APPROVED_SUFFIXES: tuple[str, ...] = (
    "mab",      # monoclonal antibodies
    "nib",      # kinase inhibitors
    "ciclib",   # cyclin-dependent kinase inhibitors
    "parib",    # PARP inhibitors
    "sartan",   # angiotensin II receptor blockers
    "pril",     # ACE inhibitors
    "olol",     # beta-blockers
    "oxetine",  # SSRIs / SNRIs
    "azole",    # antifungals / proton pump inhibitors
    "avir",     # antivirals
    "mycin",    # antibiotics
    "cycline",  # tetracycline-class antibiotics
    "cillin",   # penicillin-class antibiotics
    "vastatin",  # statins
    "prazole",  # proton pump inhibitors
    "dipine",   # calcium channel blockers
    "tidine",   # H2 blockers
    "triptan",  # migraine-specific
    "setron",   # 5-HT3 antagonists
    "lukast",   # leukotriene antagonists
    "glitazone",# PPAR-gamma agonists
    "gliptin",  # DPP-4 inhibitors
    "gliflozin",# SGLT2 inhibitors
    "oxacin",   # fluoroquinolones
    "parin",    # anticoagulants
    "dine",     # antihistamines (many)
    "phylline",  # methylxanthines
    "sterone",  # steroids
    "olone",    # corticosteroids
    "fenac",    # NSAIDs (diclofenac family)
    "profen",   # NSAIDs (ibuprofen family)
    "oxicam",   # NSAIDs (piroxicam family)
)

# Well-known generic drug names that may not match suffix patterns
_KNOWN_DRUGS: frozenset[str] = frozenset({
    "aspirin", "ibuprofen", "paracetamol", "acetaminophen", "metformin",
    "insulin", "warfarin", "heparin", "lithium", "caffeine", "morphine",
    "codeine", "tramadol", "fentanyl", "naloxone", "naltrexone",
    "thalidomide", "rapamycin", "sirolimus", "tacrolimus", "cyclosporine",
    "prednisone", "dexamethasone", "hydrocortisone", "cortisol",
    "tamoxifen", "letrozole", "anastrozole", "fulvestrant", "trastuzumab",
    "bevacizumab", "rituximab", "pembrolizumab", "nivolumab", "ipilimumab",
    "sorafenib", "imatinib", "erlotinib", "gefitinib", "lapatinib",
    "vemurafenib", "dabrafenib", "trametinib", "crizotinib", "alectinib",
    "chloroquine", "hydroxychloroquine", "doxycycline", "azithromycin",
    "amoxicillin", "ciprofloxacin", "metronidazole", "rifampicin",
    "isoniazid", "ethambutol", "pyrazinamide", "streptomycin",
    "oseltamivir", "remdesivir", "lopinavir", "ritonavir", "tenofovir",
    "emtricitabine", "efavirenz", "atazanavir", "dolutegravir",
    "levodopa", "carbidopa", "memantine", "donepezil", "rivastigmine",
    "galantamine", "quetiapine", "olanzapine", "risperidone", "haloperidol",
    "clozapine", "aripiprazole", "sertraline", "fluoxetine", "citalopram",
    "escitalopram", "paroxetine", "venlafaxine", "duloxetine", "bupropion",
    "amitriptyline", "clomipramine", "mirtazapine", "trazodone",
    "diazepam", "alprazolam", "lorazepam", "clonazepam", "zolpidem",
    "atorvastatin", "rosuvastatin", "simvastatin", "pravastatin",
    "amlodipine", "lisinopril", "ramipril", "losartan", "valsartan",
    "carvedilol", "metoprolol", "atenolol", "bisoprolol", "digoxin",
    "furosemide", "spironolactone", "hydrochlorothiazide",
    "omeprazole", "esomeprazole", "lansoprazole", "pantoprazole",
    "ranitidine", "famotidine", "cetirizine", "loratadine", "fexofenadine",
    "montelukast", "salbutamol", "albuterol", "formoterol", "salmeterol",
    "fluticasone", "budesonide", "beclomethasone", "tiotropium",
    "methotrexate", "azathioprine", "mycophenolate", "hydroxychloroquine",
    "colchicine", "allopurinol", "febuxostat",
    "glipizide", "glyburide", "glimepiride", "sitagliptin", "saxagliptin",
    "empagliflozin", "dapagliflozin", "canagliflozin", "liraglutide",
    "semaglutide", "exenatide", "pioglitazone", "rosiglitazone",
    "levothyroxine", "methimazole", "propylthiouracil",
    "alendronate", "risedronate", "zoledronic", "denosumab",
    "sildenafil", "tadalafil", "vardenafil",
    "finasteride", "dutasteride", "tamsulosin", "doxazosin",
    "oxybutynin", "tolterodine", "solifenacin", "mirabegron",
})

# Safety red-flag keywords — presence triggers a WARN
_SAFETY_FLAGS: tuple[str, ...] = (
    "toxicity", "toxic", "hepatotoxic", "nephrotoxic", "cardiotoxic",
    "adverse effect", "side effect", "contraindicated", "contraindication",
    "lethal", "fatal", "carcinogen", "mutagenic", "teratogen",
    "drug interaction", "overdose", "withdrawal", "dependence", "abuse",
    "off-label", "black box", "boxed warning",
)

# Minimum confidence to pass without warning
_CONFIDENCE_THRESHOLD_PASS = 0.45
_CONFIDENCE_THRESHOLD_FAIL = 0.20

# Common English words / biomedical generic terms that are NOT drug names.
# These are produced by the heuristic fallback when no LLM is configured.
_STOPWORDS: frozenset[str] = frozenset({
    # Articles / pronouns / conjunctions
    "the", "this", "that", "these", "those", "their", "they", "them",
    "have", "has", "had", "been", "were", "with", "from", "also", "both",
    "since", "while", "when", "where", "which", "what", "into", "onto",
    "more", "most", "some", "such", "only", "each", "other", "than",
    # Generic biomedical terms (not drug names)
    "disease", "disorder", "syndrome", "condition", "therapy", "treatment",
    "drug", "drugs", "agent", "agents", "compound", "compounds",
    "inhibitor", "inhibitors", "agonist", "agonists", "antagonist", "antagonists",
    "activator", "activators", "blocker", "blockers", "receptor", "receptors",
    "target", "targets", "pathway", "pathways", "mechanism", "mechanisms",
    "study", "studies", "trial", "trials", "review", "reviews",
    "patient", "patients", "clinical", "effect", "effects",
    "recent", "advances", "development", "emerging", "therapeutic",
    "potential", "approach", "approaches", "strategy", "strategies",
    "evidence", "data", "result", "results", "analysis", "report",
    "cancer", "tumor", "tumour", "diabetes", "alzheimer", "parkinson",
    "mellitus", "mellitu", "type", "model", "models", "protein", "proteins",
    "gene", "genes", "cell", "cells", "level", "levels", "activity",
    "expression", "function", "functions", "role", "roles", "system",
    "response", "responses", "signaling", "signalling", "cascade",
    "associated", "related", "induced", "mediated", "dependent",
    "human", "mouse", "animal", "vivo", "vitro", "silico",
})

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _is_plausible_drug_name(name: str) -> bool:
    """Return True if the name looks like a real drug."""
    lower = name.lower().strip()

    # Too short or obviously non-drug (single letter / purely numeric)
    if len(lower) < 4 or lower.isdigit():
        return False

    # Must start with a letter
    if not lower[0].isalpha():
        return False

    # Reject known non-drug stopwords immediately
    if lower in _STOPWORDS:
        return False

    # Common generic drugs list
    if lower in _KNOWN_DRUGS:
        return True

    # INN suffix match
    if any(lower.endswith(sfx) for sfx in _APPROVED_SUFFIXES):
        return True

    # Heuristic: alphanumeric code pattern like "BMS-986325", "SB-431542"
    if re.match(r"^[A-Za-z]{1,4}[-_]?\d{4,}", name):
        return True

    return False


def _scan_safety_flags(drug_name: str, docs: list[EvidenceDoc], pmids: list[str]) -> list[str]:
    """Return a list of triggered safety flag strings found in supporting abstracts."""
    doc_map = {d.pmid: (d.title + " " + d.abstract).lower() for d in docs}
    drug_lower = drug_name.lower()

    triggered: list[str] = []
    for pmid in pmids:
        text = doc_map.get(pmid, "")
        for flag in _SAFETY_FLAGS:
            # Flag must appear near the drug name (within 300 chars) OR be a general finding
            idx_drug = text.find(drug_lower)
            idx_flag = text.find(flag)
            if idx_flag == -1:
                continue
            if idx_drug == -1 or abs(idx_drug - idx_flag) < 300:
                if flag not in triggered:
                    triggered.append(flag)

    return triggered


def _evidence_strength(pmids: list[str]) -> str:
    """Label evidence strength based on number of supporting PMIDs."""
    n = len(pmids)
    if n == 0:
        return "none"
    if n == 1:
        return "weak"
    if n <= 3:
        return "moderate"
    return "strong"


def _dedup_candidates(candidates: list[DrugCandidate]) -> list[DrugCandidate]:
    """Remove near-duplicate drug names (case-insensitive, strip trailing 's')."""
    seen: dict[str, DrugCandidate] = {}
    for c in candidates:
        key = c.drug.lower().rstrip("s ")
        if key not in seen or c.confidence > seen[key].confidence:
            seen[key] = c
    return list(seen.values())


# ---------------------------------------------------------------------------
# Main Validation Agent
# ---------------------------------------------------------------------------

def validate_candidates(
    candidates: list[DrugCandidate],
    docs: list[EvidenceDoc],
    min_confidence: float = _CONFIDENCE_THRESHOLD_FAIL,
) -> list[ValidationResult]:
    """
    Validate a list of DrugCandidate objects produced by the Reasoning Agent.

    Parameters
    ----------
    candidates : list of DrugCandidate
        Output from generate_candidates().
    docs : list of EvidenceDoc
        Retrieved PubMed documents (needed for safety scanning).
    min_confidence : float
        Candidates below this threshold automatically receive FAIL status.

    Returns
    -------
    list of ValidationResult  (sorted: PASS first, then WARN, then FAIL)
    """
    deduped = _dedup_candidates(candidates)
    results: list[ValidationResult] = []

    for candidate in deduped:
        warnings: list[str] = []
        checks: dict = {}
        adj_confidence = candidate.confidence

        # ------------------------------------------------------------------
        # Check 1 – Name plausibility
        # ------------------------------------------------------------------
        name_ok = _is_plausible_drug_name(candidate.drug)
        checks["name_plausible"] = name_ok
        if not name_ok:
            warnings.append(f"'{candidate.drug}' does not resemble a recognised drug name.")
            adj_confidence *= 0.60

        # ------------------------------------------------------------------
        # Check 2 – Evidence grounding (PMID count)
        # ------------------------------------------------------------------
        evidence_level = _evidence_strength(candidate.evidence_pmids)
        checks["evidence_strength"] = evidence_level
        if evidence_level == "none":
            warnings.append("No PMID evidence provided — cannot verify claim.")
            adj_confidence *= 0.40
        elif evidence_level == "weak":
            warnings.append("Only 1 supporting PMID — evidence is thin.")
            adj_confidence *= 0.80

        # ------------------------------------------------------------------
        # Check 3 – Safety keyword scan
        # ------------------------------------------------------------------
        safety_hits = _scan_safety_flags(candidate.drug, docs, candidate.evidence_pmids)
        checks["safety_flags"] = safety_hits
        if safety_hits:
            flag_str = ", ".join(safety_hits[:3])
            warnings.append(f"Safety concern detected in evidence: [{flag_str}].")
            adj_confidence *= max(0.50, 1.0 - 0.10 * len(safety_hits))

        # ------------------------------------------------------------------
        # Check 4 – Confidence threshold
        # ------------------------------------------------------------------
        checks["original_confidence"] = round(candidate.confidence, 3)
        checks["adjusted_confidence"] = round(adj_confidence, 3)

        if candidate.confidence < min_confidence:
            warnings.append(
                f"Original confidence {candidate.confidence:.2f} is below minimum threshold {min_confidence:.2f}."
            )

        # ------------------------------------------------------------------
        # Determine status
        # ------------------------------------------------------------------
        if not name_ok:
            # Implausible name is always a hard FAIL — regardless of evidence.
            # This handles heuristic-mode noise (common words extracted as candidates).
            status = "FAIL"
        elif candidate.confidence < min_confidence:
            status = "FAIL"
        elif warnings:
            status = "WARN"
        else:
            status = "PASS"
            # Small confidence boost for fully validated candidates
            adj_confidence = min(1.0, adj_confidence * 1.05)
            checks["adjusted_confidence"] = round(adj_confidence, 3)

        # ------------------------------------------------------------------
        # Build one-line summary
        # ------------------------------------------------------------------
        evidence_count = len(candidate.evidence_pmids)
        if status == "PASS":
            summary = (
                f"[PASS] {candidate.drug} | "
                f"confidence={adj_confidence:.2f} | "
                f"evidence={evidence_level} ({evidence_count} PMID{'s' if evidence_count != 1 else ''})"
            )
        elif status == "WARN":
            summary = (
                f"[WARN] {candidate.drug} | "
                f"confidence={adj_confidence:.2f} | "
                f"{len(warnings)} concern(s)"
            )
        else:
            summary = (
                f"[FAIL] {candidate.drug} | "
                f"confidence={adj_confidence:.2f} | "
                f"rejected ({warnings[0] if warnings else 'unknown reason'})"
            )

        results.append(
            ValidationResult(
                drug=candidate.drug,
                status=status,
                confidence=candidate.confidence,
                adjusted_confidence=round(adj_confidence, 3),
                checks=checks,
                warnings=warnings,
                summary=summary,
            )
        )

    # Sort: PASS → WARN → FAIL, then by adjusted confidence descending
    status_order = {"PASS": 0, "WARN": 1, "FAIL": 2}
    results.sort(key=lambda r: (status_order[r.status], -r.adjusted_confidence))
    return results


def validation_summary_stats(results: list[ValidationResult]) -> dict:
    """Return aggregate statistics across all validation results."""
    total = len(results)
    passed = sum(1 for r in results if r.status == "PASS")
    warned = sum(1 for r in results if r.status == "WARN")
    failed = sum(1 for r in results if r.status == "FAIL")

    avg_orig = (
        sum(r.confidence for r in results) / total if total else 0.0
    )
    avg_adj = (
        sum(r.adjusted_confidence for r in results) / total if total else 0.0
    )

    all_flags: list[str] = []
    for r in results:
        all_flags.extend(r.checks.get("safety_flags", []))

    top_flags: list[str] = []
    seen_flags: set[str] = set()
    for f in all_flags:
        if f not in seen_flags:
            top_flags.append(f)
            seen_flags.add(f)

    return {
        "total": total,
        "passed": passed,
        "warned": warned,
        "failed": failed,
        "pass_rate": round(passed / total, 3) if total else 0.0,
        "avg_original_confidence": round(avg_orig, 3),
        "avg_adjusted_confidence": round(avg_adj, 3),
        "top_safety_flags": top_flags[:5],
    }
