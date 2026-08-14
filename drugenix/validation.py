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

# Negative-trial-outcome phrases — catches a candidate that already
# underwent a clinical trial for this disease and failed, even when that
# trial completed on schedule rather than being stopped early. Pipeline-
# level ClinicalTrials.gov status checks (TERMINATED/WITHDRAWN/SUSPENDED)
# can't see this case — most negative trials run to completion and are
# simply reported as failures in the literature, not cut short. Real
# example this catches: Riluzole underwent a 537-patient, 3-year Phase 3
# RCT for Huntington's disease that completed on schedule and found no
# neuroprotective or beneficial effect — the trial report is in PubMed and
# uses exactly this kind of language, but its ClinicalTrials.gov status is
# just "COMPLETED" like any successful trial would be.
_NEGATIVE_TRIAL_OUTCOME_PHRASES: tuple[str, ...] = (
    "did not meet its primary endpoint",
    "did not meet the primary endpoint",
    "failed to meet the primary endpoint",
    "failed to meet its primary endpoint",
    "no significant difference between",
    "no intergroup difference",
    "no significant improvement",
    "no beneficial effect",
    "no neuroprotective effect",
    "did not show a beneficial effect",
    "failed to demonstrate efficacy",
    "lack of efficacy",
    "no clinical benefit",
    "not superior to placebo",
    "no different from placebo",
    "no different than placebo",
    "trial was discontinued",
    "program was discontinued",
    "terminated early due to futility",
    "did not slow disease progression",
    "did not slow the progression",
    "demonstrated no efficacy",
    "showed no significant effect",
    "negative trial",
)

# Minimum confidence to pass without warning
_CONFIDENCE_THRESHOLD_PASS = 0.45
_CONFIDENCE_THRESHOLD_FAIL = 0.20

# Drug classes that are well-established (textbook pharmacology, not a
# research claim) to be actively inappropriate — not just "unproven" —
# when there is no confirmed diagnosis behind the query, i.e. a symptom
# query that only inferred a likely condition rather than a named one.
# Corticosteroids/immunosuppressants are the clearest case: they suppress
# the immune response and can mask or worsen an undiagnosed infection,
# so surfacing one as a confident "Recommended" result for someone who
# just typed symptoms (no clinician, no confirmed cause) is a real risk,
# regardless of how strong the underlying literature evidence looks.
_HIGH_RISK_UNDIAGNOSED_CLASSES: dict[str, frozenset[str]] = {
    "corticosteroid/immunosuppressant": frozenset({
        "prednisone", "prednisolone", "dexamethasone", "methylprednisolone",
        "hydrocortisone", "betamethasone", "triamcinolone", "cortisone",
        "azathioprine", "mycophenolate", "cyclosporine", "tacrolimus",
        "sirolimus", "everolimus", "methotrexate", "cyclophosphamide",
        "adalimumab", "infliximab", "etanercept", "tofacitinib", "baricitinib",
        # IL-6/IL-1 inhibitors and other biologics/small molecules with the
        # same failure mode — these are the exact drug class used alongside
        # dexamethasone/baricitinib in severe-COVID-19 protocols, and carry
        # the identical risk of masking a new/worsening undiagnosed infection.
        "tocilizumab", "sarilumab", "anakinra", "canakinumab", "rituximab",
        "abatacept", "belimumab", "ustekinumab", "secukinumab", "ixekizumab",
        "upadacitinib", "ruxolitinib", "filgotinib",
    }),
}


def _high_risk_undiagnosed_class(drug_name: str) -> str | None:
    lower = drug_name.lower().strip()
    for label, names in _HIGH_RISK_UNDIAGNOSED_CLASSES.items():
        if lower in names:
            return label
    return None

# Common English words / biomedical generic terms that are NOT drug names.
# These are produced by the heuristic fallback when no LLM is configured.
_STOPWORDS: frozenset[str] = frozenset({
    # Articles / pronouns / conjunctions
    "the", "this", "that", "these", "those", "their", "they", "them",
    "have", "has", "had", "been", "were", "with", "from", "also", "both",
    "since", "while", "when", "where", "which", "what", "into", "onto",
    "more", "most", "some", "such", "only", "each", "other", "than",
    # Common function words / connectives (frequent false positives from the
    # heuristic fallback, which forces title-case on ANY matched word).
    "against", "however", "therefore", "moreover", "furthermore",
    "additionally", "although", "though", "because", "between", "during",
    "without", "through", "before", "after", "further", "again", "once",
    "here", "there", "why", "how", "all", "any", "few", "many", "much",
    "same", "very", "just", "then", "now", "still", "yet", "even",
    "over", "under", "above", "below", "within", "across", "among",
    "toward", "towards", "despite", "regarding", "including", "include",
    "included", "several", "various", "significant", "significantly",
    "increased", "decreased", "reduced", "improved", "compared",
    "group", "groups", "control", "controls", "week", "weeks", "month",
    "months", "year", "years", "dose", "doses", "dosage",
    "administration", "administered", "combination", "combined",
    "single", "multiple", "primary", "secondary", "novel", "current",
    "currently", "previous", "previously", "future", "possible",
    "possibly", "likely", "unlikely", "does", "did", "doing", "done",
    "using", "used", "use", "shown", "showed", "show", "found", "found",
    "either", "neither", "whether", "unless", "until", "upon",
    "thus", "hence", "overall", "notably", "importantly", "interestingly",
    "first", "second", "third", "finally", "next", "last", "well",
    "known", "unknown", "new", "old", "high", "low", "further",
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

    # Reject multi-word generic phrases like "tau inhibitor", "BACE1 inhibitor"
    words = lower.split()
    if len(words) > 2:
        return False
    if len(words) == 2 and words[-1] in {
        "inhibitor", "agonist", "antagonist", "blocker", "activator",
        "drug", "compound", "therapy", "treatment", "agent",
    }:
        return False

    # Common generic drugs list
    if lower in _KNOWN_DRUGS:
        return True

    # INN suffix match
    if any(lower.endswith(sfx) for sfx in _APPROVED_SUFFIXES):
        return True

    # Alphanumeric research compound code like "BMS-986325", "SB-431542"
    if re.match(r"^[A-Za-z]{1,4}[-_]?\d{4,}", name):
        return True

    # Any capitalized word of >= 6 chars that is not a common English word
    # is likely a proper drug name (e.g. Rapamycin, Berberine, Nilotinib,
    # Deferiprone, Simufilam, Masitinib, Crenezumab …)
    if len(lower) >= 6 and lower[0].isalpha():
        # Must contain at least 2 vowels (rules out abbreviations like "BDNF")
        vowels = sum(1 for ch in lower if ch in "aeiou")
        if vowels >= 2:
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


def _scan_negative_trial_outcome(drug_name: str, docs: list[EvidenceDoc], pmids: list[str]) -> list[str]:
    """Return negative-trial-outcome phrases found near this drug's name in
    its cited abstracts. See _NEGATIVE_TRIAL_OUTCOME_PHRASES for why this
    exists as a separate check from _scan_safety_flags and the
    ClinicalTrials.gov status check in pipeline.py."""
    doc_map = {d.pmid: (d.title + " " + d.abstract).lower() for d in docs}
    drug_lower = drug_name.lower()

    triggered: list[str] = []
    for pmid in pmids:
        text = doc_map.get(pmid, "")
        for phrase in _NEGATIVE_TRIAL_OUTCOME_PHRASES:
            idx_drug = text.find(drug_lower)
            idx_phrase = text.find(phrase)
            if idx_phrase == -1:
                continue
            if idx_drug == -1 or abs(idx_drug - idx_phrase) < 300:
                if phrase not in triggered:
                    triggered.append(phrase)

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


# Words too generic to count as a meaningful disease-match when checking
# whether a candidate is already indicated for the queried disease.
_GENERIC_DISEASE_WORDS: frozenset[str] = frozenset({
    "disease", "disorder", "syndrome", "type", "chronic", "acute",
    "condition", "conditions", "cancer", "cancers", "investigational",
    "various", "other", "related", "with", "and", "the", "for",
})


def _disease_tokens(text: str) -> set[str]:
    """Extract significant, crudely-stemmed disease-name tokens from text."""
    words = re.findall(r"[a-zA-Z]{4,}", text.lower())
    return {w.rstrip("s") for w in words if w not in _GENERIC_DISEASE_WORDS}


def _is_circular_candidate(original_use: str, disease_query: str) -> bool:
    """
    True if the candidate's current/original indication overlaps with the
    disease being queried — i.e. it is already used for this disease, so
    proposing it as a "repurposing" candidate is circular/invalid.

    This only catches it if the LLM/heuristic's own self-reported
    "original_use" field happens to mention the queried disease. In
    practice that field is frequently incomplete — e.g. a biologic with
    five approved indications might get reported with only one or two of
    them, and rheumatoid-arthritis-approved TNF inhibitors like Adalimumab
    or Etanercept have been observed coming back with "original_use: Crohn's
    disease / Plaque psoriasis" (both true, both real indications) while
    omitting that they are ALSO already first-line RA drugs. Relying on
    this field alone lets already-standard-of-care drugs slip through as
    if they were novel repurposing hypotheses — see
    _is_known_drug_for_disease for the fix (a live ground-truth check).
    """
    if not original_use or not disease_query:
        return False
    query_tokens = _disease_tokens(disease_query)
    if not query_tokens:
        return False
    use_tokens = _disease_tokens(original_use)
    return bool(query_tokens & use_tokens)


def _known_drugs_for_disease(disease_query: str) -> set[str]:
    """
    Live ground-truth cross-check: drugs Open Targets actually lists as
    known/approved for this disease, independent of whatever the LLM
    self-reported as a candidate's "original_use". Returns a lowercased
    name set; empty (fails open, not closed) if the disease can't be
    resolved or the live lookup fails — a validation feature going dark
    should never block otherwise-valid candidates.
    """
    if not disease_query:
        return set()
    try:
        from drugenix.knowledge_graph import get_knowledge_graph

        kg = get_knowledge_graph()
        return {d.lower().strip() for d in kg.get_known_drugs_for_disease(disease_query) if d}
    except Exception:
        return set()


def _is_known_drug_for_disease(drug_name: str, known_drugs: set[str]) -> bool:
    """True if `drug_name` matches (case-insensitively) a drug Open Targets
    already lists as known/approved for the queried disease — catches
    circularity the LLM's self-reported original_use field missed."""
    if not drug_name or not known_drugs:
        return False
    name = drug_name.lower().strip()
    return name in known_drugs


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
    disease_query: str = "",
    symptom_derived: bool = False,
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
    disease_query : str
        The disease being queried. Used to reject candidates that are
        already approved/indicated for this same disease (circular
        "repurposing" suggestions).
    symptom_derived : bool
        True when this disease_query was itself inferred from a symptom
        description rather than named directly by the user — i.e. there is
        no confirmed diagnosis. Triggers the high-risk-drug-class check
        (see _HIGH_RISK_UNDIAGNOSED_CLASSES): candidates like corticosteroids
        that are inappropriate without a confirmed diagnosis get a hard
        WARN + confidence penalty regardless of how well-evidenced they are.

    Returns
    -------
    list of ValidationResult  (sorted: PASS first, then WARN, then FAIL)
    """
    deduped = _dedup_candidates(candidates)
    results: list[ValidationResult] = []

    # Fetched once per call (not per-candidate) — live Open Targets ground
    # truth for what's already known/approved for this disease, used as a
    # second, independent circularity signal alongside the LLM's
    # self-reported original_use text (see _known_drugs_for_disease).
    known_drugs = _known_drugs_for_disease(disease_query)

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
        # Check 1b – Circular candidate (already indicated for this disease)
        # ------------------------------------------------------------------
        is_circular_self_reported = _is_circular_candidate(candidate.original_use, disease_query)
        is_circular_known_drug = _is_known_drug_for_disease(candidate.drug, known_drugs)
        is_circular = is_circular_self_reported or is_circular_known_drug
        checks["already_indicated_for_query"] = is_circular
        if is_circular_known_drug and not is_circular_self_reported:
            # The LLM reported a different original_use and never flagged
            # this itself — Open Targets' live data is what caught it. Say
            # so explicitly so a researcher can see this wasn't the LLM
            # being self-aware, it was an independent ground-truth check.
            warnings.append(
                f"'{candidate.drug}' is listed by Open Targets as an existing/approved drug "
                f"for '{disease_query}' — not a valid repurposing candidate, even though the "
                "reasoning step reported a different original indication."
            )
            adj_confidence *= 0.30
        elif is_circular:
            warnings.append(
                f"'{candidate.drug}' appears to already be indicated for "
                f"'{disease_query}' — not a valid repurposing candidate "
                "(must currently be approved for a DIFFERENT disease)."
            )
            adj_confidence *= 0.30

        # ------------------------------------------------------------------
        # Check 2 – Evidence grounding (PMID count)
        # ------------------------------------------------------------------
        evidence_level = _evidence_strength(candidate.evidence_pmids)
        checks["evidence_strength"] = evidence_level
        is_mechanistic = getattr(candidate, "evidence_tier", "literature") == "mechanistic"

        if is_mechanistic and evidence_level == "none":
            # Mechanistic-tier candidates are proposed via target/pathway
            # overlap specifically BECAUSE direct literature is thin or
            # absent for this disease — zero PMIDs is expected by design,
            # not a red flag. Don't apply the literature-grounding penalty;
            # aggregation.py separately caps their overall confidence and
            # keeps them ranked below literature-backed candidates.
            warnings.append(
                "Mechanistic/indirect candidate — proposed via target-overlap "
                "reasoning, not direct PubMed literature for this disease."
            )
        elif evidence_level == "none":
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
        # Check 3b – High-risk drug class without a confirmed diagnosis
        # ------------------------------------------------------------------
        high_risk_class = _high_risk_undiagnosed_class(candidate.drug)
        checks["high_risk_undiagnosed_class"] = high_risk_class
        if symptom_derived and high_risk_class:
            warnings.append(
                f"{candidate.drug} is a {high_risk_class} — it can suppress the immune response and "
                "mask or worsen an undiagnosed infection. This query's disease was inferred from "
                "symptoms, not confirmed by a clinician, so this class of drug is flagged regardless "
                "of literature support: do not treat this as symptom relief."
            )
            adj_confidence *= 0.45

        # ------------------------------------------------------------------
        # Check 3c – Negative trial outcome already reported in literature
        # ------------------------------------------------------------------
        # Complements pipeline.py's ClinicalTrials.gov status check (which
        # only catches trials that were stopped early). This catches trials
        # that ran to completion and simply failed — the far more common
        # case — by scanning the candidate's own cited abstracts for
        # explicit negative-outcome language near its name. See
        # _NEGATIVE_TRIAL_OUTCOME_PHRASES for a real example (Riluzole /
        # Huntington's disease) this was built to catch.
        negative_outcome_hits = _scan_negative_trial_outcome(candidate.drug, docs, candidate.evidence_pmids)
        checks["negative_trial_outcome"] = negative_outcome_hits
        if negative_outcome_hits:
            hit_str = "; ".join(negative_outcome_hits[:2])
            warnings.append(
                f"'{candidate.drug}' already has literature-reported evidence of a failed/negative "
                f"clinical outcome for this disease (\"{hit_str}\") — this is not a novel repurposing "
                "hypothesis, it was already tried and did not work. Verify before treating this as a "
                "fresh candidate."
            )
            # Deliberately harsh enough to push almost any candidate below
            # the WARN recommendation threshold (0.45) regardless of how
            # strong the rest of its evidence looked — a documented failed
            # trial for this exact disease should never let a candidate
            # present as a confident top recommendation.
            adj_confidence *= 0.35

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
        if not name_ok or is_circular:
            # Implausible name, or a candidate that's circular (already
            # indicated for the queried disease), is always a hard FAIL —
            # regardless of evidence. Implausible names handle heuristic-mode
            # noise (common words extracted as candidates).
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
