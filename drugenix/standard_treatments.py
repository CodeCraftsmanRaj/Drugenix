"""
Drugenix - Standard Treatment Reference
===========================================
Deliberately separate from the repurposing engine. Everything in
pipeline.py/reasoning.py/validation.py is built to find NON-OBVIOUS
existing drugs for a disease — validation.py's circularity check
explicitly rejects a candidate if it's already indicated for the queried
condition (see _is_circular_candidate). That's correct for a repurposing
tool, but it means the "obvious" answer (paracetamol for fever) never
shows up there by design, which is confusing if you just want to know
what's normally used.

This module is a small curated, textbook-level reference — NOT scored,
NOT ranked, NOT part of the evidence chain — so a symptom query still
surfaces the standard first-line answer for context, alongside (not
instead of) the repurposing candidates.
"""

from __future__ import annotations

# symptom keyword -> [(drug, note), ...]. Deliberately basic/uncontroversial
# pharmacology, not requiring literature lookup.
_SYMPTOM_TREATMENTS: dict[str, list[tuple[str, str]]] = {
    "fever": [
        ("Paracetamol (acetaminophen)", "First-line antipyretic/analgesic for fever."),
        ("Ibuprofen", "NSAID antipyretic/analgesic alternative — avoid if GI or kidney concerns."),
    ],
    "cough": [
        ("Dextromethorphan", "OTC cough suppressant for dry, non-productive cough."),
        ("Guaifenesin", "Expectorant for productive/chesty cough."),
    ],
    "cold": [
        ("Paracetamol (acetaminophen)", "For fever/aches accompanying a common cold."),
        ("Pseudoephedrine", "Oral decongestant for nasal congestion."),
    ],
    "flu": [
        ("Oseltamivir (Tamiflu)", "Prescription antiviral; most effective if started within 48h of symptom onset."),
        ("Paracetamol (acetaminophen)", "For fever/aches."),
    ],
    "influenza": [
        ("Oseltamivir (Tamiflu)", "Prescription antiviral; most effective if started within 48h of symptom onset."),
        ("Paracetamol (acetaminophen)", "For fever/aches."),
    ],
    "sore throat": [
        ("Paracetamol (acetaminophen)", "Pain relief."),
        ("Throat lozenges", "Local symptomatic relief."),
    ],
    "headache": [
        ("Paracetamol (acetaminophen)", "First-line for mild-to-moderate headache."),
        ("Ibuprofen", "NSAID alternative."),
    ],
    "body ache": [
        ("Paracetamol (acetaminophen)", "General analgesic for myalgia."),
        ("Ibuprofen", "NSAID alternative."),
    ],
    "muscle ache": [
        ("Paracetamol (acetaminophen)", "General analgesic for myalgia."),
        ("Ibuprofen", "NSAID alternative."),
    ],
    "congestion": [
        ("Pseudoephedrine", "Oral decongestant."),
        ("Saline nasal spray", "Non-drug option, safe for frequent use."),
    ],
    "runny nose": [
        ("Cetirizine (antihistamine)", "For allergic/rhinitis-related runny nose."),
    ],
    "nausea": [
        ("Ondansetron", "Antiemetic — often prescription-only."),
    ],
    "vomiting": [
        ("Ondansetron", "Antiemetic — often prescription-only."),
    ],
    "diarrhea": [
        ("Loperamide", "Symptomatic control of acute diarrhea — avoid if infective/bloody diarrhea suspected."),
    ],
    "pain": [
        ("Paracetamol (acetaminophen)", "General analgesic, first-line for mild-moderate pain."),
        ("Ibuprofen", "NSAID alternative."),
    ],
    "sneezing": [
        ("Cetirizine (antihistamine)", "For allergy-related sneezing."),
    ],
    "itchy eyes": [
        ("Cetirizine (antihistamine)", "For allergic conjunctivitis-related itching."),
    ],
    "rash": [
        ("Cetirizine (antihistamine)", "For itching associated with an allergic-type rash."),
    ],
}


def get_standard_treatments(symptom_text: str) -> list[dict]:
    """
    Simple keyword match against the free-text symptom description.
    Returns a deduped list of {"drug", "note", "matched_symptom"}.
    Informational reference only — not scored, ranked, or evidence-linked;
    always shown separately from (not mixed into) the repurposing results.
    """
    text = (symptom_text or "").lower()
    seen: dict[str, dict] = {}
    for keyword, entries in _SYMPTOM_TREATMENTS.items():
        if keyword in text:
            for drug, note in entries:
                if drug not in seen:
                    seen[drug] = {"drug": drug, "note": note, "matched_symptom": keyword}
    return list(seen.values())
