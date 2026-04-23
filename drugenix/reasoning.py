from __future__ import annotations

import json
import os
import re
import urllib.request
from collections import defaultdict

from drugenix.models import DrugCandidate, EvidenceDoc

_DRUG_SUFFIXES = (
    "mab",
    "nib",
    "ciclib",
    "parib",
    "sartan",
    "pril",
    "oxetine",
    "azole",
    "avir",
    "mycin",
)

_CANDIDATE_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9\-]{2,})\b")


def _build_context_snippets(docs: list[EvidenceDoc], max_chars_per_doc: int = 700) -> str:
    chunks: list[str] = []
    for d in docs:
        abstract = d.abstract.replace("\n", " ").strip()
        abstract = abstract[:max_chars_per_doc]
        chunks.append(
            f"PMID: {d.pmid}\nTitle: {d.title}\nJournal: {d.journal} ({d.year})\nAbstract: {abstract}"
        )
    return "\n\n---\n\n".join(chunks)


def _call_openai_chat(prompt: str) -> list[DrugCandidate]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return []

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a biomedical reasoning assistant. "
                    "Only propose drugs supported by provided evidence snippets."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
    }

    req = urllib.request.Request(
        url=f"{base_url}/chat/completions",
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        data=json.dumps(payload).encode("utf-8"),
    )

    with urllib.request.urlopen(req, timeout=40) as response:
        data = response.read().decode("utf-8")

    response_json = json.loads(data)
    content = response_json["choices"][0]["message"]["content"]
    parsed = json.loads(content)

    out: list[DrugCandidate] = []
    for item in parsed.get("candidates", []):
        out.append(
            DrugCandidate(
                drug=str(item.get("drug", "")).strip(),
                mechanism=str(item.get("mechanism", "")).strip(),
                confidence=float(item.get("confidence", 0.0)),
                evidence_pmids=[str(p) for p in item.get("evidence_pmids", [])],
            )
        )

    return [c for c in out if c.drug]


def _heuristic_reason(docs: list[EvidenceDoc]) -> list[DrugCandidate]:
    evidence_map: dict[str, set[str]] = defaultdict(set)
    mechanism_map: dict[str, list[str]] = defaultdict(list)

    for doc in docs:
        text = f"{doc.title}. {doc.abstract}".replace("\n", " ")
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

        for sentence in sentences:
            for match in _CANDIDATE_RE.finditer(sentence):
                token = match.group(1)
                lower = token.lower()

                if any(lower.endswith(sfx) for sfx in _DRUG_SUFFIXES):
                    name = token[0].upper() + token[1:]
                    evidence_map[name].add(doc.pmid)
                    if len(mechanism_map[name]) < 3:
                        mechanism_map[name].append(sentence[:220])

    candidates: list[DrugCandidate] = []
    for drug, pmids in evidence_map.items():
        support = len(pmids)
        confidence = min(0.95, 0.25 + 0.15 * support)
        mechanism = " ".join(mechanism_map.get(drug, []))
        candidates.append(
            DrugCandidate(
                drug=drug,
                mechanism=mechanism,
                confidence=round(confidence, 2),
                evidence_pmids=sorted(pmids),
            )
        )

    candidates.sort(key=lambda c: (c.confidence, len(c.evidence_pmids)), reverse=True)
    return candidates[:10]


def generate_candidates(disease_query: str, docs: list[EvidenceDoc]) -> list[DrugCandidate]:
    context = _build_context_snippets(docs)
    prompt = f"""
Disease query: {disease_query}

Evidence snippets from PubMed:
{context}

Task:
Return a JSON object with key "candidates" containing up to 8 drug candidates.
Each candidate must include:
- drug (string)
- mechanism (string; concise, evidence-grounded)
- confidence (number between 0 and 1)
- evidence_pmids (array of PMID strings)

Rules:
- Use only provided evidence.
- Prefer repurposing-relevant drugs.
- If evidence is weak, reduce confidence.
""".strip()

    try:
        llm_candidates = _call_openai_chat(prompt)
        if llm_candidates:
            return llm_candidates
    except Exception:
        # Graceful fallback to deterministic heuristic reasoning.
        pass

    return _heuristic_reason(docs)
