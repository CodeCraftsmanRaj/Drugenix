from __future__ import annotations

import json
import os
import re
import urllib.error
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


def _normalize_base_url(url: str) -> str:
    base = url.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def _build_context_snippets(docs: list[EvidenceDoc], max_chars_per_doc: int = 700) -> str:
    chunks: list[str] = []
    for d in docs:
        abstract = d.abstract.replace("\n", " ").strip()
        abstract = abstract[:max_chars_per_doc]
        chunks.append(
            f"PMID: {d.pmid}\nTitle: {d.title}\nJournal: {d.journal} ({d.year})\nAbstract: {abstract}"
        )
    return "\n\n---\n\n".join(chunks)


def _resolve_provider_config() -> dict[str, str]:
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()

    if provider in {"openai", "open-ai"}:
        return {
            "provider": "openai",
            "api_key": os.getenv("OPENAI_API_KEY", "").strip(),
            "base_url": _normalize_base_url(os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")),
            "model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        }

    if provider in {"gemini", "gemmini", "google"}:
        # Gemini OpenAI-compatible endpoint.
        return {
            "provider": "gemini",
            "api_key": os.getenv("GEMINI_API_KEY", "").strip(),
            "base_url": _normalize_base_url(
                os.getenv(
                    "GEMINI_BASE_URL",
                    "https://generativelanguage.googleapis.com/v1beta/openai",
                )
            ),
            "model": os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        }

    if provider == "ollama":
        return {
            "provider": "ollama",
            "api_key": os.getenv("OLLAMA_API_KEY", "ollama").strip() or "ollama",
            "base_url": _normalize_base_url(os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")),
            "model": os.getenv("OLLAMA_MODEL", "llama3.2:latest"),
        }

    # OpenAI-compatible generic mode: OpenRouter, Groq, Together, etc.
    return {
        "provider": provider,
        "api_key": os.getenv("LLM_API_KEY", "").strip(),
        "base_url": _normalize_base_url(os.getenv("LLM_BASE_URL", "")),
        "model": os.getenv("LLM_MODEL", ""),
    }


def _extract_json_payload(content: str) -> dict:
    stripped = content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return json.loads(stripped)

    # Robust extraction for models that wrap JSON in markdown/code fences.
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(stripped[start : end + 1])

    return {}


def _call_openai_compatible_chat(prompt: str) -> list[DrugCandidate]:
    config = _resolve_provider_config()
    api_key = config["api_key"]
    base_url = config["base_url"]
    model = config["model"]

    if not base_url or not model:
        return []

    # For local Ollama in default config, key is optional; for hosted APIs key is usually required.
    if config["provider"] != "ollama" and not api_key:
        return []

    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a biomedical reasoning assistant. "
                    "Only propose drugs supported by provided evidence snippets. "
                    "Always return valid JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }

    # Some providers may not support response_format. Keep it optional.
    payload["response_format"] = {"type": "json_object"}

    req = urllib.request.Request(
        url=f"{base_url}/chat/completions",
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        data=json.dumps(payload).encode("utf-8"),
    )

    try:
        with urllib.request.urlopen(req, timeout=50) as response:
            data = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        # Retry without response_format for compatibility.
        if payload.get("response_format") is not None:
            payload.pop("response_format", None)
            retry_req = urllib.request.Request(
                url=f"{base_url}/chat/completions",
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                data=json.dumps(payload).encode("utf-8"),
            )
            with urllib.request.urlopen(retry_req, timeout=50) as response:
                data = response.read().decode("utf-8")
        else:
            raise exc

    response_json = json.loads(data)
    content = response_json["choices"][0]["message"]["content"]
    parsed = _extract_json_payload(content)

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


def _groundedness_score(candidate: DrugCandidate, docs: list[EvidenceDoc]) -> float:
    evidence_pmids = set(candidate.evidence_pmids)
    if not evidence_pmids:
        return 0.0

    doc_map = {d.pmid: (d.title + " " + d.abstract).lower() for d in docs}
    mechanism_tokens = [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", candidate.mechanism)]
    if not mechanism_tokens:
        return 0.5

    supported = 0
    for token in mechanism_tokens[:20]:
        if any(token in doc_map.get(pmid, "") for pmid in evidence_pmids):
            supported += 1

    return supported / max(1, min(20, len(mechanism_tokens)))


def _average_groundedness(candidates: list[DrugCandidate], docs: list[EvidenceDoc]) -> float:
    if not candidates:
        return 0.0
    scores = [_groundedness_score(c, docs) for c in candidates]
    return sum(scores) / len(scores)


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
- Return at most 8 high-quality candidates.
""".strip()

    # Up to 2 LLM attempts; accept when groundedness is acceptable.
    for attempt in range(2):
        try:
            llm_candidates = _call_openai_compatible_chat(prompt)
            if llm_candidates:
                avg_groundedness = _average_groundedness(llm_candidates, docs)
                if avg_groundedness >= 0.45 or attempt == 1:
                    return llm_candidates

                prompt += "\n\nImportant retry instruction: improve groundedness. Cite only PMIDs present in snippets and use wording directly supported by those snippets."
        except Exception:
            # Graceful fallback to deterministic heuristic reasoning.
            break

    return _heuristic_reason(docs)
