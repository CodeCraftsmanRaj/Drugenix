from __future__ import annotations

import json
import os
import re
import sys
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
_DRUG_KEYWORDS = ("inhibitor", "agonist", "antagonist", "activator", "blocker", "therapy", "treatment", "drug", "agent", "compound", "receptor", "target")


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
        # Gemini's OpenAI-compatible endpoint already has the correct path
        # (".../v1beta/openai") and must NOT go through _normalize_base_url,
        # which would append an extra "/v1" and produce a 404 — Gemini's
        # base URL structure is different from OpenAI/Ollama's.
        gemini_base = os.getenv(
            "GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai",
        ).rstrip("/")
        return {
            "provider": "gemini",
            "api_key": os.getenv("GEMINI_API_KEY", "").strip(),
            "base_url": gemini_base,
            # gemini-2.0-flash was shut down 2026-06-01. Plain gemini-2.5-flash
            # is restricted for newer accounts ("no longer available to new
            # users" per Google's own error message) — gemini-2.5-flash-lite
            # is what Google's current docs actually use for this endpoint.
            "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"),
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
        with urllib.request.urlopen(req, timeout=180) as response:
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
            with urllib.request.urlopen(retry_req, timeout=180) as response:
                data = response.read().decode("utf-8")
        else:
            raise exc

    response_json = json.loads(data)
    content = response_json["choices"][0]["message"]["content"]
    parsed = _extract_json_payload(content)

    out: list[DrugCandidate] = []
    for item in parsed.get("candidates", []):
        raw_name = str(item.get("drug", "")).strip()
        # Strip parenthetical suffixes: "Rapamycin (mTOR modulator)" → "Rapamycin"
        clean_name = re.sub(r"\s*\(.*?\)\s*$", "", raw_name).strip()
        out.append(
            DrugCandidate(
                drug=clean_name,
                mechanism=str(item.get("mechanism", "")).strip(),
                confidence=float(item.get("confidence", 0.0)),
                evidence_pmids=[str(p) for p in item.get("evidence_pmids", [])],
                original_use=str(item.get("original_use", "")).strip(),
                repurposing_rationale=str(item.get("repurposing_rationale", "")).strip(),
            )
        )

    return [c for c in out if c.drug]


def _call_gemini_native(prompt: str, config: dict[str, str]) -> list[DrugCandidate]:
    """
    Call Gemini's native generateContent API instead of the OpenAI-compatible
    shim. As of mid-2026 Google issues newer "AQ." format Auth keys from AI
    Studio by default, and those keys are not reliably accepted by the
    OpenAI-compatible endpoint's `Authorization: Bearer` auth path (it can
    return 400/401/404 depending on account) — only the native endpoint with
    an `x-goog-api-key` header is guaranteed to work for these keys.
    """
    api_key = config["api_key"]
    model = config["model"]

    if not api_key or not model:
        return []

    # Pass the key as a query param (the original, most universally-supported
    # Gemini auth method) rather than relying solely on a custom header.
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
        },
    }

    req = urllib.request.Request(
        url=url,
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode("utf-8"),
    )

    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            data = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        # Surface Google's actual error body (e.g. "API key not valid",
        # "Generative Language API has not been used in project...") instead
        # of a bare status code — this is what actually explains failures.
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API error {exc.code}: {body}") from exc

    response_json = json.loads(data)
    content = response_json["candidates"][0]["content"]["parts"][0]["text"]
    parsed = _extract_json_payload(content)

    out: list[DrugCandidate] = []
    for item in parsed.get("candidates", []):
        raw_name = str(item.get("drug", "")).strip()
        clean_name = re.sub(r"\s*\(.*?\)\s*$", "", raw_name).strip()
        out.append(
            DrugCandidate(
                drug=clean_name,
                mechanism=str(item.get("mechanism", "")).strip(),
                confidence=float(item.get("confidence", 0.0)),
                evidence_pmids=[str(p) for p in item.get("evidence_pmids", [])],
                original_use=str(item.get("original_use", "")).strip(),
                repurposing_rationale=str(item.get("repurposing_rationale", "")).strip(),
            )
        )

    return [c for c in out if c.drug]


def _call_llm(prompt: str) -> list[DrugCandidate]:
    """Dispatch to the correct call path based on the resolved provider."""
    config = _resolve_provider_config()
    if config["provider"] == "gemini":
        return _call_gemini_native(prompt, config)
    return _call_openai_compatible_chat(prompt)


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
    drug_candidates_by_pmid: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for doc in docs:
        text = f"{doc.title}. {doc.abstract}".replace("\n", " ")
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

        for sentence in sentences:
            lower_sent = sentence.lower()
            has_drug_context = any(kw in lower_sent for kw in _DRUG_KEYWORDS)
            if not has_drug_context:
                continue

            for match in _CANDIDATE_RE.finditer(sentence):
                token = match.group(1)
                lower = token.lower()

                if any(lower.endswith(sfx) for sfx in _DRUG_SUFFIXES) or (has_drug_context and len(token) > 3):
                    name = token[0].upper() + token[1:]
                    if name not in [x[0] for x in drug_candidates_by_pmid[doc.pmid]]:
                        drug_candidates_by_pmid[doc.pmid].append((name, sentence[:250]))

    evidence_map: dict[str, set[str]] = defaultdict(set)
    mechanism_map: dict[str, list[str]] = defaultdict(list)
    for pmid, drugs_and_sents in drug_candidates_by_pmid.items():
        for drug, sent in drugs_and_sents:
            evidence_map[drug].add(pmid)
            if len(mechanism_map[drug]) < 2:
                mechanism_map[drug].append(sent)

    candidates: list[DrugCandidate] = []
    for drug, pmids in evidence_map.items():
        support = len(pmids)
        confidence = min(0.85, 0.35 + 0.12 * support)
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
    return candidates[:15]


def generate_candidates(disease_query: str, docs: list[EvidenceDoc]) -> list[DrugCandidate]:
    context = _build_context_snippets(docs)
    prompt = f"""
Disease query: {disease_query}

Evidence snippets from PubMed:
{context}

Task:
You are a biomedical researcher identifying drug repurposing candidates.
Return a JSON object with key "candidates" containing up to 8 candidates.

Each candidate MUST include these exact fields:
- drug (string): A SPECIFIC drug name only — e.g. "Rapamycin", "Metformin", "Donepezil".
  NEVER use a drug class ("Cholinesterase inhibitors"), mechanism description ("mTOR inhibitor"),
  or multi-word phrase ("Brain Shuttle Antibody"). Use the generic or brand name of ONE molecule.
- original_use (string): The disease or condition this drug is CURRENTLY approved/used for.
  Example: "Rapamycin -> organ transplant rejection", "Metformin -> Type 2 diabetes".
  If unknown, write "Investigational".
- repurposing_rationale (string): 1-2 sentences explaining WHY this drug could work for the
  queried disease, based on the evidence snippets. Connect the original mechanism to the new disease.
  Example: "Rapamycin inhibits mTOR, activating autophagy which clears amyloid-beta plaques
  implicated in Alzheimer's neurodegeneration."
- mechanism (string): Short mechanistic summary (how it works at the molecular level).
- confidence (number 0-1): Your confidence this is a valid repurposing candidate.
- evidence_pmids (array): PMID strings from the snippets above that support this candidate.

Rules:
- Only propose drugs that appear or are strongly implied in the provided PubMed evidence.
- Each "drug" field must be one specific molecule name.
- Prioritise drugs already approved for another disease — they can be repurposed faster.
- CRITICAL: Do NOT propose a drug whose "original_use" already IS (or is a synonym of)
  the queried disease "{disease_query}" itself. A repurposing candidate must currently
  be approved/indicated for a DIFFERENT disease than the one being queried — proposing
  a drug already used to treat {disease_query} is not repurposing and must be excluded.
- Set confidence lower if evidence is limited to 1 paper.
- Return at most 8 candidates.
""".strip()

    # Up to 2 LLM attempts; accept when groundedness is acceptable.
    for attempt in range(2):
        try:
            llm_candidates = _call_llm(prompt)
            if llm_candidates:
                avg_groundedness = _average_groundedness(llm_candidates, docs)
                if avg_groundedness >= 0.45 or attempt == 1:
                    return llm_candidates

                prompt += "\n\nImportant retry instruction: improve groundedness. Cite only PMIDs present in snippets and use wording directly supported by those snippets."
            else:
                print(
                    f"[Drugenix] LLM returned no candidates (provider={_resolve_provider_config()['provider']}); "
                    "falling back to heuristic extraction.",
                    file=sys.stderr,
                )
        except Exception as exc:
            # Graceful fallback to deterministic heuristic reasoning, but tell the
            # user why — silent fallback makes "no LLM configured" indistinguishable
            # from "LLM call failed" (timeout, bad model name, server down, etc).
            print(
                f"[Drugenix] LLM call failed ({type(exc).__name__}: {exc}); "
                "falling back to heuristic extraction.",
                file=sys.stderr,
            )
            break

    return _heuristic_reason(docs)
