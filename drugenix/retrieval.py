from __future__ import annotations

import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter

from drugenix.models import EvidenceDoc

logger = logging.getLogger(__name__)

_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9\-]+")
# NCBI E-utilities rate-limits unauthenticated traffic to 3 req/sec; without
# retry logic, a researcher running several queries back-to-back would
# silently get throttled (HTTP 429) into an empty-looking result instead of
# an actual error. Retry with backoff instead of failing on the first hit.
_MAX_RETRIES = 3
_CLEAN_PREFIXES = (
    "find ",
    "suggest ",
    "show ",
    "give me ",
    "find repurposed drug candidates for ",
    "find promising repurposed drugs for ",
    "find candidate drugs for ",
    "repurposed drug candidates for ",
    "repurposed drugs for ",
)
_CLEAN_SUFFIXES = (
    " with evidence from pubmed",
    " from pubmed",
    " using pubmed",
    " with pubmed evidence",
    " please",
)


def _normalize_query(query: str) -> str:
    text = query.strip().lower()

    for prefix in _CLEAN_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()

    # Prefer the phrase after the last "for" when present.
    if " for " in text:
        text = text.rsplit(" for ", 1)[-1].strip()

    for suffix in _CLEAN_SUFFIXES:
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()

    # Remove trailing punctuation and filler words.
    text = re.sub(r"[?.!,]+$", "", text).strip()
    text = re.sub(r"\b(drug candidates|candidate drugs|repurposed drugs|repurposed drug candidates)\b", "", text).strip()
    return text or query.strip()


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def _overlap_score(query: str, title: str, abstract: str) -> float:
    query_terms = _tokenize(query)
    if not query_terms:
        return 0.0

    doc_terms = _tokenize(f"{title} {abstract}")
    if not doc_terms:
        return 0.0

    q_counts = Counter(query_terms)
    d_counts = Counter(doc_terms)

    overlap = 0.0
    for term, q_freq in q_counts.items():
        overlap += min(q_freq, d_counts.get(term, 0))

    return overlap / max(1.0, len(q_counts))


def _http_get(url: str, max_retries: int = _MAX_RETRIES) -> bytes:
    """
    GET with retry/backoff for transient failures (rate limiting, timeouts,
    5xx). A single failed request used to propagate straight up and (via
    pipeline.py's try/except around retrieval) silently turn into "0 papers
    found" — indistinguishable from a genuine empty result. Retrying first
    means a real NCBI hiccup doesn't masquerade as "no evidence exists".
    """
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Drugenix/1.0 (research pipeline)"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=25) as response:
                return response.read()
        except urllib.error.HTTPError as e:
            last_exc = e
            if e.code == 429 or 500 <= e.code < 600:
                retry_after = e.headers.get("Retry-After") if e.headers else None
                wait = float(retry_after) if retry_after else 0.5 * (2 ** attempt)
                logger.warning(
                    "PubMed request throttled/failed (HTTP %d), retrying in %.1fs (attempt %d/%d): %s",
                    e.code, wait, attempt + 1, max_retries, url,
                )
                time.sleep(wait)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_exc = e
            wait = 0.5 * (2 ** attempt)
            logger.warning(
                "PubMed request failed (%s), retrying in %.1fs (attempt %d/%d): %s",
                e, wait, attempt + 1, max_retries, url,
            )
            time.sleep(wait)
            continue

    assert last_exc is not None
    raise last_exc


def _pubmed_search_ids(disease_query: str, retmax: int) -> list[str]:
    normalized_query = _normalize_query(disease_query)
    term = f"({normalized_query}) AND (drug therapy[MeSH Terms] OR therapeutics[Subheading])"
    params = {
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retmax": str(retmax),
        "sort": "relevance",
    }
    url = f"{_EUTILS_BASE}/esearch.fcgi?{urllib.parse.urlencode(params)}"
    data = _http_get(url)

    import json

    payload = json.loads(data.decode("utf-8"))
    return payload.get("esearchresult", {}).get("idlist", [])


def _extract_abstract_text(article_node: ET.Element) -> str:
    parts: list[str] = []
    for node in article_node.findall(".//Abstract/AbstractText"):
        label = node.attrib.get("Label", "").strip()
        text = "".join(node.itertext()).strip()
        if not text:
            continue
        parts.append(f"{label}: {text}" if label else text)
    return "\n".join(parts)


def _pubmed_fetch_records(pmids: list[str]) -> list[EvidenceDoc]:
    if not pmids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    url = f"{_EUTILS_BASE}/efetch.fcgi?{urllib.parse.urlencode(params)}"
    xml_bytes = _http_get(url)

    root = ET.fromstring(xml_bytes)
    docs: list[EvidenceDoc] = []

    for article in root.findall(".//PubmedArticle"):
        pmid = (article.findtext(".//PMID") or "").strip()
        title = (article.findtext(".//ArticleTitle") or "").strip()
        abstract = _extract_abstract_text(article)
        journal = (article.findtext(".//Journal/Title") or "").strip()
        year = (
            article.findtext(".//PubDate/Year")
            or article.findtext(".//ArticleDate/Year")
            or ""
        ).strip()

        if not pmid or not (title or abstract):
            continue

        docs.append(
            EvidenceDoc(
                pmid=pmid,
                title=title,
                abstract=abstract,
                journal=journal,
                year=year,
            )
        )

    return docs


def retrieve_pubmed(disease_query: str, retmax: int = 40, top_k: int = 8) -> list[EvidenceDoc]:
    normalized_query = _normalize_query(disease_query)
    pmids = _pubmed_search_ids(disease_query=normalized_query, retmax=retmax)
    docs = _pubmed_fetch_records(pmids)

    for doc in docs:
        doc.retrieval_score = _overlap_score(normalized_query, doc.title, doc.abstract)

    docs.sort(key=lambda d: d.retrieval_score, reverse=True)
    return docs[:top_k]
