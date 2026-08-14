"""
Drugenix - Open Targets Platform Live Client
===============================================
Replaces "any disease we happened to curate" with "any disease in the
Open Targets Platform" (which covers essentially every human disease/
phenotype via the EFO ontology, backed by ChEMBL, ClinVar, ClinGen,
Orphanet, PheWAS, gene burden studies, etc).

Two calls, both live GraphQL requests against
https://api.platform.opentargets.org/api/v4/graphql (free, no API key):

  resolve_disease(text)      -> best-matching {id, name} EFO disease node
  get_disease_kg_edges(text) -> drug/target/disease triples for that
                                 disease, in the same shape the curated
                                 knowledge graph uses, so they can be
                                 merged straight into the NetworkX graph.

Every call is wrapped defensively: a network failure, a schema change on
Open Targets' side, or a disease with zero hits all degrade to an empty
result rather than raising, so the rest of the pipeline (which already
has a curated-KG fallback) keeps working.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

_OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"

_SEARCH_QUERY = """
query DrugenixDiseaseSearch($q: String!) {
  search(queryString: $q, entityNames: ["disease"], page: {index: 0, size: 5}) {
    hits {
      id
      name
      entity
    }
  }
}
""".strip()

_KNOWN_DRUGS_QUERY = """
query DrugenixKnownDrugs($efoId: String!, $size: Int!) {
  disease(efoId: $efoId) {
    id
    name
    knownDrugs(size: $size) {
      count
      rows {
        drug {
          id
          name
          isApproved
        }
        mechanismOfAction
        phase
        status
        target {
          id
          approvedSymbol
        }
      }
    }
  }
}
""".strip()

_ASSOCIATED_TARGETS_QUERY = """
query DrugenixAssociatedTargets($efoId: String!, $size: Int!) {
  disease(efoId: $efoId) {
    id
    name
    associatedTargets(page: {index: 0, size: $size}) {
      rows {
        score
        target {
          id
          approvedSymbol
        }
      }
    }
  }
}
""".strip()

_disease_search_cache: dict[str, Optional[dict]] = {}
_known_drugs_cache: dict[str, list[dict]] = {}
_associated_targets_cache: dict[str, list[dict]] = {}


def _post_graphql(query: str, variables: dict, timeout: int = 20) -> dict:
    body = json.dumps({"query": query, "variables": variables}).encode("utf-8")
    req = urllib.request.Request(
        _OT_URL,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "User-Agent": "Drugenix/1.0 (live knowledge graph layer)",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    if payload.get("errors"):
        logger.warning("Open Targets GraphQL errors: %s", payload["errors"])

    return payload.get("data") or {}


def resolve_disease(query_text: str, timeout: int = 15) -> Optional[dict]:
    """
    Resolve free-text (any phrasing, any disease) to a canonical Open
    Targets / EFO disease node: {"id": "EFO_...", "name": "..."}.

    Returns None if nothing matches or the API is unreachable.
    """
    key = query_text.strip().lower()
    if not key:
        return None
    if key in _disease_search_cache:
        return _disease_search_cache[key]

    result: Optional[dict] = None
    try:
        data = _post_graphql(_SEARCH_QUERY, {"q": query_text.strip()}, timeout=timeout)
        hits = (data.get("search") or {}).get("hits") or []
        if hits:
            best = hits[0]
            result = {"id": best["id"], "name": best["name"]}
    except Exception as e:
        logger.info("Open Targets disease search unavailable for '%s': %s", query_text, e)

    _disease_search_cache[key] = result
    return result


def get_known_drugs(efo_id: str, size: int = 60, timeout: int = 20) -> list[dict]:
    """
    Return known drug/target rows for a resolved EFO disease id, normalized
    to: {"drug", "target", "disease", "mechanism", "phase", "status", "is_approved"}.
    """
    if efo_id in _known_drugs_cache:
        return _known_drugs_cache[efo_id]

    rows_out: list[dict] = []
    try:
        data = _post_graphql(
            _KNOWN_DRUGS_QUERY, {"efoId": efo_id, "size": size}, timeout=timeout
        )
        disease_node = data.get("disease") or {}
        disease_name = disease_node.get("name", "")
        rows = ((disease_node.get("knownDrugs") or {}).get("rows")) or []

        for row in rows:
            try:
                drug = (row.get("drug") or {}).get("name", "").strip()
                if not drug:
                    continue
                target = (row.get("target") or {}).get("approvedSymbol", "") or ""
                rows_out.append({
                    "drug": drug,
                    "target": target,
                    "disease": disease_name,
                    "mechanism": row.get("mechanismOfAction") or "",
                    "phase": row.get("phase"),
                    "status": row.get("status") or "",
                    "is_approved": bool((row.get("drug") or {}).get("isApproved")),
                })
            except Exception:
                continue
    except Exception as e:
        logger.info("Open Targets knownDrugs unavailable for '%s': %s", efo_id, e)

    _known_drugs_cache[efo_id] = rows_out
    return rows_out


def get_associated_targets(efo_id: str, size: int = 25, timeout: int = 20) -> list[dict]:
    """Return top scored gene targets associated with a disease (for mechanism-similarity reasoning)."""
    if efo_id in _associated_targets_cache:
        return _associated_targets_cache[efo_id]

    out: list[dict] = []
    try:
        data = _post_graphql(
            _ASSOCIATED_TARGETS_QUERY, {"efoId": efo_id, "size": size}, timeout=timeout
        )
        rows = ((data.get("disease") or {}).get("associatedTargets") or {}).get("rows") or []
        for row in rows:
            symbol = (row.get("target") or {}).get("approvedSymbol", "")
            if symbol:
                out.append({"target": symbol, "score": row.get("score", 0.0)})
    except Exception as e:
        logger.info("Open Targets associatedTargets unavailable for '%s': %s", efo_id, e)

    _associated_targets_cache[efo_id] = out
    return out


def get_disease_kg_edges(disease_query: str, size: int = 60) -> list[dict]:
    """
    One-call convenience: resolve free-text disease -> live drug/target/
    disease triples, in the same {"drug","target","disease","mechanism"}
    shape as the curated relationships list, ready to merge into the
    NetworkX knowledge graph. Works for any disease Open Targets knows
    about (which is essentially any disease with genetic/clinical/
    chemical evidence — tens of thousands of EFO terms).
    """
    resolved = resolve_disease(disease_query)
    if not resolved:
        return []

    rows = get_known_drugs(resolved["id"], size=size)
    edges = []
    for row in rows:
        edges.append({
            "drug": row["drug"],
            "target": row["target"] or "UNKNOWN",
            "disease": disease_query.strip(),
            "mechanism": row["mechanism"] or f"Open Targets evidence (phase {row.get('phase')}, {row.get('status')})",
            "source": "OpenTargets",
            "is_approved": row.get("is_approved", False),
        })
    return edges


if __name__ == "__main__":
    # Manual sanity check: `python -m drugenix.opentargets "type 2 diabetes"`
    # Only useful where outbound HTTPS to api.platform.opentargets.org is
    # allowed (this call is blocked in sandboxed dev environments with
    # network allowlists, e.g. Claude's own build sandbox — it will work
    # fine on a normal server/laptop/Render deployment).
    import sys

    logging.basicConfig(level=logging.INFO)
    q = " ".join(sys.argv[1:]) or "type 2 diabetes"
    d = resolve_disease(q)
    print("Resolved:", d)
    if d:
        edges = get_disease_kg_edges(q)
        print(f"{len(edges)} known-drug edges. First 5:")
        for e in edges[:5]:
            print(" ", e)
