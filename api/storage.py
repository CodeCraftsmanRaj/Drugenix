"""
Drugenix API - Query History Storage
=======================================
Lightweight JSON-file-based storage for query history.
No external DB needed for a project of this scale.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

_HISTORY_DIR = Path(__file__).resolve().parent.parent / "outputs" / "history"
_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
_INDEX_FILE = _HISTORY_DIR / "_index.json"


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60] or "query"


def _load_index() -> list[dict]:
    if not _INDEX_FILE.exists():
        return []
    try:
        return json.loads(_INDEX_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_index(index: list[dict]) -> None:
    _INDEX_FILE.write_text(json.dumps(index, indent=2), encoding="utf-8")


def save_query_result(query: str, result: dict) -> str:
    """Save a pipeline result and return its generated query_id."""
    query_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now(timezone.utc).isoformat()
    slug = _slugify(query)
    filename = f"{timestamp[:10]}_{slug}_{query_id}.json"
    filepath = _HISTORY_DIR / filename

    payload = {
        "query_id": query_id,
        "query": query,
        "timestamp": timestamp,
        "result": result,
    }
    filepath.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    index = _load_index()
    aggregation = result.get("aggregation", {})
    index.append({
        "query_id": query_id,
        "query": query,
        "timestamp": timestamp,
        "filename": filename,
        "candidate_count": len(result.get("candidates", [])),
        "recommended_count": len(aggregation.get("recommended", [])),
    })
    _save_index(index)

    return query_id


def get_history(limit: int = 50) -> list[dict]:
    index = _load_index()
    return sorted(index, key=lambda x: x["timestamp"], reverse=True)[:limit]


def get_query_result(query_id: str) -> dict | None:
    index = _load_index()
    entry = next((e for e in index if e["query_id"] == query_id), None)
    if not entry:
        return None

    filepath = _HISTORY_DIR / entry["filename"]
    if not filepath.exists():
        return None

    return json.loads(filepath.read_text(encoding="utf-8"))


def delete_query_result(query_id: str) -> bool:
    index = _load_index()
    entry = next((e for e in index if e["query_id"] == query_id), None)
    if not entry:
        return False

    filepath = _HISTORY_DIR / entry["filename"]
    if filepath.exists():
        filepath.unlink()

    index = [e for e in index if e["query_id"] != query_id]
    _save_index(index)
    return True
