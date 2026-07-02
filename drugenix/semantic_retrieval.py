"""
Drugenix - Semantic Retrieval Layer
====================================
Replaces token-overlap reranking with embedding-based semantic retrieval
using sentence-transformers and ChromaDB vector store.

This fulfills the FAISS/ChromaDB + HuggingFace claims from the project proposal.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import uuid
from typing import Optional

from drugenix.models import EvidenceDoc

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_COLLECTION_NAME = "drugenix_pubmed"

_model = None
_chroma_client = None
_lock = threading.Lock()


def _get_embedding_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", _EMBEDDING_MODEL)
        _model = SentenceTransformer(_EMBEDDING_MODEL)
    return _model


def _get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        _chroma_client = chromadb.Client()
    return _chroma_client


def _doc_id(doc: EvidenceDoc) -> str:
    return f"pmid_{doc.pmid}"


def _doc_text(doc: EvidenceDoc) -> str:
    return f"{doc.title}. {doc.abstract}"[:2000]


def embed_and_store(docs: list[EvidenceDoc], collection_name: str = _COLLECTION_NAME) -> None:
    """Embed documents and store them in ChromaDB."""
    if not docs:
        return

    client = _get_chroma_client()
    model = _get_embedding_model()

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [_doc_text(d) for d in docs]
    ids = [_doc_id(d) for d in docs]
    metadatas = [
        {
            "pmid": d.pmid,
            "title": d.title,
            "journal": d.journal,
            "year": d.year,
        }
        for d in docs
    ]

    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    existing_ids = set()
    try:
        existing = collection.get(ids=ids)
        existing_ids = set(existing["ids"])
    except Exception:
        pass

    new_indices = [i for i, doc_id in enumerate(ids) if doc_id not in existing_ids]
    if new_indices:
        collection.add(
            ids=[ids[i] for i in new_indices],
            embeddings=[embeddings[i] for i in new_indices],
            documents=[texts[i] for i in new_indices],
            metadatas=[metadatas[i] for i in new_indices],
        )

    logger.info("Stored %d documents in ChromaDB (collection=%s)", len(new_indices), collection_name)


def semantic_rerank(
    query: str,
    docs: list[EvidenceDoc],
    top_k: int = 8,
    collection_name: Optional[str] = None,
) -> list[EvidenceDoc]:
    """
    Rerank retrieved documents using semantic similarity.

    1. Embeds all docs into a fresh, per-call ChromaDB collection
       (isolated so concurrent calls from the Planner Agent's parallel
       sub-tasks never race on a shared collection).
    2. Queries with the disease query embedding.
    3. Returns top_k most semantically similar documents.
    """
    if not docs:
        return []

    # Each call gets its own collection to stay safe under concurrent
    # (multi-disease / Planner Agent) execution.
    own_collection = collection_name or f"{_COLLECTION_NAME}_{uuid.uuid4().hex[:12]}"

    with _lock:
        client = _get_chroma_client()
        model = _get_embedding_model()

        embed_and_store(docs, own_collection)
        collection = client.get_collection(name=own_collection)

        query_embedding = model.encode([query], show_progress_bar=False).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, len(docs)),
            include=["metadatas", "distances"],
        )

        if collection_name is None:
            try:
                client.delete_collection(name=own_collection)
            except Exception:
                pass

    pmid_to_doc = {d.pmid: d for d in docs}
    ranked_docs: list[EvidenceDoc] = []

    if results and results["ids"]:
        for i, doc_id_list in enumerate(results["ids"]):
            for j, doc_id in enumerate(doc_id_list):
                pmid = results["metadatas"][i][j]["pmid"]
                if pmid in pmid_to_doc:
                    doc = pmid_to_doc[pmid]
                    distance = results["distances"][i][j] if results["distances"] else 0.0
                    doc.retrieval_score = 1.0 - distance
                    ranked_docs.append(doc)

    for doc in docs:
        if doc not in ranked_docs:
            doc.retrieval_score = 0.0
            ranked_docs.append(doc)

    return ranked_docs[:top_k]


def compute_query_embedding(query: str) -> list[float]:
    """Return embedding vector for a query string."""
    model = _get_embedding_model()
    return model.encode([query], show_progress_bar=False).tolist()[0]


def compute_similarity(query: str, text: str) -> float:
    """Compute cosine similarity between a query and a text snippet."""
    model = _get_embedding_model()
    embeddings = model.encode([query, text], show_progress_bar=False)
    from numpy import dot
    from numpy.linalg import norm
    a, b = embeddings[0], embeddings[1]
    return float(dot(a, b) / (norm(a) * norm(b)))
