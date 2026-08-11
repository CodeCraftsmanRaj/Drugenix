"""
Drugenix API - FastAPI Backend
=================================
REST API wrapping the full agentic pipeline:
  Planner -> Retrieval -> NER -> Reasoning -> Validation -> Knowledge Graph -> Aggregation

Endpoints:
  POST /api/query              - run a full pipeline query (handles multi-disease via Planner)
  GET  /api/history             - list past queries
  GET  /api/history/{query_id}  - get a specific past result
  DELETE /api/history/{query_id} - delete a past result
  GET  /api/knowledge-graph/{disease} - get KG subgraph for visualization
  GET  /api/knowledge-graph/stats     - KG summary stats
  GET  /api/metrics              - evaluation metrics for plots/dashboard
  GET  /api/health               - health check
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.storage import (
    delete_query_result,
    get_history,
    get_query_result,
    save_query_result,
)
from drugenix.knowledge_graph import get_knowledge_graph
from drugenix.pipeline import run_pipeline
from drugenix.planner import create_plan, execute_plan

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("drugenix.api")

app = FastAPI(
    title="Drugenix API",
    description="Agentic AI System for Accelerated Drug Repurposing",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Disease query, may include multiple diseases.")
    retmax: int = Field(40, ge=5, le=200)
    top_k: int = Field(8, ge=3, le=30)
    provider: Optional[str] = Field(None, description="LLM provider override: openai | gemini | ollama | custom")
    model: Optional[str] = None
    use_semantic_retrieval: bool = True
    use_ner: bool = True
    use_kg: bool = True
    save_to_history: bool = True


class QueryResponse(BaseModel):
    query_id: Optional[str]
    elapsed_seconds: float
    result: dict


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "drugenix-api", "version": "0.2.0"}


@app.post("/api/query", response_model=QueryResponse)
def run_query(req: QueryRequest) -> QueryResponse:
    import os

    # Gemini model names Google has discontinued / restricted for new
    # accounts during 2026. A stale frontend form field can keep re-sending
    # one of these on every request — ignore them so the code's own current
    # default (kept up to date in drugenix/reasoning.py) applies instead of
    # silently reintroducing a dead model.
    _DEAD_GEMINI_MODELS = {
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    }

    if req.provider:
        os.environ["LLM_PROVIDER"] = req.provider.strip().lower()

    provider = (req.provider or os.getenv("LLM_PROVIDER", "openai")).strip().lower()

    # Reset every per-provider model override on EVERY request, then only
    # re-set the one for the active provider if a model was actually given
    # this time. Render keeps one worker process alive across all requests
    # (it doesn't restart per-request), so without this reset, a model typed
    # into the form on one request would silently keep overriding every
    # later request in that process's lifetime — including ones where the
    # field was left blank. This was the actual root cause of the model
    # override "sticking" no matter what the current request sent.
    for _env_key in ("OLLAMA_MODEL", "OPENAI_MODEL", "GEMINI_MODEL", "LLM_MODEL"):
        os.environ.pop(_env_key, None)

    if req.model:
        is_dead_gemini_model = (
            provider in {"gemini", "gemmini", "google"}
            and req.model.strip().lower() in _DEAD_GEMINI_MODELS
        )
        if is_dead_gemini_model:
            logger.warning(
                "Ignoring discontinued Gemini model '%s' from request; "
                "using current default instead.",
                req.model,
            )
        elif provider == "ollama":
            os.environ["OLLAMA_MODEL"] = req.model
        elif provider in {"openai", "open-ai"}:
            os.environ["OPENAI_MODEL"] = req.model
        elif provider in {"gemini", "gemmini", "google"}:
            os.environ["GEMINI_MODEL"] = req.model
        else:
            os.environ["LLM_MODEL"] = req.model

    start = time.time()

    try:
        plan = create_plan(req.query, retmax=req.retmax, top_k=req.top_k)

        def _pipeline_fn(disease_query: str, retmax: int, top_k: int) -> dict:
            return run_pipeline(
                disease_query=disease_query,
                retmax=retmax,
                top_k=top_k,
                use_semantic_retrieval=req.use_semantic_retrieval,
                use_ner=req.use_ner,
                use_kg=req.use_kg,
            )

        if len(plan.tasks) == 1 and not plan.is_multi_disease:
            result = _pipeline_fn(req.query, req.retmax, req.top_k)
            result["planner"] = {
                "is_multi_disease": False,
                "strategy": "single",
                "sub_tasks": [{"disease_query": req.query, "status": "completed"}],
                "total_tasks": 1,
                "completed_tasks": 1,
                "failed_tasks": 0,
            }
        else:
            result = execute_plan(plan, _pipeline_fn)

    except Exception as e:
        logger.exception("Pipeline execution failed")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    elapsed = time.time() - start

    query_id = None
    if req.save_to_history:
        try:
            query_id = save_query_result(req.query, result)
        except Exception as e:
            logger.warning("Could not save to history: %s", e)

    return QueryResponse(query_id=query_id, elapsed_seconds=round(elapsed, 2), result=result)


@app.get("/api/history")
def list_history(limit: int = 50) -> list[dict]:
    return get_history(limit=limit)


@app.get("/api/history/{query_id}")
def get_history_item(query_id: str) -> dict:
    result = get_query_result(query_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Query not found")
    return result


@app.delete("/api/history/{query_id}")
def delete_history_item(query_id: str) -> dict:
    ok = delete_query_result(query_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Query not found")
    return {"deleted": True, "query_id": query_id}


@app.get("/api/knowledge-graph/stats")
def kg_stats() -> dict:
    kg = get_knowledge_graph()
    return kg.get_graph_stats()


@app.get("/api/knowledge-graph/{disease}")
def kg_subgraph(disease: str, max_hops: int = 2) -> dict:
    kg = get_knowledge_graph()
    subgraph = kg.get_subgraph_for_disease(disease, max_hops=max_hops)
    if not subgraph["nodes"]:
        raise HTTPException(status_code=404, detail=f"No knowledge graph data found for '{disease}'")
    return subgraph


@app.get("/api/metrics")
def get_metrics() -> dict:
    """Return evaluation metrics for the dashboard (from saved benchmark file if present)."""
    import json

    metrics_path = PROJECT_ROOT / "data" / "metrics_drugenix_mvp.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    template_path = PROJECT_ROOT / "data" / "metrics_template.json"
    if template_path.exists():
        return json.loads(template_path.read_text(encoding="utf-8"))

    return {"retrieval": {}, "generation": {}, "ablation": {}}


@app.get("/api/comparative-evaluation")
def get_comparative_evaluation() -> dict:
    """Return Pure LLM vs Pure Graph vs Hybrid comparative evaluation results."""
    import json

    eval_path = PROJECT_ROOT / "data" / "comparative_evaluation.json"
    if eval_path.exists():
        return json.loads(eval_path.read_text(encoding="utf-8"))

    raise HTTPException(
        status_code=404,
        detail="Comparative evaluation not run yet. Run scripts/run_benchmark.py first.",
    )


@app.post("/api/run-benchmark")
def run_benchmark_endpoint(
    diseases: int = 5,
    fast: bool = True,
) -> dict:
    """
    Trigger the benchmark evaluation in the background.
    Returns immediately with a status message; results are written to data/.
    Use GET /api/comparative-evaluation to check when done.
    """
    import subprocess
    import threading

    script = PROJECT_ROOT / "scripts" / "run_benchmark.py"
    args = ["python", str(script)]
    if diseases:
        args += ["--diseases", str(diseases)]
    if fast:
        args += ["--fast"]

    def _run():
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=1800)
            logger.info("Benchmark finished: %s", result.stdout[-500:])
            if result.returncode != 0:
                logger.error("Benchmark error: %s", result.stderr[-500:])
        except Exception as e:
            logger.error("Benchmark subprocess error: %s", e)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return {
        "status": "started",
        "message": f"Benchmark running for {diseases} diseases (fast={fast}). Takes 5–30 min depending on Ollama speed.",
        "check_url": "/api/comparative-evaluation",
    }


@app.get("/api/admet/{drug}")
def get_admet(drug: str) -> dict:
    """Return ADMET/drug-likeness properties for a drug name via PubChem."""
    from drugenix.admet import get_admet_properties
    result = get_admet_properties(drug)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Drug '{drug}' not found in PubChem.")
    return result


@app.get("/api/clinical-trials/{drug}")
def get_clinical_trials(drug: str, disease: str = "", limit: int = 3) -> dict:
    """Return ClinicalTrials.gov results for a drug (optionally filtered by disease)."""
    from drugenix.clinical_trials import search_trials, get_highest_phase
    trials = search_trials(drug, disease, max_results=limit)
    return {
        "drug": drug,
        "disease": disease,
        "trials": trials,
        "trial_count": len(trials),
        "highest_phase": get_highest_phase(trials),
        "has_trials": len(trials) > 0,
    }


@app.get("/api/link-prediction/{disease}")
def link_prediction(disease: str, top_k: int = 10) -> dict:
    """Return TransE-predicted novel drug candidates for a disease."""
    import json
    from drugenix.kg_embeddings import get_embeddings_model

    model = get_embeddings_model()
    if not model.trained:
        raise HTTPException(
            status_code=503,
            detail="TransE model not trained yet. Run scripts/train_transe.py first.",
        )
    predictions = model.predict_novel_drugs(disease, top_k=top_k)
    return {
        "disease": disease,
        "model": "TransE",
        "predictions": predictions,
        "model_trained": model.trained,
    }


@app.get("/api/link-prediction-status")
def link_prediction_status() -> dict:
    """Check whether the TransE model is trained and ready."""
    from drugenix.kg_embeddings import MODEL_DIR
    meta_path = MODEL_DIR / "meta.json"
    trained = meta_path.exists()
    return {"trained": trained, "model_dir": str(MODEL_DIR)}


@app.post("/api/train-embeddings")
def train_embeddings_endpoint(epochs: int = 150, dim: int = 64) -> dict:
    """Train TransE KG embeddings in the background."""
    import subprocess
    import threading

    script = PROJECT_ROOT / "scripts" / "train_transe.py"

    def _run():
        try:
            result = subprocess.run(
                ["python", str(script), "--epochs", str(epochs), "--dim", str(dim)],
                capture_output=True, text=True, timeout=1800
            )
            logger.info("TransE training done: %s", result.stdout[-300:])
        except Exception as e:
            logger.error("TransE training error: %s", e)

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "message": f"TransE training started ({epochs} epochs, dim={dim}). Takes 2-5 min."}


@app.get("/api/case-studies")
def get_case_studies() -> list:
    """Return real case study validation results."""
    import json
    path = PROJECT_ROOT / "data" / "case_studies.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="Case studies not run yet. Run scripts/run_case_studies.py first.")


@app.get("/api/predict-benchmark")
def get_predict_benchmark() -> dict:
    """Return PREDICT-style benchmark results for TransE."""
    import json
    path = PROJECT_ROOT / "data" / "predict_benchmark.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="PREDICT benchmark not run yet. Run scripts/train_transe.py first.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
