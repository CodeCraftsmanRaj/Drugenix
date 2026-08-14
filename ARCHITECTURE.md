# Drugenix — System Architecture

**Agentic AI System for Accelerated Drug Repurposing**
Autonomous Multi-Agent System for Explainable Biomedical Reasoning

## 0. What's new: real-time, disease-agnostic, symptom-aware

Four additions on top of the architecture described below:

- **Live knowledge graph, any disease.** `drugenix/opentargets.py` queries the
  Open Targets Platform GraphQL API per-disease at request time (no API key,
  covers essentially any disease/phenotype in the EFO ontology). This is now
  the primary source `knowledge_graph.py` uses; the ~80-relationship curated
  seed list and pre-fetched ChEMBL dump are kept only as an offline fallback
  if Open Targets is unreachable. Nothing in the live query path is limited
  to a fixed disease list.
- **Symptom input mode.** `POST /api/symptom-query` (backed by
  `drugenix/symptom_reasoning.py`) lets a user describe symptoms instead of
  naming a disease. An LLM proposes ranked candidate conditions, each
  grounded against a live PubMed search before being handed to the normal
  pipeline. Explicitly a research-hypothesis tool, not a diagnostic one —
  every response carries `symptom_analysis.disclaimer`.
- **Mechanism-similarity fallback for new/rare diseases.**
  `drugenix/mechanism_fallback.py` triggers when PubMed literature for a
  query is thin (a genuinely new or very rare disease may have little to no
  published literature yet). It proposes drugs via shared molecular-target
  overlap with related diseases (using Open Targets' genetic/clinical
  association data, which exists independent of literature), tagged
  `evidence_tier="mechanistic"` end-to-end so these are never confused with
  literature-backed candidates — capped confidence, separate UI badge,
  excluded from the "zero PMIDs = hallucination" check.
- **Research-grade metrics.** `drugenix/evaluation.py` now reports bootstrap
  95% confidence intervals and sample size (`n`) alongside every aggregate
  metric. Each ranked candidate in a pipeline result also carries a live
  `groundedness` score (fraction of its stated mechanism's key terms found
  verbatim in the cited PubMed abstracts) — combined with the existing
  confidence, KG path, ADMET verdict, and clinical-trial phase, this is the
  shortlist a researcher needs before deciding what to pursue further.

## 1. Overview

Drugenix is a multi-agent pipeline that takes a disease query and produces ranked,
evidence-grounded, explainable drug repurposing hypotheses. It integrates literature
evidence (PubMed), structured biomedical knowledge (a drug-target-disease graph), and
named-entity confirmation, fusing all three through a Master Agent that performs
conflict detection and hallucination flagging.

```
                                User Query
                                    |
                                    v
                          ┌─────────────────┐
                          │  PLANNER AGENT   │  Decomposes multi-disease queries,
                          │                  │  runs sub-tasks in parallel,
                          └────────┬─────────┘  merges results.
                                   |
                 (one sub-task per disease, executed in parallel)
                                   |
                                   v
                      ┌─────────────────────┐
                      │   RETRIEVAL AGENT    │  PubMed E-Utilities search +
                      │                      │  semantic reranking
                      │  + Semantic Layer    │  (sentence-transformers + ChromaDB)
                      └──────────┬───────────┘
                                 |
                                 v
                      ┌─────────────────────┐
                      │     NER AGENT        │  spaCy-based biomedical entity
                      │                      │  extraction: drugs, targets,
                      │                      │  drug-target co-occurrence pairs
                      └──────────┬───────────┘
                                 |
                                 v
                      ┌─────────────────────┐
                      │   REASONING AGENT    │  LLM (OpenAI/Gemini/Ollama) or
                      │                      │  heuristic candidate generation
                      └──────────┬───────────┘
                                 |
                                 v
                      ┌─────────────────────┐
                      │  VALIDATION AGENT    │  Name plausibility, evidence
                      │                      │  grounding, safety keyword scan,
                      │                      │  confidence threshold, dedup
                      └──────────┬───────────┘
                                 |
                                 v
                      ┌─────────────────────┐
                      │  KNOWLEDGE GRAPH     │  NetworkX drug→target→disease
                      │                      │  graph (curated + CTD data).
                      │                      │  Validates biological plausibility
                      │                      │  via direct/indirect graph paths.
                      └──────────┬───────────┘
                                 |
                                 v
                      ┌─────────────────────┐
                      │   MASTER AGENT       │  Weighted evidence fusion:
                      │  (Aggregation /      │    Literature   40%
                      │   Explanation Agent) │    Validation    25%
                      │                      │    Knowledge Graph 20%
                      │                      │    NER            15%
                      │                      │  Conflict detection,
                      │                      │  hallucination flagging,
                      │                      │  reasoning trace generation.
                      └──────────┬───────────┘
                                 |
                                 v
                  Ranked Output: Recommended / Review / Rejected
                  + full evidence-linked reasoning trace per drug
```

## 2. Agent Responsibilities

### Planner Agent (`drugenix/planner.py`)
Detects multi-disease queries ("drugs for X and Y"), decomposes them into independent
sub-tasks, executes each through the full pipeline in parallel (`ThreadPoolExecutor`),
and merges the results into a single response with per-disease attribution.

### Retrieval Agent (`drugenix/retrieval.py` + `drugenix/semantic_retrieval.py`)
1. Normalizes the natural-language query into a PubMed search term.
2. Searches NCBI E-Utilities (`esearch` + `efetch`) for relevant abstracts.
3. Reranks results semantically: abstracts are embedded with
   `sentence-transformers/all-MiniLM-L6-v2` and stored in a ChromaDB collection;
   the query embedding is matched against stored documents via cosine similarity.

### NER Agent (`drugenix/ner_extraction.py`)
Uses spaCy (biomedical model when available, falling back to `en_core_web_sm`) plus
curated WHO-INN suffix rules and a gene-symbol gazetteer to extract:
- Drug entities (normalized, salt-form stripped)
- Target/gene entities (e.g. EGFR, ACHE, SGLT2)
- Drug–target co-occurrence pairs from sentence-level proximity

This output independently confirms (or fails to confirm) what the Reasoning Agent
proposes — one of the four signals fused by the Master Agent.

### Reasoning Agent (`drugenix/reasoning.py`)
Primary mode: LLM-based generation (OpenAI / Gemini / Ollama / OpenAI-compatible),
constrained to propose drugs only from retrieved evidence, returning structured JSON
with mechanism explanations and PMID citations.
Fallback mode: deterministic heuristic extraction (regex + WHO-INN suffix matching +
co-occurrence scoring) used when no LLM credentials are configured.

### Validation Agent (`drugenix/validation.py`)
Five-stage rule pipeline: name plausibility, evidence grounding (PMID count),
safety keyword scanning, confidence thresholding, and near-duplicate deduplication.
Produces PASS / WARN / FAIL per candidate.

### Knowledge Graph (`drugenix/knowledge_graph.py`)
A `networkx.DiGraph` of drug → target → disease relationships, seeded from curated
mechanism data (30+ well-known repurposing cases) and optionally extended with CTD
(Comparative Toxicogenomics Database) chemical-disease associations. For each
candidate, checks for a direct drug→disease edge or an indirect path through a known
target, returning path length and mechanism as biological-plausibility evidence.

### Master Agent (`drugenix/aggregation.py`)
The "Aggregation / Explanation Agent" from the project proposal. For each candidate:
1. Collects scores from all four sources (literature, validation, KG, NER).
2. Computes a weighted final confidence score.
3. Detects conflicts (e.g., LLM proposes a drug with no KG path — flagged as a
   potential hallucination).
4. Builds a step-by-step reasoning trace explaining how the final score was reached.
5. Categorizes candidates into Recommended / Review Needed / Rejected.

## 3. Comparative Evaluation

`scripts/comparative_evaluation.py` implements the three-way experiment promised in
the project proposal: **Pure LLM vs Pure Graph vs Hybrid Pipeline**, each evaluated
against a curated gold-standard drug list per disease using Hit@k, Precision@k, mean
confidence, mean groundedness, and hallucination rate. Results are saved to
`data/comparative_evaluation.json` and plotted to `plots/comparative_evaluation.png`.

## 4. Interfaces

- **CLI** (`main.py`): terminal-based pipeline execution with structured report output.
- **REST API** (`api/main.py`, FastAPI): wraps the full pipeline, including the
  Planner Agent, with endpoints for querying, history, knowledge graph exploration,
  and metrics.
- **Web UI** (`frontend/`, React + Vite): dashboard, query interface, results
  explorer with evidence traces, knowledge graph visualization (force-directed
  graph), query history, and an evaluation metrics dashboard.

## 5. Data Sources

- **PubMed** (NCBI E-Utilities) — live literature retrieval, no local copy required.
- **CTD** (Comparative Toxicogenomics Database) — optional chemical-disease
  association download for extending the knowledge graph beyond curated data.
- **Curated drug-target-disease relationships** — 30+ well-documented repurposing
  cases spanning Alzheimer's, Parkinson's, diabetes, cancer, cardiovascular,
  respiratory, and autoimmune conditions.
