# Drugenix

**RAG-based Drug Repurposing AI** — three-stage agentic pipeline.

Given a disease query, this system:
1. **Retrieves** relevant PubMed evidence (Retrieval Agent),
2. **Reasons** over abstracts to generate candidate drugs with mechanism explanations and PMID-backed evidence (Reasoning Agent),
3. **Validates** each candidate for name plausibility, evidence grounding, and safety signals (Validation Agent).

## What is implemented

- **Retrieval Agent** (`drugenix/retrieval.py`): PubMed search + abstract fetch + reranking via NCBI E-Utilities.
- **Reasoning Agent** (`drugenix/reasoning.py`):
  - primary mode: LLM-based reasoning (OpenAI / Gemini / Ollama),
  - fallback mode: deterministic heuristic extraction from abstracts.
- **Validation Agent** (`drugenix/validation.py`) — *newly added*:
  - Name plausibility check against WHO INN drug suffixes + known drug list + stopword filter.
  - Evidence grounding check (PMID count → weak / moderate / strong).
  - Safety keyword scan (toxicity, adverse effects, contraindications, etc.).
  - Confidence threshold gate (auto-FAIL below minimum).
  - Near-duplicate drug name deduplication.
  - Outputs `PASS`, `WARN`, or `FAIL` with adjusted confidence for each candidate.
- **Pipeline orchestration** (`drugenix/pipeline.py`): end-to-end query → candidates → validated results.
- **CLI** (`main.py`): run from terminal, view structured report, optionally save JSON.

## Project structure

- [main.py](main.py) — CLI entrypoint
- [drugenix/pipeline.py](drugenix/pipeline.py) — orchestration (Retrieval → Reasoning → Validation)
- [drugenix/retrieval.py](drugenix/retrieval.py) — PubMed retrieval + reranking
- [drugenix/reasoning.py](drugenix/reasoning.py) — LLM/heuristic candidate generation
- [drugenix/validation.py](drugenix/validation.py) — **Validation Agent** *(new)*
- [drugenix/models.py](drugenix/models.py) — shared dataclasses (EvidenceDoc, DrugCandidate, ValidationResult)

## Run

### 1) Install dependencies

This MVP uses only Python standard library.

### 2) (Optional) Configure LLM for stronger reasoning

Set environment variables:

- `OPENAI_API_KEY`
- `OPENAI_MODEL` (optional, default: `gpt-4.1-mini`)
- `OPENAI_BASE_URL` (optional, default: `https://api.openai.com/v1`)

Additional providers are supported:

- **Gemini** (includes `gemmini` alias support):
	- `LLM_PROVIDER=gemini`
	- `GEMINI_API_KEY`
	- `GEMINI_MODEL` (default: `gemini-2.0-flash`)
	- `GEMINI_BASE_URL` (default: `https://generativelanguage.googleapis.com/v1beta/openai`)

- **Ollama**:
	- `LLM_PROVIDER=ollama`
	- `OLLAMA_MODEL` (example: `llama3.2:latest`, `qwen3.5:4b`)
	- `OLLAMA_BASE_URL` (default: `http://localhost:11434/v1`)

- **OpenAI-compatible custom providers** (OpenRouter/Groq/Together etc.):
	- `LLM_PROVIDER=<name>`
	- `LLM_API_KEY`
	- `LLM_MODEL`
	- `LLM_BASE_URL`

If no API key is set, the system automatically uses heuristic reasoning.

### 3) Execute pipeline

Example:

```bash
python main.py "Alzheimer disease" --retmax 50 --top-k 10 --save-json outputs/alzheimer_mvp.json
```

Or with script entrypoint:

```bash
drugenix "Type 2 diabetes" --retmax 40 --top-k 8
```

Provider from CLI (preferred for experiments):

```bash
python main.py "Parkinson disease" --provider ollama --model llama3.2:latest --retmax 40 --top-k 8 --save-json outputs/parkinson_ollama.json
```

## Output format

The JSON contains:

- `query`
- `retrieved`: list of PubMed docs (`pmid`, `title`, `abstract`, `journal`, `year`, `retrieval_score`)
- `candidates`: list of candidate drugs (`drug`, `mechanism`, `confidence`, `evidence_pmids`)
- `validation`:
  - `results`: list of validated candidates (`drug`, `status`, `confidence`, `adjusted_confidence`, `checks`, `warnings`, `summary`)
  - `stats`: aggregate stats (`total`, `passed`, `warned`, `failed`, `pass_rate`, `avg_original_confidence`, `avg_adjusted_confidence`, `top_safety_flags`)

### Validation status values

| Status | Meaning |
|--------|---------|
| `PASS` | All checks passed; confidence maintained or boosted |
| `WARN` | Minor concern (thin evidence or safety flag); confidence penalised |
| `FAIL` | Critical issue (implausible name, no evidence, or very low confidence); rejected |

### CLI flag: skip validation

```bash
python main.py "Alzheimer disease" --no-validation
```

## Temporary MVP evaluation (aligned with your point #3)

For this stage, track:

- **Top-k retrieval relevance** (manual relevance checks on retrieved abstracts)
- **Qualitative candidate quality** (biomedical plausibility + evidence grounding)
- **Hallucination proxy**: fraction of generated claims without matching PMID evidence in context

## Metrics implemented for this stage

Retrieval metrics:

- `Precision@k`
- `Recall@k`
- `nDCG@k`

Generation/grounding metrics:

- Groundedness / Faithfulness (token support proxy)
- `Top-k Accuracy`

Utility scripts:

- [scripts/evaluate_mvp.py](scripts/evaluate_mvp.py)
- [scripts/plot_metrics.py](scripts/plot_metrics.py)
- [data/metrics_template.json](data/metrics_template.json)

Example evaluation run:

```bash
python scripts/evaluate_mvp.py --predictions outputs/preds.jsonl --qrels data/qrels.json --gold-drugs data/gold_drugs.json
```

Example plotting run:

```bash
python scripts/plot_metrics.py --metrics data/metrics_template.json --outdir plots
```

## Presentation plots covered

- Precision@k vs k
- nDCG@k
- Top-k Accuracy
- Groundedness histogram
- Ablation chart: Without RAG vs With RAG

## Literature benchmark ranges (for framing)

- Retrieval: `Precision@5 ≈ 0.6–0.8`, `nDCG@10 ≈ 0.65–0.85`
- Repurposing ranking: `Hit@10 ≈ 0.5–0.7`, `MRR ≈ 0.3–0.5`
- LLM groundedness with RAG: typically `60–80%` and lower without RAG

## Datasets you may need (download links)

For this MVP, **you can run without downloading any dataset** (live PubMed retrieval).  
But for scale/reproducibility and future validation layers, these are recommended:

1. **PubMed Baseline** (local/offline literature index)
	- https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/

2. **MeSH descriptors** (disease synonym expansion)
	- https://www.nlm.nih.gov/databases/download/mesh.html

3. **CTD: Chemical-Disease and Chemical-Gene associations** (validation/graph layer)
	- http://ctdbase.org/downloads/

4. **ChEMBL** (drug-target bioactivity for future validation)
	- https://ftp.ebi.ac.uk/pub/databases/chembl/

5. **DrugCentral** (drug indications/targets; practical for repurposing checks)
	- https://drugcentral.org/download

6. **Open Targets Platform data** (disease-target evidence)
	- https://platform.opentargets.org/downloads

## Notes

- This is intentionally modular so you can add:
  - Validation Agent (knowledge graph + safety constraints),
  - Aggregation/Explanation Agent (scoring + traceable reasoning),
  without rewriting retrieval/reasoning core.
