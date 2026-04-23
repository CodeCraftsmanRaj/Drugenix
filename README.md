# Drugenix

MVP implementation of your **Retrieval + Reasoning (RAG-based)** stage for drug repurposing.

Given a disease query, this prototype:
1. retrieves relevant PubMed evidence,
2. reranks top abstracts,
3. generates candidate drugs with mechanism explanations and PMID-backed evidence.

## What is implemented now (your point #2)

- **Retrieval module**: PubMed search + abstract fetch using NCBI E-Utilities.
- **Reasoning module**:
  - primary mode: LLM-based reasoning (if `OPENAI_API_KEY` is set),
  - fallback mode: deterministic heuristic extraction from abstracts.
- **Pipeline orchestration**: end-to-end query -> candidates.
- **CLI**: run from terminal and optionally save JSON output.

## Project structure

- [main.py](main.py) - CLI entrypoint
- [drugenix/pipeline.py](drugenix/pipeline.py) - orchestration
- [drugenix/retrieval.py](drugenix/retrieval.py) - PubMed retrieval + reranking
- [drugenix/reasoning.py](drugenix/reasoning.py) - LLM/heuristic candidate generation
- [drugenix/models.py](drugenix/models.py) - shared dataclasses

## Run

### 1) Install dependencies

This MVP uses only Python standard library.

### 2) (Optional) Configure LLM for stronger reasoning

Set environment variables:

- `OPENAI_API_KEY`
- `OPENAI_MODEL` (optional, default: `gpt-4.1-mini`)
- `OPENAI_BASE_URL` (optional, default: `https://api.openai.com/v1`)

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

## Output format

The JSON contains:

- `query`
- `retrieved`: list of PubMed docs (`pmid`, `title`, `abstract`, `journal`, `year`, `retrieval_score`)
- `candidates`: list of candidate drugs (`drug`, `mechanism`, `confidence`, `evidence_pmids`)

## Temporary MVP evaluation (aligned with your point #3)

For this stage, track:

- **Top-k retrieval relevance** (manual relevance checks on retrieved abstracts)
- **Qualitative candidate quality** (biomedical plausibility + evidence grounding)
- **Hallucination proxy**: fraction of generated claims without matching PMID evidence in context

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
