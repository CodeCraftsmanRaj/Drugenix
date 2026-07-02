# Drugenix — Agentic AI System for Drug Repurposing

> **Major Project · Fourth Year · Biomedical AI**  
> Multi-agent pipeline that identifies repurposing opportunities for approved drugs using PubMed evidence, biomedical NER, Knowledge Graph validation, and KG embeddings.

---

## Pipeline Architecture

```
Query
  │
  ▼
┌──────────────────┐
│  Planner Agent   │  Decomposes multi-disease queries into sub-tasks
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Retrieval Agent  │  PubMed API + semantic reranking (sentence-transformers)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   NER Agent      │  Biomedical entity extraction (HuggingFace d4data/biomedical-ner-all)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Reasoning Agent  │  LLM/heuristic candidate generation (Ollama / OpenAI / Gemini)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│Validation Agent  │  Plausibility · evidence · safety · hallucination checks
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│Knowledge Graph   │  Drug→Target→Disease paths (NetworkX + ChEMBL + CTD)
│    Agent         │  + TransE link prediction (PyKEEN)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Master Agent    │  Evidence fusion · final ranking · ADMET filter · ClinicalTrials.gov
└──────────────────┘
```

---

## Results & Benchmark

### Three-Way Comparative Evaluation (15 diseases)

| System | Hit@1 | Hit@5 | Hit@10 | MRR | AUROC |
|--------|-------|-------|--------|-----|-------|
| Pure LLM | 0.21 | 0.44 | 0.56 | 0.31 | 0.64 |
| Pure Graph | 0.27 | 0.52 | 0.67 | 0.38 | 0.71 |
| **Drugenix Hybrid** | **0.41** | **0.73** | **0.87** | **0.55** | **0.83** |

### TransE KG Embedding (PREDICT-style Benchmark)

| Metric | Score |
|--------|-------|
| AUROC | **0.883** |
| Hit@1 | 0.200 |
| Hit@5 | 0.733 |
| Hit@10 | **0.933** |
| Coverage | 1.000 |
| Test set | 133 positive / 399 negative pairs (ChEMBL Phase 4) |

### Comparison with Published Methods

| Method | AUROC | Source |
|--------|-------|--------|
| PREDICT (Gottlieb et al.) | 0.900 | Mol Syst Biol 2011 |
| BioKG-BERT | 0.871 | Sung et al. 2021 |
| DRKG (Drug Repurposing KG) | 0.861 | Zitnik et al. 2018 |
| **Drugenix TransE** | **0.883** | This work |
| Drugenix Hybrid Pipeline | **0.830** | This work |

> Our TransE implementation is competitive with DRKG and BioKG-BERT, and within 2% of the original PREDICT method — validated on ChEMBL Phase 4 FDA-approved drug-disease pairs.

---

## Knowledge Graph

| Component | Count |
|-----------|-------|
| Total nodes | 1,652 |
| Drugs | 1,577 |
| Protein targets | 50 |
| Diseases | 25 |
| Edges | 2,306 |
| ChEMBL drug-disease pairs | 2,161 |
| Data sources | ChEMBL · CTD · curated |

---

## Case Study Validations

| Drug | Original Use | Repurposing Target | Evidence | KG Path |
|------|--------------|--------------------|----------|---------|
| Metformin | Type 2 Diabetes | Cancer (AMPK/mTOR) | 50+ meta-analyses | Direct |
| Sildenafil | Erectile Dysfunction | Pulmonary Arterial Hypertension | SUPER-1 RCT (2005) | Direct |
| Thalidomide | Morning Sickness | Multiple Myeloma | FDA-approved 2006 | Direct |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI · Python 3.11 |
| LLM | Ollama (llama3.2) · OpenAI · Gemini |
| NER | HuggingFace `d4data/biomedical-ner-all` |
| Retrieval | PubMed E-utilities · sentence-transformers · ChromaDB |
| Knowledge Graph | NetworkX · ChEMBL REST API · CTD |
| KG Embeddings | PyKEEN TransE |
| ADMET | PubChem REST API (Lipinski Ro5 + Veber) |
| Clinical Trials | ClinicalTrials.gov v2 API |
| Frontend | React 18 · Vite · Recharts · react-force-graph-2d |

---

## Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- [Ollama](https://ollama.ai) (optional — for local LLM reasoning)

### Backend

```bash
git clone https://github.com/your-username/drugenix.git
cd drugenix
pip install -e ".[dev]"

# Fetch ChEMBL data (builds knowledge graph)
python scripts/fetch_chembl.py

# Train TransE embeddings (optional, ~5 min)
python scripts/train_transe.py

# Start backend
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# -> http://localhost:5173
```

### With Ollama (local LLM)

```bash
ollama pull llama3.2
# Backend auto-detects Ollama if running on localhost:11434
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/query` | Run full pipeline |
| GET | `/api/history` | List past queries |
| GET | `/api/knowledge-graph/{disease}` | KG subgraph |
| GET | `/api/metrics` | Evaluation metrics |
| GET | `/api/comparative-evaluation` | 3-way benchmark |
| GET | `/api/link-prediction/{disease}` | TransE novel predictions |
| GET | `/api/admet/{drug}` | ADMET properties (PubChem) |
| GET | `/api/clinical-trials/{drug}` | ClinicalTrials.gov lookup |
| GET | `/api/predict-benchmark` | PREDICT-style benchmark |
| GET | `/api/case-studies` | Landmark case study results |

---

## Project Structure

```
drugenix/
├── api/               # FastAPI backend
├── drugenix/
│   ├── pipeline.py        # Main orchestration (7 agents)
│   ├── retrieval.py       # PubMed + semantic reranking
│   ├── ner_extraction.py  # Biomedical NER
│   ├── reasoning.py       # LLM candidate generation
│   ├── validation.py      # Plausibility checks
│   ├── knowledge_graph.py # NetworkX KG
│   ├── kg_embeddings.py   # PyKEEN TransE
│   ├── admet.py           # PubChem ADMET scoring
│   ├── clinical_trials.py # ClinicalTrials.gov API
│   ├── planner.py         # Multi-disease planning
│   └── aggregation.py     # Master agent
├── scripts/
│   ├── fetch_chembl.py    # Build ChEMBL KG
│   ├── train_transe.py    # Train + evaluate TransE
│   ├── run_benchmark.py   # 15-disease evaluation
│   └── run_case_studies.py
├── frontend/          # React + Vite UI
└── data/              # KG, embeddings, benchmark results
```

---

## References

- Gottlieb A, et al. *PREDICT: a method for inferring novel drug indications.* Mol Syst Biol. 2011.
- Zitnik M, et al. *Modeling polypharmacy side effects with graph convolutional networks.* Bioinformatics. 2018.
- Sung C, et al. *Pre-trained language model for biomedical KG link prediction.* arXiv. 2021.
- Bordes A, et al. *Translating embeddings for modeling multi-relational data (TransE).* NeurIPS. 2013.
- Ali M, et al. *PyKEEN 1.0: A Python library for KG embeddings.* JMLR. 2021.
