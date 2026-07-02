# Drugenix: Agent Technical Details & Algorithms

## System Architecture Overview
```
┌─────────────────────────────────────────────────────────┐
│                 User Query (Natural Language)            │
│        "Find repurposed drugs for Alzheimer disease"    │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  1️⃣  RETRIEVAL AGENT         │
        │  (PubMed E-Utilities)        │
        │  - Query normalization       │
        │  - Abstract fetching         │
        │  - Relevance reranking       │
        │  Output: ~40 PubMed papers   │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  2️⃣  REASONING AGENT         │
        │  (LLM + Heuristic Fallback)  │
        │  - Context building from     │
        │    abstracts                 │
        │  - LLM-based generation or  │
        │    heuristic extraction      │
        │  Output: 8 drug candidates   │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  3️⃣  VALIDATION AGENT        │
        │  (Rule-Based Checks)         │
        │  - Name plausibility         │
        │  - Evidence grounding        │
        │  - Safety keywords scan      │
        │  - Confidence thresholding    │
        │  Output: Validated candidates│
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   Final Report & Metrics     │
        │   JSON, Markdown, Plots      │
        └──────────────────────────────┘
```

---

## 1️⃣ RETRIEVAL AGENT

### Purpose
Fetch relevant biomedical literature from PubMed that discusses therapies for the target disease.

### Algorithm

#### Step 1: Query Normalization
**Problem:** Users ask in natural language, PubMed needs precise disease terms.

**Solution:** Strip common prefixes and suffixes, extract disease term after "for" keyword.

```
Input:  "Find repurposed drug candidates for Alzheimer disease with evidence from PubMed"
         ↓ Remove prefix "find repurposed drug candidates for"
         ↓ Remove suffix "with evidence from pubmed"
Output: "Alzheimer disease"
```

**Clean Patterns Used:**
- Prefixes: "find ", "suggest ", "repurposed drug candidates for "
- Suffixes: " with evidence from pubmed", " please"
- Removes: "drug candidates", "repurposed drugs" keywords

#### Step 2: PubMed Search
**Query Construction:**
```
NCBI E-Search with MeSH Terms:
  (Alzheimer disease) AND (drug therapy[MeSH Terms] OR therapeutics[Subheading])
```

**Parameters:**
- `retmax=40`: Retrieve top 40 most relevant articles (ranked by relevance)
- `sort=relevance`: Use NCBI's relevance algorithm
- Returns: List of 40 PubMed IDs (PMIDs)

#### Step 3: Abstract Fetching
**Process:**
```
PMID List → E-Fetch XML → Parse XML → Extract:
  - PMID (ID)
  - ArticleTitle
  - Abstract text (handle multiple sections: Background, Methods, Results, etc.)
  - Journal name
  - Publication year
```

**Output Format per Document:**
```python
@dataclass
class EvidenceDoc:
    pmid: str              # "12345678"
    title: str             # "Beta-secretase inhibitors for Alzheimer disease"
    abstract: str          # Full text with section labels
    journal: str           # "Nature Medicine"
    year: str              # "2023"
```

#### Step 4: Relevance Reranking
**Algorithm: Token Overlap Scoring**

```python
def overlap_score(query: str, title: str, abstract: str) -> float:
    # Tokenize both query and document
    query_terms = tokenize("Alzheimer disease")  
                  # → ["alzheimer", "disease"]
    
    doc_terms = tokenize(title + abstract)
                # → [all words from title + abstract]
    
    # Count overlapping terms
    overlap = sum(min(query_count[term], doc_count[term]) 
                  for term in query_terms)
    
    # Normalize: overlap / number of unique query terms
    score = overlap / len(query_terms)
    
    return score  # 0.0 to 1.0
```

**Reranking:**
- Retrieve papers with high title/abstract overlap with disease term
- Keep top 8-20 papers with highest overlap scores
- Pass to next agent

---

## 2️⃣ REASONING AGENT

### Purpose
Generate a ranked list of drug candidates that could be repurposed for the disease, based on retrieved evidence.

### Algorithm

#### Option A: LLM-Based Reasoning (Default, if API available)
**Providers Supported:**
- OpenAI (GPT-4, GPT-4 mini)
- Gemini (Google)
- Ollama (local: llama3.1:8b, llama3.2)
- OpenAI-compatible (OpenRouter, Groq, Together, etc.)

**Process:**
```
1. Build context snippet from top 8-20 papers:
   Each paper contributes: PMID, Title, Journal, Year, 700-char abstract excerpt
   Total context: ~10-15 KB of biomedical text

2. Construct LLM prompt:
   System: "You are a biomedical reasoning assistant. 
            Only propose drugs supported by provided evidence snippets. 
            Always return valid JSON only."
   
   User: [Context of papers] + 
         "For {disease}, list 8 drug candidates with confidence 0-100. 
          Return JSON: {drugs: [{name, confidence, reasoning}]}"

3. Call LLM (temperature=0.2 for consistency):
   - Low temperature keeps output grounded and less hallucinated
   - response_format=json_object enforces JSON output

4. Parse JSON response:
   Extract drug name and confidence score
   Create DrugCandidate objects
```

**Advantages:**
- Leverages LLM reasoning from abstracts
- Contextual understanding
- Works with any OpenAI-compatible API

**Example Output:**
```json
{
  "drugs": [
    {"name": "Donepezil", "confidence": 95, "reasoning": "Acetylcholinesterase inhibitor, demonstrated in PMID 123456"},
    {"name": "Memantine", "confidence": 88, "reasoning": "NMDA antagonist, Phase 3 trials mentioned in PMID 234567"},
    ...
  ]
}
```

#### Option B: Heuristic Fallback (if LLM unavailable)
**Triggered:** When LLM call fails (API down, no credentials, network timeout)

**Process:**
```
1. Extract drug-like named entities from abstracts:
   Regex pattern: \b([A-Za-z][A-Za-z0-9\-]{2,})\b
   Example: "donepezil", "memantine", "aducanumab"

2. Filter by drug name plausibility:
   - Check if matches WHO INN suffixes:
     ("mab", "nib", "statin", "pril", "sartan", "prazole")
   - Check if in known generic drug database (~350 drugs)
   - Check length >= 3 characters, not common words

3. Score by co-occurrence in context:
   - Count how many times drug-like term appears
   - Higher frequency → higher confidence
   - Weight by proximity to disease term

4. Rank by confidence (heuristic: term frequency × suffix match)
```

**Example:**
```
Abstract: "Donepezil and rivastigmine, acetylcholinesterase inhibitors, 
          are FDA-approved for Alzheimer disease..."

Extraction: donepezil (appears 1x, suffix match) → high confidence
            rivastigmine (appears 1x, suffix match) → high confidence
            FDA (proper noun, not drug-like) → filtered out
```

#### Output Structure
```python
@dataclass
class DrugCandidate:
    drug: str                 # "Donepezil"
    confidence: float         # 0-100 (95.0)
    pmid_count: int          # papers mentioning this drug (5)
    mentions: list[str]       # ["PMID:123456", "PMID:234567", ...]
```

---

## 3️⃣ VALIDATION AGENT

### Purpose
Quality-check candidates for name plausibility, evidence strength, safety concerns, and confidence credibility.

### Multi-Stage Validation Pipeline

#### Stage 1: Name Plausibility Check
**Goal:** Ensure the proposed drug name is real, not hallucinated.

**Rules:**
```python
# Rule 1: Suffix-based validation (WHO INN naming conventions)
APPROVED_SUFFIXES = (
    "mab" → monoclonal antibody      (Trastuzumab, Pembrolizumab)
    "nib" → kinase inhibitor          (Imatinib, Sorafenib)
    "statin" → HMG-CoA reductase      (Atorvastatin)
    "pril" → ACE inhibitor            (Lisinopril)
    "sartan" → AT1 antagonist         (Losartan)
    "azole" → antifungal/PPI          (Omeprazole)
    ...20+ more suffixes
)

# Rule 2: Known drugs database (~350 common drugs)
KNOWN_DRUGS = frozenset({
    "aspirin", "ibuprofen", "insulin", "metformin", 
    "warfarin", "caffeine", "morphine", "tamoxifen",
    ...
})

if drug_name.lower() in KNOWN_DRUGS:
    ✓ VALID
else if any(drug_name.lower().endswith(suffix) for suffix in APPROVED_SUFFIXES):
    ✓ VALID
else:
    ✗ INVALID or WARN depending on confidence
```

**Check Level:**
- Confidence < 50: WARN ("Name needs verification")
- Confidence >= 50: PASS if suffix/known, FAIL if not

#### Stage 2: Evidence Grounding Check
**Goal:** Ensure the drug is mentioned in the retrieved abstracts with supporting evidence.

**Algorithm:**
```python
pmid_count = number of unique PMIDs mentioning the drug

if pmid_count >= 3:
    Status: PASS (good evidence)
    Confidence adjustment: +5%
    Message: "Cited in 3+ papers"
    
elif pmid_count == 1:
    Status: WARN (thin evidence)
    Confidence adjustment: -15%
    Message: "Only 1 paper mentions this drug"
    
elif pmid_count == 0:
    Status: FAIL (no evidence)
    Confidence: → 0%
    Message: "No supporting evidence in retrieved abstracts"
    ✗ FILTERED OUT
```

**Why This Matters:**
Drug mentioned in 5+ papers → likely legitimate repurposing candidate
Drug only in 1 paper → might be speculative or off-topic
Drug not mentioned → pure hallucination

#### Stage 3: Safety Keyword Scan
**Goal:** Flag if abstracts contain red-flag safety terms.

**Red Flags Detected:**
```
CONTRAINDICATION KEYWORDS:
  ["contraindicated", "adverse effect", "toxicity", "hepatotoxicity",
   "nephrotoxicity", "cardiotoxicity", "teratogenic", "embryotoxic"]

ALLERGY KEYWORDS:
  ["allergy", "hypersensitivity", "anaphylaxis", "rash", "urticaria"]

SAFETY CONCERNS:
  ["severe side effect", "black box warning", "death", "fatal",
   "discontinue", "withdraw", "contraindicate"]

ACTION:
if red_flag_count >= 2:
    Status: WARN (significant concerns)
    Confidence: -10% to -25%
    Warnings: ["Hepatotoxicity reported", "Black box warning noted"]
    
else if red_flag_count == 1:
    Status: WARN (minor concern)
    Confidence: -5%
    Warnings: ["Single adverse event mentioned"]
    
else:
    Status: OK (no safety flags)
```

#### Stage 4: Confidence Threshold Gate
**Goal:** Filter out low-confidence candidates (< 30% by default).

```python
if adjusted_confidence < 30:
    Status: FAIL
    Action: REMOVE from final recommendations
    Reason: "Confidence below 30% threshold"
    
elif 30 <= adjusted_confidence < 50:
    Status: WARN
    Action: INCLUDE but flag for review
    Message: "Low confidence; needs manual review"
    
else:  # >= 50
    Status: PASS
    Action: INCLUDE in recommendations
```

#### Stage 5: Cross-Candidate Deduplication
**Goal:** Collapse similar drug names (spelling variants, brand names).

**Algorithm:**
```python
# Simple string similarity for variants
"Aspirin" vs "aspirin" → Same (lowercase comparison)
"Amlodipine" vs "Amlodipine besylate" → Similar (remove salt/form)

# Keep highest confidence version, merge PMID evidence
{
  "aspirin": {confidence: 85, mentions: [PMID1, PMID2, PMID3, PMID4]},
  "ASPIRIN": {confidence: 75, mentions: [PMID5]}
}
→ Merge into single entry: confidence=85, mentions=[all 5 PMIDs]
```

### Validation Output
```python
@dataclass
class ValidationResult:
    drug: str              # "Donepezil"
    status: str           # "PASS" | "WARN" | "FAIL"
    confidence: float     # Original: 95.0
    adjusted_confidence: float  # After validation: 92.0
    checks: dict          # {
                          #   "name_plausible": True,
                          #   "evidence_grounded": True,  (3 PMIDs)
                          #   "safety_clean": True,
                          #   "confidence_threshold": True
                          # }
    warnings: list        # ["Minor adverse event in 1 abstract"]
    summary: str          # "Passes all validation checks"
```

---

## 4️⃣ PIPELINE ORCHESTRATION

```python
def run_pipeline(disease_query: str, retmax=40, top_k=8):
    # Stage 1: Retrieval
    retrieved_docs = retrieve_pubmed(disease_query, retmax, top_k)
    # → [EvidenceDoc, EvidenceDoc, ...]  (8-20 papers)
    
    # Stage 2: Reasoning
    candidates = generate_candidates(disease_query, retrieved_docs)
    # → [DrugCandidate, DrugCandidate, ...]  (8 candidates)
    
    # Stage 3: Validation
    validation_results = validate_candidates(candidates, retrieved_docs)
    # → [ValidationResult, ValidationResult, ...]  (8 results)
    
    # Compute validation statistics
    stats = validation_summary_stats(validation_results)
    # → {"pass_count": 6, "warn_count": 2, "fail_count": 0, ...}
    
    # Return complete pipeline output
    return {
        "query": disease_query,
        "runtime": {llm_provider, model, etc.},
        "retrieved": [papers with all metadata],
        "candidates": [candidates with confidence],
        "validation": {
            "results": [detailed validation per drug],
            "stats": {aggregate statistics}
        }
    }
```

---

## 5️⃣ PENDING AGENTS (Future Implementation)

### ❌ Planner Agent (Not Yet Implemented)
**Purpose:** Decompose complex multi-disease queries into sub-tasks.

**Example:**
- Input: "Find drugs for Parkinson's disease or Alzheimer's disease"
- Decompose: Task 1: Search for Parkinson's drugs, Task 2: Search for Alzheimer's drugs
- Aggregate: Final ranked list of all candidates

**Algorithm:** Query intent recognition + task tree generation

---

### ❌ Aggregation/Explanation Agent (Not Yet Implemented)
**Purpose:** Final ranking and generate human-readable explanations for why each drug was selected.

**Features:**
- Cross-candidate ranking (combine validation scores + evidence metrics)
- Traceability: Show which PMID supports each claim
- Natural language summary per drug
- Confidence + evidence + reasoning chain

**Example Output:**
```
🏆 #1: Donepezil (93% confidence)
   └─ Rationale: FDA-approved acetylcholinesterase inhibitor
   └─ Evidence: PMID:12345678, PMID:23456789, PMID:34567890
   └─ Supporting Facts: 
       • Improves cognition in mild-to-moderate Alzheimer's disease
       • Long safety track record (20+ years)
       • Mechanism: Blocks breakdown of acetylcholine
```

---

## 6️⃣ METRICS & EVALUATION

### Computed Metrics (from evaluation scripts)

**Precision@K:** 
```
Fraction of top-K predictions that are correct repurposing candidates
P@8 = (number of correct drugs in top 8) / 8
Example: 7/8 correct → Precision@8 = 0.875
```

**nDCG@K (Normalized Discounted Cumulative Gain):**
```
Rewards high-ranking predictions more than low-ranking ones
Penalizes wrong drugs appearing early
nDCG@8 ranges from 0 to 1 (1.0 = perfect ranking)
```

**Top-K Accuracy:**
```
Does the gold standard drug appear in top K predictions?
Top-8 Accuracy = 1 if gold drug in top 8, else 0
Averaged across all queries
```

**Groundedness Score:**
```
For each candidate drug:
  Groundedness = (# PMIDs mentioning drug) / (# retrieved papers)
Example: Drug appears in 5 of 10 papers → 0.5 groundedness
Averaged: Mean groundedness across all candidates
Typical range: 0.8-1.0 (most candidates are grounded in evidence)
```

**Ablation Study:**
```
Remove validation stage, compare performance
Shows: Does validation help filter hallucinations?
Typical result: Validation removes low-confidence junk, improves precision
```

---

## 7️⃣ IMPLEMENTATION SUMMARY

| Agent | Status | Input | Output | Algorithm |
|-------|--------|-------|--------|-----------|
| **Retrieval** | ✅ Complete | Disease query (text) | 8-20 PubMed papers | Query norm + E-Utilities API + Token overlap ranking |
| **Reasoning** | ✅ Complete | Papers + disease | 8 drug candidates + confidence | LLM chat (OpenAI/Ollama) OR heuristic extraction |
| **Validation** | ✅ Complete | Candidates + papers | Validated candidates + status | 5-stage rule-based pipeline |
| **Planner** | ❌ Pending | Complex query | Task decomposition | Intent recognition + task graph |
| **Aggregation** | ❌ Pending | All candidates | Final ranked list + explanations | Cross-candidate scoring + traceback |

---

## 8️⃣ EXAMPLE RUN: Alzheimer's Disease

```
INPUT: "Find repurposed drug candidates for Alzheimer disease"

STAGE 1 – RETRIEVAL:
  Query normalized: "Alzheimer disease"
  Search: (Alzheimer disease) AND (drug therapy[MeSH Terms])
  Retrieved: 40 PubMed papers
  Top 8 by overlap: [PMID:1234, PMID:2345, ..., PMID:8901]

STAGE 2 – REASONING (LLM):
  Context: 8 abstracts (10 KB text) about Alzheimer therapies
  Prompt: "List 8 drug candidates for Alzheimer based on these papers"
  Response (JSON):
    [
      {"name": "Donepezil", "confidence": 95},
      {"name": "Memantine", "confidence": 88},
      {"name": "Rivastigmine", "confidence": 82},
      ...
    ]

STAGE 3 – VALIDATION:
  Donepezil:
    ✓ Name: matches "–epezil" suffix pattern → VALID
    ✓ Evidence: appears in 5 of 8 papers → GROUNDED
    ✓ Safety: no red flags → SAFE
    ✓ Confidence: 95% > 50% threshold → PASS
    ✓ Status: PASS | Adjusted confidence: 95%
  
  Memantine:
    ✓ All checks pass → PASS | Confidence: 88%
  
  Rivastigmine:
    ✓ All checks pass → PASS | Confidence: 82%
  
  [Other candidates...]

FINAL OUTPUT:
  ✅ Recommended (PASS): Donepezil, Memantine, Rivastigmine, ...
  ⚠️  Review (WARN): [Any with minor concerns]
  ❌ Rejected (FAIL): [Any with critical issues]
  
  Metrics:
    - Precision@8: 7/8 (87.5%)
    - Groundedness: 0.91 (91% of candidates grounded in evidence)
    - Validation stats: 7 PASS, 1 WARN, 0 FAIL
```

---

## Summary

**3-Stage Pipeline:**
1. **Retrieval:** Query → PubMed papers (live API, no local DB)
2. **Reasoning:** Papers → Drug candidates (LLM-based with heuristic fallback)
3. **Validation:** Candidates → Quality-checked list (5-stage rule pipeline)

**Key Features:**
- ✅ Evidence-grounded: All candidates linked to specific PMIDs
- ✅ Dynamic: Results change per query and retrieved evidence
- ✅ Robust: Heuristic fallback if LLM unavailable
- ✅ Safe: Red-flag scanning for adverse effects
- ✅ Measurable: 5 metrics computed per query
