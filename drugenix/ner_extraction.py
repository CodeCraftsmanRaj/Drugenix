"""
Drugenix - Biomedical Named Entity Recognition
=================================================
Hybrid NER: HuggingFace biomedical BERT model (d4data/biomedical-ner-all)
+ comprehensive rule-based drug dictionary fallback.

Recognises drug names, gene/protein targets, and disease entities
from PubMed abstracts with high recall for drug repurposing contexts.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from drugenix.models import EvidenceDoc

logger = logging.getLogger(__name__)

_nlp = None
_hf_ner = None
_NER_MODEL = "d4data/biomedical-ner-all"


@dataclass
class BioEntity:
    text: str
    label: str  # DRUG, TARGET, DISEASE
    start: int = 0
    end: int = 0
    source_pmid: str = ""
    normalized: str = ""


@dataclass
class ExtractionResult:
    drugs: list[BioEntity] = field(default_factory=list)
    targets: list[BioEntity] = field(default_factory=list)
    diseases: list[BioEntity] = field(default_factory=list)
    drug_target_pairs: list[tuple[str, str]] = field(default_factory=list)
    drug_disease_pairs: list[tuple[str, str]] = field(default_factory=list)


# WHO INN stems — highly reliable drug suffixes
_DRUG_SUFFIXES = (
    "mab", "nib", "tinib", "ciclib", "parib", "sartan", "pril",
    "olol", "oxetine", "azole", "avir", "mycin", "cycline", "cillin",
    "vastatin", "statin", "prazole", "dipine", "tidine", "triptan",
    "setron", "lukast", "glitazone", "gliptin", "gliflozin", "oxacin",
    "parin", "phylline", "sterone", "olone", "fenac", "profen", "oxicam",
    "vudine", "navir", "tegravir", "buvir", "piravir", "previr",
    "zumab", "ximab", "mumab", "lizumab", "kinumab", "lumab",
    "zomib", "rafenib", "lasertib", "cetinib",
)

# Comprehensive FDA-approved + investigational drug dictionary
_KNOWN_DRUGS: frozenset[str] = frozenset({
    # Pain / anti-inflammatory
    "aspirin", "ibuprofen", "paracetamol", "acetaminophen", "naproxen",
    "celecoxib", "indomethacin", "diclofenac", "ketorolac", "piroxicam",
    "meloxicam", "tramadol", "morphine", "codeine", "fentanyl", "oxycodone",
    "hydrocodone", "buprenorphine", "naloxone", "naltrexone", "methadone",
    # Antibiotics
    "amoxicillin", "ampicillin", "penicillin", "ciprofloxacin", "levofloxacin",
    "doxycycline", "azithromycin", "clarithromycin", "erythromycin",
    "metronidazole", "clindamycin", "vancomycin", "linezolid", "rifampicin",
    "rifampin", "isoniazid", "ethambutol", "pyrazinamide", "tetracycline",
    "ceftriaxone", "cefazolin", "meropenem", "piperacillin", "minocycline",
    "trimethoprim", "sulfamethoxazole", "nitrofurantoin", "fosfomycin",
    # Antivirals
    "oseltamivir", "zanamivir", "baloxavir", "peramivir", "favipiravir",
    "remdesivir", "nirmatrelvir", "molnupiravir", "lopinavir", "ritonavir",
    "tenofovir", "emtricitabine", "efavirenz", "dolutegravir", "raltegravir",
    "abacavir", "lamivudine", "zidovudine", "nevirapine", "atazanavir",
    "acyclovir", "valacyclovir", "ganciclovir", "valganciclovir",
    "sofosbuvir", "ledipasvir", "daclatasvir", "asunaprevir", "glecaprevir",
    "hydroxychloroquine", "chloroquine", "ivermectin", "nitazoxanide",
    # Cardiovascular
    "metoprolol", "atenolol", "bisoprolol", "carvedilol", "propranolol",
    "losartan", "valsartan", "irbesartan", "candesartan", "olmesartan",
    "lisinopril", "enalapril", "ramipril", "captopril", "perindopril",
    "amlodipine", "nifedipine", "diltiazem", "verapamil", "felodipine",
    "hydrochlorothiazide", "furosemide", "spironolactone", "eplerenone",
    "digoxin", "amiodarone", "sotalol", "flecainide", "dronedarone",
    "warfarin", "heparin", "enoxaparin", "apixaban", "rivaroxaban",
    "dabigatran", "clopidogrel", "ticagrelor", "prasugrel", "aspirin",
    "atorvastatin", "rosuvastatin", "simvastatin", "pravastatin", "pitavastatin",
    "ezetimibe", "evolocumab", "alirocumab", "inclisiran",
    "sildenafil", "tadalafil", "vardenafil", "minoxidil",
    # Diabetes
    "metformin", "insulin", "glipizide", "glibenclamide", "gliclazide",
    "pioglitazone", "rosiglitazone", "sitagliptin", "saxagliptin", "alogliptin",
    "linagliptin", "vildagliptin", "empagliflozin", "dapagliflozin",
    "canagliflozin", "ertugliflozin", "liraglutide", "semaglutide",
    "exenatide", "dulaglutide", "albiglutide", "tirzepatide",
    "acarbose", "repaglinide", "nateglinide",
    # Oncology
    "tamoxifen", "letrozole", "anastrozole", "exemestane", "fulvestrant",
    "trastuzumab", "pertuzumab", "bevacizumab", "cetuximab", "panitumumab",
    "rituximab", "ofatumumab", "obinutuzumab", "daratumumab", "isatuximab",
    "pembrolizumab", "nivolumab", "ipilimumab", "atezolizumab", "durvalumab",
    "avelumab", "cemiplimab", "tremelimumab",
    "imatinib", "dasatinib", "nilotinib", "bosutinib", "ponatinib",
    "erlotinib", "gefitinib", "afatinib", "osimertinib", "lapatinib",
    "crizotinib", "alectinib", "brigatinib", "lorlatinib", "ceritinib",
    "vemurafenib", "dabrafenib", "trametinib", "cobimetinib", "binimetinib",
    "palbociclib", "ribociclib", "abemaciclib",
    "olaparib", "rucaparib", "niraparib", "talazoparib", "veliparib",
    "sorafenib", "sunitinib", "pazopanib", "axitinib", "cabozantinib",
    "regorafenib", "lenvatinib", "vandetanib",
    "bortezomib", "carfilzomib", "ixazomib",
    "thalidomide", "lenalidomide", "pomalidomide",
    "oxaliplatin", "cisplatin", "carboplatin", "fluorouracil", "capecitabine",
    "gemcitabine", "paclitaxel", "docetaxel", "cabazitaxel", "vinorelbine",
    "irinotecan", "topotecan", "etoposide", "doxorubicin", "epirubicin",
    "cyclophosphamide", "ifosfamide", "melphalan", "busulfan",
    "methotrexate", "pemetrexed", "raltitrexed",
    "temozolomide", "dacarbazine", "carmustine", "lomustine",
    "rapamycin", "sirolimus", "everolimus", "temsirolimus",
    "idelalisib", "copanlisib", "duvelisib", "alpelisib",
    # Neurology / Psychiatry
    "levodopa", "carbidopa", "pramipexole", "ropinirole", "rotigotine",
    "rasagiline", "selegiline", "safinamide", "entacapone", "tolcapone",
    "donepezil", "rivastigmine", "galantamine", "memantine",
    "amantadine", "biperiden", "trihexyphenidyl",
    "fluoxetine", "sertraline", "paroxetine", "citalopram", "escitalopram",
    "fluvoxamine", "venlafaxine", "duloxetine", "desvenlafaxine",
    "bupropion", "mirtazapine", "trazodone", "nefazodone",
    "amitriptyline", "clomipramine", "imipramine", "nortriptyline",
    "phenelzine", "tranylcypromine", "moclobemide",
    "quetiapine", "olanzapine", "risperidone", "aripiprazole", "clozapine",
    "haloperidol", "ziprasidone", "asenapine", "lurasidone", "brexpiprazole",
    "lithium", "valproate", "lamotrigine", "carbamazepine", "oxcarbazepine",
    "phenytoin", "levetiracetam", "topiramate", "zonisamide", "lacosamide",
    "gabapentin", "pregabalin", "phenobarbital", "clonazepam", "diazepam",
    "lorazepam", "alprazolam", "midazolam", "zolpidem", "zaleplon",
    "ketamine", "esketamine", "dextromethorphan",
    # Immunology / Autoimmune
    "adalimumab", "infliximab", "etanercept", "golimumab", "certolizumab",
    "tocilizumab", "sarilumab", "siltuximab", "anakinra", "canakinumab",
    "abatacept", "belimumab", "anifrolumab",
    "tofacitinib", "baricitinib", "upadacitinib", "filgotinib", "ruxolitinib",
    "azathioprine", "mycophenolate", "cyclosporine", "tacrolimus",
    "hydroxychloroquine", "sulfasalazine", "leflunomide", "colchicine",
    "prednisolone", "prednisone", "methylprednisolone", "dexamethasone",
    "hydrocortisone", "betamethasone", "triamcinolone",
    # Respiratory
    "salbutamol", "albuterol", "formoterol", "salmeterol", "indacaterol",
    "tiotropium", "ipratropium", "umeclidinium", "aclidinium", "glycopyrronium",
    "fluticasone", "budesonide", "beclomethasone", "ciclesonide", "mometasone",
    "montelukast", "zafirlukast", "zileuton",
    "theophylline", "aminophylline", "roflumilast",
    "omalizumab", "mepolizumab", "reslizumab", "benralizumab", "dupilumab",
    # GI / Metabolic
    "omeprazole", "esomeprazole", "lansoprazole", "pantoprazole", "rabeprazole",
    "ranitidine", "famotidine", "cimetidine", "sucralfate",
    "metoclopramide", "domperidone", "ondansetron", "granisetron",
    "loperamide", "mesalazine", "sulfasalazine", "infliximab",
    "cholestyramine", "colesevelam", "colestipol",
    "orlistat", "sibutramine", "naltrexone", "bupropion",
    # Endocrine
    "levothyroxine", "liothyronine", "methimazole", "propylthiouracil",
    "alendronate", "risedronate", "zoledronate", "denosumab", "teriparatide",
    "calcitonin", "raloxifene",
    # Infectious disease / antimalarials
    "artemisinin", "artesunate", "mefloquine", "quinine", "atovaquone",
    "proguanil", "primaquine", "doxycycline",
    # Repurposing candidates (well-known)
    "berberine", "resveratrol", "curcumin", "quercetin",
    "deferiprone", "deferoxamine", "deferasirox",
    "ambroxol", "bromocriptine", "cabergoline",
    "itraconazole", "fluconazole", "voriconazole",
    "minocycline", "doxycycline", "azithromycin",
    "celecoxib", "naproxen", "diclofenac",
    "propranolol", "atenolol", "metoprolol",
    "digoxin", "colchicine", "dapsone",
    "isradipine", "nimodipine", "nifedipine",
    "exenatide", "liraglutide", "semaglutide",
    "psilocybin", "mdma", "ketamine",
})

_GENE_SYMBOLS: frozenset[str] = frozenset({
    "EGFR", "BRAF", "KRAS", "TP53", "BRCA1", "BRCA2", "HER2", "ALK",
    "ROS1", "MET", "RET", "FGFR", "PIK3CA", "PTEN", "AKT", "MTOR",
    "JAK2", "BCR", "ABL", "VEGF", "VEGFR", "PDGFR", "KIT", "FLT3",
    "IDH1", "IDH2", "NTRK", "MEK", "ERK", "RAF", "RAS", "WNT",
    "NOTCH", "SHH", "TNF", "IL6", "IL1", "IL2", "IL17", "CXCR4",
    "PD1", "PDL1", "CTLA4", "CD20", "CD19", "CD38", "BCL2",
    "ACE", "ACE2", "RAAS", "COX1", "COX2", "HMGCR", "DPP4", "SGLT2",
    "GLP1", "PPAR", "AMPK", "HDAC", "DNMT", "BET", "CDK4", "CDK6",
    "PARP", "ATR", "ATM", "CHK1", "WEE1", "AURK", "PLK1",
    "GABA", "NMDA", "AMPA", "5HT", "DOPA", "MAO", "COMT", "ACHE",
    "BCHE", "SERT", "NET", "DAT", "VMAT", "CHAT",
    "APP", "MAPT", "SNCA", "LRRK2", "PARK", "GBA",
    "ACE2", "TMPRSS2", "SPIKE", "NSP",
    "ESR1", "ESR2", "AR", "PGR", "ERBB2",
    "ABL1", "BCR", "FLT3", "JAK1", "JAK2", "JAK3", "TYK2",
    "STAT3", "STAT1", "NF2", "TSC1", "TSC2", "VHL", "MDM2",
    "CDK2", "CDK4", "CDK6", "CCND1", "RB1", "E2F",
})

_STOPWORDS: frozenset[str] = frozenset({
    "the", "this", "that", "these", "those", "their", "they", "them",
    "have", "has", "had", "been", "were", "with", "from", "also", "both",
    "since", "while", "when", "where", "which", "what", "into", "onto",
    "more", "most", "some", "such", "only", "each", "other", "than",
    "disease", "disorder", "syndrome", "condition", "therapy", "treatment",
    "drug", "drugs", "agent", "agents", "compound", "compounds",
    "inhibitor", "inhibitors", "agonist", "agonists", "antagonist",
    "study", "studies", "trial", "trials", "review", "reviews",
    "patient", "patients", "clinical", "effect", "effects",
    "recent", "advances", "development", "emerging", "therapeutic",
    "potential", "approach", "methods", "results", "analysis",
    "cancer", "tumor", "tumour", "diabetes", "mellitus", "type",
    "model", "models", "protein", "proteins", "gene", "genes",
    "cell", "cells", "level", "levels", "human", "mouse",
    "conclusion", "background", "objective", "purpose", "aim",
})


def _load_hf_ner():
    """Load HuggingFace biomedical NER pipeline (lazy, cached)."""
    global _hf_ner
    if _hf_ner is not None:
        return _hf_ner
    try:
        from transformers import pipeline
        _hf_ner = pipeline(
            "ner",
            model="d4data/biomedical-ner-all",
            aggregation_strategy="simple",
            device=-1,  # CPU
        )
        logger.info("Loaded HuggingFace biomedical NER: d4data/biomedical-ner-all")
        return _hf_ner
    except Exception as e:
        logger.warning("HuggingFace biomedical NER unavailable: %s", e)
        return None


def _load_spacy_nlp():
    """Load spaCy model as secondary extractor (lazy, cached)."""
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy
        for model_name in ["en_core_sci_sm", "en_core_web_sm"]:
            try:
                _nlp = spacy.load(model_name)
                logger.info("Loaded spaCy model: %s", model_name)
                return _nlp
            except OSError:
                continue
        from spacy.cli import download
        download("en_core_web_sm")
        _nlp = spacy.load("en_core_web_sm")
        return _nlp
    except Exception as e:
        logger.warning("spaCy unavailable: %s", e)
        return None


def _is_drug_like(token: str) -> bool:
    lower = token.lower().strip()
    if len(lower) < 4 or lower in _STOPWORDS:
        return False
    if lower in _KNOWN_DRUGS:
        return True
    if any(lower.endswith(sfx) for sfx in _DRUG_SUFFIXES):
        return True
    if re.match(r"^[A-Za-z]{1,4}[-_]?\d{4,}", token):
        return True
    return False


def _is_target_like(token: str) -> bool:
    upper = token.upper().strip()
    if upper in _GENE_SYMBOLS:
        return True
    lower = token.lower()
    return any(kw in lower for kw in {"receptor", "kinase", "enzyme", "transporter", "channel", "protease", "synthase"})


def _normalize_drug_name(name: str) -> str:
    normalized = name.strip()
    normalized = re.sub(
        r"\s+(hydrochloride|mesylate|besylate|tartrate|maleate|fumarate|succinate|sodium|potassium|calcium)$",
        "", normalized, flags=re.IGNORECASE,
    )
    return normalized[0].upper() + normalized[1:] if normalized else normalized


def _extract_with_hf(text: str, pmid: str) -> ExtractionResult:
    """Extract entities using HuggingFace biomedical NER model."""
    result = ExtractionResult()
    ner = _load_hf_ner()
    if ner is None:
        return result

    try:
        # Model max length is 512 tokens — truncate to ~1500 chars
        entities = ner(text[:1500])
        for ent in entities:
            label = ent.get("entity_group", "").upper()
            word = ent.get("word", "").strip()
            if len(word) < 3:
                continue

            if label in {"CHEMICAL", "DRUG", "MEDICATION"}:
                if _is_drug_like(word) or label == "CHEMICAL":
                    result.drugs.append(BioEntity(
                        text=word, label="DRUG",
                        start=ent.get("start", 0), end=ent.get("end", 0),
                        source_pmid=pmid,
                        normalized=_normalize_drug_name(word),
                    ))
            elif label in {"GENE", "PROTEIN", "DNA", "RNA"}:
                result.targets.append(BioEntity(
                    text=word, label="TARGET",
                    start=ent.get("start", 0), end=ent.get("end", 0),
                    source_pmid=pmid,
                    normalized=word.upper(),
                ))
            elif label in {"DISEASE", "DISORDER", "SYMPTOM"}:
                result.diseases.append(BioEntity(
                    text=word, label="DISEASE",
                    source_pmid=pmid,
                    normalized=word.lower(),
                ))
    except Exception as e:
        logger.debug("HF NER failed on doc %s: %s", pmid, e)

    return result


def _extract_rule_based(text: str, pmid: str) -> ExtractionResult:
    """Fast rule-based extraction using drug dictionary + regex."""
    result = ExtractionResult()

    # Drug names: capitalized words matching known drugs or suffixes
    drug_pattern = re.compile(r"\b([A-Z][a-z]{3,}(?:[a-z]+)?)\b")
    for match in drug_pattern.finditer(text):
        token = match.group(1)
        if _is_drug_like(token):
            result.drugs.append(BioEntity(
                text=token, label="DRUG",
                start=match.start(), end=match.end(),
                source_pmid=pmid,
                normalized=_normalize_drug_name(token),
            ))

    # Gene/target symbols
    for match in re.finditer(r"\b([A-Z][A-Z0-9]{1,6}(?:-[A-Z0-9]+)?)\b", text):
        token = match.group(1)
        if token.upper() in _GENE_SYMBOLS:
            result.targets.append(BioEntity(
                text=token, label="TARGET",
                start=match.start(), end=match.end(),
                source_pmid=pmid,
                normalized=token.upper(),
            ))

    return result


def extract_entities_from_doc(doc: EvidenceDoc) -> ExtractionResult:
    """Extract biomedical entities from a single PubMed document."""
    text = f"{doc.title}. {doc.abstract}"

    # Primary: HuggingFace biomedical NER
    hf_result = _extract_with_hf(text, doc.pmid)

    # Secondary: rule-based (adds drugs missed by HF model)
    rule_result = _extract_rule_based(text, doc.pmid)

    # Merge — HF takes priority, rule-based fills gaps
    combined = ExtractionResult()
    seen_drugs = {d.normalized.lower() for d in hf_result.drugs}
    combined.drugs = list(hf_result.drugs)
    for d in rule_result.drugs:
        if d.normalized.lower() not in seen_drugs:
            combined.drugs.append(d)
            seen_drugs.add(d.normalized.lower())

    seen_targets = {t.normalized for t in hf_result.targets}
    combined.targets = list(hf_result.targets)
    for t in rule_result.targets:
        if t.normalized not in seen_targets:
            combined.targets.append(t)
            seen_targets.add(t.normalized)

    combined.diseases = hf_result.diseases

    # Build drug-target pairs from co-occurrence in sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sentence in sentences:
        sent_drugs = [d for d in combined.drugs if d.text.lower() in sentence.lower()]
        sent_targets = [t for t in combined.targets if t.text in sentence]
        for drug in sent_drugs:
            for target in sent_targets:
                pair = (drug.normalized, target.normalized)
                if pair not in combined.drug_target_pairs:
                    combined.drug_target_pairs.append(pair)

    return combined


def extract_entities_from_docs(docs: list[EvidenceDoc]) -> ExtractionResult:
    """Extract and aggregate entities from multiple documents."""
    combined = ExtractionResult()
    drug_counts: Counter = Counter()
    target_counts: Counter = Counter()

    for doc in docs:
        doc_result = extract_entities_from_doc(doc)

        for drug in doc_result.drugs:
            drug_counts[drug.normalized] += 1
            already = any(d.normalized == drug.normalized for d in combined.drugs)
            if not already:
                combined.drugs.append(drug)
            else:
                for d in combined.drugs:
                    if d.normalized == drug.normalized and doc.pmid not in d.source_pmid:
                        d.source_pmid += f",{doc.pmid}"

        for target in doc_result.targets:
            target_counts[target.normalized] += 1
            if not any(t.normalized == target.normalized for t in combined.targets):
                combined.targets.append(target)

        combined.drug_target_pairs.extend(doc_result.drug_target_pairs)
        combined.drug_disease_pairs.extend(doc_result.drug_disease_pairs)

    combined.drug_target_pairs = list(set(combined.drug_target_pairs))
    combined.drug_disease_pairs = list(set(combined.drug_disease_pairs))
    combined.drugs.sort(key=lambda d: drug_counts.get(d.normalized, 0), reverse=True)
    combined.targets.sort(key=lambda t: target_counts.get(t.normalized, 0), reverse=True)

    return combined


def get_drug_mentions_per_pmid(docs: list[EvidenceDoc]) -> dict[str, list[str]]:
    """Return {drug_name: [pmid1, pmid2, ...]} mapping."""
    drug_pmids: dict[str, set[str]] = defaultdict(set)
    for doc in docs:
        result = extract_entities_from_doc(doc)
        for drug in result.drugs:
            drug_pmids[drug.normalized].add(doc.pmid)
    return {drug: sorted(pmids) for drug, pmids in drug_pmids.items()}
