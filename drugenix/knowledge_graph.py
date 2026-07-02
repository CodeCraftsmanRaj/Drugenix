"""
Drugenix - Knowledge Graph Validation Layer
=============================================
Builds a drug-target-disease knowledge graph using:
  - CTD (Comparative Toxicogenomics Database) open data
  - Local curated relationships
  - NetworkX for graph operations

Validates LLM-generated hypotheses by checking biological plausibility
via known drug -> target -> disease paths in the graph.
"""

from __future__ import annotations

import csv
import gzip
import io
import logging
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx

from drugenix.models import DrugCandidate, EvidenceDoc

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent / "data" / "kg"

# CTD download URLs (tab-separated, gzipped)
_CTD_CHEM_DISEASE_URL = "https://ctdbase.org/reports/CTD_chemicals_diseases.tsv.gz"
_CTD_CHEM_GENE_URL = "https://ctdbase.org/reports/CTD_chem_gene_ixns.tsv.gz"

# Curated drug-target-disease relationships for common repurposing cases
_CURATED_RELATIONSHIPS: list[dict] = [
    # Alzheimer's Disease
    {"drug": "donepezil", "target": "ACHE", "disease": "Alzheimer disease", "mechanism": "acetylcholinesterase inhibitor"},
    {"drug": "memantine", "target": "NMDA", "disease": "Alzheimer disease", "mechanism": "NMDA receptor antagonist"},
    {"drug": "rivastigmine", "target": "ACHE", "disease": "Alzheimer disease", "mechanism": "cholinesterase inhibitor"},
    {"drug": "galantamine", "target": "ACHE", "disease": "Alzheimer disease", "mechanism": "cholinesterase inhibitor"},
    {"drug": "rapamycin", "target": "MTOR", "disease": "Alzheimer disease", "mechanism": "mTOR inhibitor, autophagy inducer"},
    # Parkinson's Disease
    {"drug": "levodopa", "target": "DOPA", "disease": "Parkinson disease", "mechanism": "dopamine precursor"},
    {"drug": "carbidopa", "target": "DOPA", "disease": "Parkinson disease", "mechanism": "DOPA decarboxylase inhibitor"},
    {"drug": "rasagiline", "target": "MAO", "disease": "Parkinson disease", "mechanism": "MAO-B inhibitor"},
    {"drug": "pramipexole", "target": "DOPA", "disease": "Parkinson disease", "mechanism": "dopamine agonist"},
    # Type 2 Diabetes
    {"drug": "metformin", "target": "AMPK", "disease": "Type 2 diabetes", "mechanism": "AMPK activator"},
    {"drug": "empagliflozin", "target": "SGLT2", "disease": "Type 2 diabetes", "mechanism": "SGLT2 inhibitor"},
    {"drug": "sitagliptin", "target": "DPP4", "disease": "Type 2 diabetes", "mechanism": "DPP-4 inhibitor"},
    {"drug": "semaglutide", "target": "GLP1", "disease": "Type 2 diabetes", "mechanism": "GLP-1 receptor agonist"},
    {"drug": "pioglitazone", "target": "PPAR", "disease": "Type 2 diabetes", "mechanism": "PPAR-gamma agonist"},
    # Cancer (general repurposing)
    {"drug": "imatinib", "target": "BCR", "disease": "Chronic myeloid leukemia", "mechanism": "BCR-ABL kinase inhibitor"},
    {"drug": "tamoxifen", "target": "ESR1", "disease": "Breast cancer", "mechanism": "estrogen receptor modulator"},
    {"drug": "pembrolizumab", "target": "PD1", "disease": "Cancer", "mechanism": "PD-1 immune checkpoint inhibitor"},
    {"drug": "bevacizumab", "target": "VEGF", "disease": "Cancer", "mechanism": "VEGF inhibitor, anti-angiogenic"},
    {"drug": "sorafenib", "target": "RAF", "disease": "Hepatocellular carcinoma", "mechanism": "multi-kinase inhibitor"},
    {"drug": "thalidomide", "target": "TNF", "disease": "Multiple myeloma", "mechanism": "TNF-alpha inhibitor, immunomodulator"},
    {"drug": "methotrexate", "target": "DHFR", "disease": "Rheumatoid arthritis", "mechanism": "dihydrofolate reductase inhibitor"},
    # Cardiovascular
    {"drug": "atorvastatin", "target": "HMGCR", "disease": "Hypercholesterolemia", "mechanism": "HMG-CoA reductase inhibitor"},
    {"drug": "losartan", "target": "RAAS", "disease": "Hypertension", "mechanism": "angiotensin II receptor blocker"},
    {"drug": "lisinopril", "target": "ACE", "disease": "Hypertension", "mechanism": "ACE inhibitor"},
    {"drug": "amlodipine", "target": "CACNA1C", "disease": "Hypertension", "mechanism": "calcium channel blocker"},
    # Asthma / Respiratory
    {"drug": "montelukast", "target": "CYSLTR1", "disease": "Asthma", "mechanism": "leukotriene receptor antagonist"},
    {"drug": "salbutamol", "target": "ADRB2", "disease": "Asthma", "mechanism": "beta-2 adrenergic agonist"},
    {"drug": "fluticasone", "target": "NR3C1", "disease": "Asthma", "mechanism": "glucocorticoid receptor agonist"},
    {"drug": "budesonide", "target": "NR3C1", "disease": "Asthma", "mechanism": "inhaled corticosteroid"},
    {"drug": "tiotropium", "target": "CHRM3", "disease": "COPD", "mechanism": "muscarinic receptor antagonist"},
    # COVID-19 / Coronavirus
    {"drug": "remdesivir", "target": "RdRp", "disease": "COVID-19", "mechanism": "RNA polymerase inhibitor"},
    {"drug": "dexamethasone", "target": "NR3C1", "disease": "COVID-19", "mechanism": "corticosteroid, anti-inflammatory"},
    {"drug": "baricitinib", "target": "JAK1", "disease": "COVID-19", "mechanism": "JAK1/JAK2 inhibitor, anti-inflammatory"},
    {"drug": "tocilizumab", "target": "IL6R", "disease": "COVID-19", "mechanism": "IL-6 receptor antagonist"},
    {"drug": "nirmatrelvir", "target": "3CLpro", "disease": "COVID-19", "mechanism": "3CL protease inhibitor"},
    {"drug": "molnupiravir", "target": "RdRp", "disease": "COVID-19", "mechanism": "RNA polymerase inhibitor, mutagenic"},
    {"drug": "hydroxychloroquine", "target": "TLR9", "disease": "COVID-19", "mechanism": "immunomodulator, endosomal pH modifier"},
    {"drug": "ivermectin", "target": "KCNQ2", "disease": "COVID-19", "mechanism": "anti-parasitic, ion channel modulator"},
    # Influenza / Flu
    {"drug": "oseltamivir", "target": "NA", "disease": "Influenza", "mechanism": "neuraminidase inhibitor"},
    {"drug": "zanamivir", "target": "NA", "disease": "Influenza", "mechanism": "neuraminidase inhibitor"},
    {"drug": "baloxavir", "target": "PA", "disease": "Influenza", "mechanism": "cap-dependent endonuclease inhibitor"},
    {"drug": "favipiravir", "target": "RdRp", "disease": "Influenza", "mechanism": "RNA polymerase inhibitor"},
    # Breast Cancer
    {"drug": "tamoxifen", "target": "ESR1", "disease": "Breast cancer", "mechanism": "estrogen receptor modulator"},
    {"drug": "trastuzumab", "target": "ERBB2", "disease": "Breast cancer", "mechanism": "HER2 monoclonal antibody"},
    {"drug": "letrozole", "target": "CYP19A1", "disease": "Breast cancer", "mechanism": "aromatase inhibitor"},
    {"drug": "palbociclib", "target": "CDK4", "disease": "Breast cancer", "mechanism": "CDK4/6 inhibitor"},
    {"drug": "olaparib", "target": "PARP1", "disease": "Breast cancer", "mechanism": "PARP inhibitor (BRCA1/2 mutant)"},
    # Lung Cancer
    {"drug": "erlotinib", "target": "EGFR", "disease": "Lung cancer", "mechanism": "EGFR tyrosine kinase inhibitor"},
    {"drug": "gefitinib", "target": "EGFR", "disease": "Lung cancer", "mechanism": "EGFR tyrosine kinase inhibitor"},
    {"drug": "crizotinib", "target": "ALK", "disease": "Lung cancer", "mechanism": "ALK/MET/ROS1 inhibitor"},
    {"drug": "pembrolizumab", "target": "PD1", "disease": "Lung cancer", "mechanism": "PD-1 checkpoint inhibitor"},
    {"drug": "bevacizumab", "target": "VEGF", "disease": "Lung cancer", "mechanism": "anti-VEGF antibody"},
    # Colorectal / GI Cancer
    {"drug": "cetuximab", "target": "EGFR", "disease": "Colorectal cancer", "mechanism": "EGFR monoclonal antibody"},
    {"drug": "oxaliplatin", "target": "DNA", "disease": "Colorectal cancer", "mechanism": "DNA cross-linking agent"},
    # HIV / AIDS
    {"drug": "tenofovir", "target": "RT", "disease": "HIV infection", "mechanism": "nucleotide reverse transcriptase inhibitor"},
    {"drug": "dolutegravir", "target": "INSTI", "disease": "HIV infection", "mechanism": "integrase strand transfer inhibitor"},
    {"drug": "lopinavir", "target": "PR", "disease": "HIV infection", "mechanism": "HIV protease inhibitor"},
    # Depression / Mental Health
    {"drug": "fluoxetine", "target": "SLC6A4", "disease": "Depression", "mechanism": "SSRI, serotonin reuptake inhibitor"},
    {"drug": "sertraline", "target": "SLC6A4", "disease": "Depression", "mechanism": "SSRI, serotonin reuptake inhibitor"},
    {"drug": "venlafaxine", "target": "SLC6A4", "disease": "Depression", "mechanism": "SNRI, serotonin-norepinephrine reuptake inhibitor"},
    {"drug": "ketamine", "target": "GRIN2B", "disease": "Treatment-resistant depression", "mechanism": "NMDA receptor antagonist"},
    # Rheumatoid Arthritis / Autoimmune
    {"drug": "methotrexate", "target": "DHFR", "disease": "Rheumatoid arthritis", "mechanism": "dihydrofolate reductase inhibitor"},
    {"drug": "adalimumab", "target": "TNF", "disease": "Rheumatoid arthritis", "mechanism": "TNF-alpha monoclonal antibody"},
    {"drug": "tofacitinib", "target": "JAK1", "disease": "Rheumatoid arthritis", "mechanism": "JAK inhibitor"},
    {"drug": "hydroxychloroquine", "target": "TLR9", "disease": "Lupus", "mechanism": "immunomodulator, TLR9 inhibitor"},
    # Repurposing highlights
    {"drug": "sildenafil", "target": "PDE5A", "disease": "Pulmonary hypertension", "mechanism": "PDE5 inhibitor (repurposed from erectile dysfunction)"},
    {"drug": "colchicine", "target": "TUBB", "disease": "Pericarditis", "mechanism": "tubulin inhibitor, anti-inflammatory"},
    {"drug": "minoxidil", "target": "KATP", "disease": "Alopecia", "mechanism": "potassium channel opener (repurposed from hypertension)"},
    {"drug": "metformin", "target": "AMPK", "disease": "Cancer", "mechanism": "AMPK activator, anti-proliferative"},
    {"drug": "aspirin", "target": "PTGS1", "disease": "Colorectal cancer", "mechanism": "COX-1 inhibitor, anti-platelet (repurposing candidate)"},
    {"drug": "rapamycin", "target": "MTOR", "disease": "Cancer", "mechanism": "mTOR inhibitor, anti-proliferative"},
    {"drug": "lithium", "target": "GSK3B", "disease": "Alzheimer disease", "mechanism": "GSK3-beta inhibitor, neuroprotective"},
    {"drug": "metformin", "target": "AMPK", "disease": "Alzheimer disease", "mechanism": "AMPK activator, reduces amyloid-beta production"},
]


@dataclass
class KGValidationResult:
    drug: str
    disease: str
    has_direct_path: bool = False
    has_indirect_path: bool = False
    path_length: int = -1
    path_nodes: list[str] = field(default_factory=list)
    known_targets: list[str] = field(default_factory=list)
    known_mechanism: str = ""
    kg_confidence_boost: float = 0.0
    explanation: str = ""


class DrugKnowledgeGraph:
    """Drug-Target-Disease knowledge graph backed by NetworkX."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self._loaded = False

    def build(self, use_ctd: bool = False) -> None:
        """Build the knowledge graph from curated data + ChEMBL (if available)."""
        if self._loaded:
            return

        self._add_curated_relationships()

        # Auto-load ChEMBL data if the fetch script has been run
        chembl_path = Path(__file__).parent.parent / "data" / "chembl_kg.json"
        if chembl_path.exists():
            self._load_chembl_data(chembl_path)

        if use_ctd:
            self._load_ctd_data()

        self._loaded = True
        logger.info(
            "Knowledge graph built: %d nodes, %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

    def _load_chembl_data(self, path: Path) -> None:
        """Load drug-disease associations from ChEMBL JSON (built by scripts/fetch_chembl.py)."""
        import json
        try:
            data: dict[str, list[dict]] = json.loads(path.read_text(encoding="utf-8"))
            added = 0
            for disease, drugs in data.items():
                disease_node = disease.lower().strip()
                self.graph.add_node(disease_node, type="disease")

                for entry in drugs:
                    drug_name = entry.get("drug", "").strip()
                    if not drug_name or len(drug_name) < 3:
                        continue
                    drug_node = drug_name.lower()
                    mechanism = entry.get("mechanism", "")
                    target_name = entry.get("target", "")
                    max_phase = entry.get("max_phase", 0)

                    self.graph.add_node(drug_node, type="drug", chembl_phase=max_phase)

                    # Drug -> Disease edge (direct therapeutic link)
                    self.graph.add_edge(
                        drug_node, disease_node,
                        relation="treats",
                        mechanism=mechanism,
                        evidence="ChEMBL",
                        max_phase=max_phase,
                    )

                    # Drug -> Target -> Disease (if target available)
                    if target_name:
                        target_node = target_name.upper()[:20]
                        self.graph.add_node(target_node, type="target")
                        self.graph.add_edge(drug_node, target_node, relation="targets", mechanism=mechanism)
                        self.graph.add_edge(target_node, disease_node, relation="associated_with")

                    added += 1

            logger.info("Loaded %d ChEMBL drug-disease pairs from %s", added, path.name)
        except Exception as e:
            logger.warning("Could not load ChEMBL data: %s", e)

    def _add_curated_relationships(self) -> None:
        for rel in _CURATED_RELATIONSHIPS:
            drug = rel["drug"].lower()
            target = rel["target"].upper()
            disease = rel["disease"].lower()
            mechanism = rel["mechanism"]

            self.graph.add_node(drug, type="drug")
            self.graph.add_node(target, type="target")
            self.graph.add_node(disease, type="disease")

            self.graph.add_edge(drug, target, relation="targets", mechanism=mechanism)
            self.graph.add_edge(target, disease, relation="associated_with")
            self.graph.add_edge(drug, disease, relation="treats", mechanism=mechanism)

    def _load_ctd_data(self) -> None:
        """Download and parse CTD chemical-disease associations."""
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = _DATA_DIR / "ctd_chem_disease_cache.tsv"

        if cache_file.exists():
            self._parse_ctd_chem_disease(cache_file)
            return

        try:
            logger.info("Downloading CTD chemical-disease data...")
            req = urllib.request.Request(
                _CTD_CHEM_DISEASE_URL,
                headers={"User-Agent": "Drugenix/0.2"},
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                compressed = response.read()

            decompressed = gzip.decompress(compressed).decode("utf-8", errors="replace")

            lines = []
            count = 0
            for line in decompressed.splitlines():
                if line.startswith("#"):
                    continue
                lines.append(line)
                count += 1
                if count >= 50000:
                    break

            cache_file.write_text("\n".join(lines), encoding="utf-8")
            self._parse_ctd_chem_disease(cache_file)
            logger.info("Loaded %d CTD chemical-disease associations", count)

        except Exception as e:
            logger.warning("Could not download CTD data: %s. Using curated data only.", e)

    def _parse_ctd_chem_disease(self, filepath: Path) -> None:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 5:
                    continue
                chem_name = parts[0].lower().strip()
                disease_name = parts[3].lower().strip()
                evidence = parts[5] if len(parts) > 5 else ""

                if not chem_name or not disease_name:
                    continue

                self.graph.add_node(chem_name, type="drug")
                self.graph.add_node(disease_name, type="disease")

                relation = "therapeutic" if "therapeutic" in evidence.lower() else "associated_with"
                self.graph.add_edge(chem_name, disease_name, relation=relation)

    def validate_candidate(self, drug_name: str, disease_query: str) -> KGValidationResult:
        """Check if a drug-disease relationship has biological plausibility in the KG."""
        if not self._loaded:
            self.build()

        drug_lower = drug_name.lower().strip()
        disease_lower = disease_query.lower().strip()

        result = KGValidationResult(drug=drug_name, disease=disease_query)

        # Check direct drug -> disease edge
        if self.graph.has_edge(drug_lower, disease_lower):
            result.has_direct_path = True
            result.path_length = 1
            result.path_nodes = [drug_lower, disease_lower]
            edge_data = self.graph.edges[drug_lower, disease_lower]
            result.known_mechanism = edge_data.get("mechanism", "")
            result.kg_confidence_boost = 0.15
            result.explanation = f"Direct KG link: {drug_name} -> {disease_query}"

            # Get targets
            for neighbor in self.graph.successors(drug_lower):
                node_data = self.graph.nodes.get(neighbor, {})
                if node_data.get("type") == "target":
                    result.known_targets.append(neighbor)

            return result

        # Check indirect path: drug -> target -> disease
        undirected = self.graph.to_undirected()
        drug_node = drug_lower if drug_lower in undirected else None
        disease_node = disease_lower if disease_lower in undirected else None

        # Try partial matching on disease name (bidirectional substring match)
        if disease_node is None:
            for node in undirected.nodes():
                node_data = self.graph.nodes.get(node, {})
                if node_data.get("type") == "disease" and (
                    disease_lower in node or node in disease_lower
                ):
                    disease_node = node
                    break

        if drug_node and disease_node:
            try:
                path = nx.shortest_path(undirected, drug_node, disease_node)
                result.has_indirect_path = True
                result.path_length = len(path) - 1
                result.path_nodes = path

                for node in path[1:-1]:
                    node_data = self.graph.nodes.get(node, {})
                    if node_data.get("type") == "target":
                        result.known_targets.append(node)

                if result.path_length <= 2:
                    result.kg_confidence_boost = 0.10
                elif result.path_length <= 3:
                    result.kg_confidence_boost = 0.05

                result.explanation = f"Indirect KG path ({result.path_length} hops): {' -> '.join(path)}"
            except nx.NetworkXNoPath:
                result.explanation = f"No KG path found between {drug_name} and {disease_query}"
        else:
            missing = []
            if not drug_node:
                missing.append(f"drug '{drug_name}'")
            if not disease_node:
                missing.append(f"disease '{disease_query}'")
            result.explanation = f"Not in KG: {', '.join(missing)}"

        return result

    def validate_candidates(
        self, candidates: list[DrugCandidate], disease_query: str
    ) -> list[KGValidationResult]:
        """Validate a batch of drug candidates."""
        if not self._loaded:
            self.build()
        return [self.validate_candidate(c.drug, disease_query) for c in candidates]

    def get_known_drugs_for_disease(self, disease_query: str) -> list[str]:
        """Return drugs known to be associated with a disease."""
        if not self._loaded:
            self.build()

        disease_lower = disease_query.lower().strip()
        drugs = []

        for node in self.graph.predecessors(disease_lower):
            node_data = self.graph.nodes.get(node, {})
            if node_data.get("type") == "drug":
                drugs.append(node)

        if not drugs:
            for node in self.graph.nodes():
                node_data = self.graph.nodes.get(node, {})
                if node_data.get("type") == "disease" and disease_lower in node:
                    for pred in self.graph.predecessors(node):
                        pred_data = self.graph.nodes.get(pred, {})
                        if pred_data.get("type") == "drug":
                            drugs.append(pred)

        return drugs

    def get_graph_stats(self) -> dict:
        if not self._loaded:
            self.build()
        drug_count = sum(1 for _, d in self.graph.nodes(data=True) if d.get("type") == "drug")
        target_count = sum(1 for _, d in self.graph.nodes(data=True) if d.get("type") == "target")
        disease_count = sum(1 for _, d in self.graph.nodes(data=True) if d.get("type") == "disease")
        chembl_edges = sum(
            1 for _, _, d in self.graph.edges(data=True)
            if d.get("evidence") == "ChEMBL"
        )
        chembl_path = Path(__file__).parent.parent / "data" / "chembl_kg.json"
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "drugs": drug_count,
            "targets": target_count,
            "diseases": disease_count,
            "chembl_edges": chembl_edges,
            "chembl_loaded": chembl_path.exists(),
            "sources": ["Curated (manual)", "ChEMBL"] if chembl_path.exists() else ["Curated (manual)"],
        }

    def get_subgraph_for_disease(self, disease_query: str, max_hops: int = 2) -> dict:
        """Return a subgraph around a disease for visualization."""
        if not self._loaded:
            self.build()

        disease_lower = disease_query.lower().strip()
        disease_node = None

        if disease_lower in self.graph:
            disease_node = disease_lower
        else:
            for node in self.graph.nodes():
                node_data = self.graph.nodes.get(node, {})
                if node_data.get("type") == "disease" and disease_lower in node:
                    disease_node = node
                    break

        if disease_node is None:
            return {"nodes": [], "edges": []}

        nodes_in_subgraph = {disease_node}
        undirected = self.graph.to_undirected()

        current_frontier = {disease_node}
        for _ in range(max_hops):
            next_frontier = set()
            for node in current_frontier:
                for neighbor in undirected.neighbors(node):
                    if neighbor not in nodes_in_subgraph:
                        nodes_in_subgraph.add(neighbor)
                        next_frontier.add(neighbor)
            current_frontier = next_frontier

        nodes = []
        for node in nodes_in_subgraph:
            node_data = self.graph.nodes.get(node, {})
            nodes.append({
                "id": node,
                "label": node.title(),
                "type": node_data.get("type", "unknown"),
            })

        edges = []
        for u, v, data in self.graph.edges(data=True):
            if u in nodes_in_subgraph and v in nodes_in_subgraph:
                edges.append({
                    "source": u,
                    "target": v,
                    "relation": data.get("relation", ""),
                    "mechanism": data.get("mechanism", ""),
                })

        return {"nodes": nodes, "edges": edges}


# Module-level singleton
_kg_instance: Optional[DrugKnowledgeGraph] = None


def get_knowledge_graph() -> DrugKnowledgeGraph:
    global _kg_instance
    if _kg_instance is None:
        _kg_instance = DrugKnowledgeGraph()
        _kg_instance.build()
    return _kg_instance
