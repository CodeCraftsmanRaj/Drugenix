"""
Drugenix - KG Embeddings via PyKEEN TransE
============================================
Trains a TransE model on the drug-disease-target knowledge graph to enable
link prediction: scoring novel (unseen) drug-disease repurposing hypotheses.

TransE: Bordes et al. (2013) - h + r ≈ t (lower L2 distance = stronger association)
Training uses PyKEEN's sLCWA (stochastic local closed-world assumption) pipeline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "data" / "transe_model"


class TransEEmbeddings:
    """Stores trained TransE entity/relation embeddings and answers link-prediction queries."""

    def __init__(self):
        self.entity_to_id: dict[str, int] = {}
        self.id_to_entity: dict[int, str] = {}
        self.relation_to_id: dict[str, int] = {}
        self.entity_embeddings: Optional[np.ndarray] = None
        self.relation_embeddings: Optional[np.ndarray] = None
        self.trained = False
        self._disease_nodes: list[str] = []
        self._drug_nodes: list[str] = []

    def load(self, model_dir: Path) -> bool:
        meta_path = model_dir / "meta.json"
        emb_path = model_dir / "embeddings.npz"
        if not meta_path.exists() or not emb_path.exists():
            return False
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.entity_to_id = meta["entity_to_id"]
            self.id_to_entity = {int(v): k for k, v in self.entity_to_id.items()}
            self.relation_to_id = meta["relation_to_id"]
            self._disease_nodes = meta.get("disease_nodes", [])
            self._drug_nodes = meta.get("drug_nodes", [])
            npz = np.load(str(emb_path))
            self.entity_embeddings = npz["entity"]
            self.relation_embeddings = npz["relation"]
            self.trained = True
            logger.info("Loaded TransE model (%d entities)", len(self.entity_to_id))
            return True
        except Exception as e:
            logger.warning("Could not load TransE model: %s", e)
            return False

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "entity_to_id": self.entity_to_id,
            "relation_to_id": self.relation_to_id,
            "disease_nodes": self._disease_nodes,
            "drug_nodes": self._drug_nodes,
        }
        (model_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        np.savez(str(model_dir / "embeddings.npz"),
                 entity=self.entity_embeddings,
                 relation=self.relation_embeddings)
        logger.info("Saved TransE model to %s", model_dir)

    def score(self, head: str, relation: str, tail: str) -> Optional[float]:
        """TransE L2 distance (lower = better fit)."""
        if not self.trained:
            return None
        h_id = self.entity_to_id.get(head)
        t_id = self.entity_to_id.get(tail)
        r_id = self.relation_to_id.get(relation)
        if h_id is None or t_id is None or r_id is None:
            return None
        h_emb = self.entity_embeddings[h_id]
        t_emb = self.entity_embeddings[t_id]
        r_emb = self.relation_embeddings[r_id]
        return float(np.linalg.norm(h_emb + r_emb - t_emb))

    def predict_novel_drugs(self, disease: str, top_k: int = 10) -> list[dict]:
        """Rank all drug nodes by TransE score for a given disease (lower dist = better)."""
        if not self.trained:
            return []
        disease_lower = disease.lower().strip()
        disease_node = None
        if disease_lower in self.entity_to_id:
            disease_node = disease_lower
        else:
            for node in self._disease_nodes:
                if disease_lower in node or node in disease_lower:
                    disease_node = node
                    break
        if disease_node is None:
            return []

        treats_id = self.relation_to_id.get("treats")
        if treats_id is None:
            return []

        disease_id = self.entity_to_id[disease_node]
        t_emb = self.entity_embeddings[disease_id]
        r_emb = self.relation_embeddings[treats_id]
        target = t_emb - r_emb  # h ≈ t - r

        scored = []
        for drug_node in self._drug_nodes:
            d_id = self.entity_to_id.get(drug_node)
            if d_id is None:
                continue
            dist = float(np.linalg.norm(self.entity_embeddings[d_id] - target))
            scored.append((drug_node, dist))

        scored.sort(key=lambda x: x[1])

        min_d = scored[0][1] if scored else 0.0
        max_d = scored[-1][1] if scored else 1.0
        rng = max(max_d - min_d, 1e-6)

        return [
            {
                "drug": drug_node.title(),
                "transe_distance": round(dist, 4),
                "confidence": round(1.0 - (dist - min_d) / rng, 3),
                "disease": disease,
            }
            for drug_node, dist in scored[:top_k]
        ]


def train_transe_on_kg(
    num_epochs: int = 100,
    embedding_dim: int = 64,
) -> TransEEmbeddings:
    """Train TransE using PyKEEN's pipeline on our drug-disease-target KG."""
    from pykeen.pipeline import pipeline as pykeen_pipeline
    from pykeen.triples import TriplesFactory
    from drugenix.knowledge_graph import get_knowledge_graph

    kg = get_knowledge_graph()
    graph = kg.graph

    print(f"Exporting triples ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)...")
    triples_list = [
        [u, data.get("relation", "related_to"), v]
        for u, v, data in graph.edges(data=True)
    ]
    print(f"  {len(triples_list)} triples")

    triples_np = np.array(triples_list, dtype=str)
    tf = TriplesFactory.from_labeled_triples(triples_np)

    # 80/20 train/test split
    train_tf, test_tf = tf.split([0.8, 0.2], random_state=42)

    print(f"Training TransE via PyKEEN (dim={embedding_dim}, epochs={num_epochs})...")
    result = pykeen_pipeline(
        training=train_tf,
        testing=test_tf,
        model="TransE",
        epochs=num_epochs,
        model_kwargs={"embedding_dim": embedding_dim},
        optimizer="Adam",
        optimizer_kwargs={"lr": 0.01},
        loss="MarginRankingLoss",
        loss_kwargs={"margin": 1.0},
        training_loop="sLCWA",
        negative_sampler="basic",
        evaluator_kwargs={"filtered": True},
        random_seed=42,
        use_tqdm=True,
    )

    # Extract numpy embeddings
    import torch
    with torch.no_grad():
        entity_emb = result.model.entity_representations[0]().cpu().numpy()
        relation_emb = result.model.relation_representations[0]().cpu().numpy()

    entity_to_id = {e: int(i) for e, i in tf.entity_to_id.items()}
    relation_to_id = {r: int(i) for r, i in tf.relation_to_id.items()}

    model = TransEEmbeddings()
    model.entity_to_id = entity_to_id
    model.id_to_entity = {v: k for k, v in entity_to_id.items()}
    model.relation_to_id = relation_to_id
    model.entity_embeddings = entity_emb
    model.relation_embeddings = relation_emb
    model.trained = True
    model._disease_nodes = [e for e in entity_to_id if graph.nodes.get(e, {}).get("type") == "disease"]
    model._drug_nodes = [e for e in entity_to_id if graph.nodes.get(e, {}).get("type") == "drug"]

    # Print PyKEEN evaluation metrics
    try:
        metrics = result.get_metric("hits@10")
        print(f"  PyKEEN evaluation — Hits@10: {metrics:.4f}")
    except Exception:
        pass

    print(f"  {len(model._drug_nodes)} drug nodes, {len(model._disease_nodes)} disease nodes in model.")
    return model


_embeddings_instance: Optional[TransEEmbeddings] = None


def get_embeddings_model(retrain: bool = False) -> TransEEmbeddings:
    global _embeddings_instance
    if _embeddings_instance is not None and not retrain:
        return _embeddings_instance
    model = TransEEmbeddings()
    if not retrain and model.load(MODEL_DIR):
        _embeddings_instance = model
        return model
    if retrain:
        model = train_transe_on_kg()
        model.save(MODEL_DIR)
        _embeddings_instance = model
    return model


def predict_novel_drugs(disease: str, top_k: int = 10) -> list[dict]:
    model = get_embeddings_model()
    if not model.trained:
        return []
    return model.predict_novel_drugs(disease, top_k=top_k)
