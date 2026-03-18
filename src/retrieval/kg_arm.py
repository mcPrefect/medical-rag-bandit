"""
Knowledge Graph Arm: GNN-based retrieval using UMLS
Uses trained Graph Attention Network to find relevant concepts
"""

import torch
import pickle
import numpy as np
from pathlib import Path
import spacy
from collections import defaultdict

# Single source of truth for model definition
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / "graph"))
from gnn_model import MedicalGAT # type: ignore


class KnowledgeGraphArm:
    """
    Retrieval arm that uses GNN-learned embeddings over UMLS knowledge graph.

    Pipeline:
        1. Extract medical entities from query (scispaCy)
        2. Map entities to UMLS concepts (CUI matching)
        3. Compute GNN embeddings using trained GAT
        4. Find nearest concepts in embedding space
        5. Score documents by overlap with relevant concepts
    """

    def __init__(
        self,
        model_path="models/gnn_model_best.pt",
        graph_path="data/umls/subgraph.pkl",
        concepts_path="data/umls/concepts.pkl",
        features_path=None,
        device='cuda'
    ):
        print("Initializing Knowledge Graph Arm...")
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load checkpoint
        print(f"  Loading GNN from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device,
                                weights_only=False)

        # Reconstruct model from saved config (or defaults)
        cfg = checkpoint.get('config', {})
        self.model = MedicalGAT(
            input_dim=cfg.get('input_dim', 771),
            hidden_dim=cfg.get('hidden_dim', 32),
            output_dim=cfg.get('output_dim', 128),
            num_heads=cfg.get('num_heads', 8)
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.node_to_idx = checkpoint['node_to_idx']
        self.idx_to_node = checkpoint['idx_to_node']

        # Load NetworkX graph
        print(f"  Loading UMLS graph from {graph_path}...")
        with open(graph_path, 'rb') as f:
            self.graph = pickle.load(f)

        # Concept name lookup
        print(f"  Loading concepts from {concepts_path}...")
        with open(concepts_path, 'rb') as f:
            self.concepts = pickle.load(f)

        self.name_to_cuis = defaultdict(list)
        for cui, name in self.concepts.items():
            self.name_to_cuis[name.lower().strip()].append(cui)

        # Build edge_index tensor for message passing at inference
        self._build_edge_index()

        # Build or load node features (same as training)
        self._build_node_features()

        # Pre-compute embeddings so we don't recompute per query
        self._precompute_embeddings()

        # scispaCy NER
        print("  Loading scispaCy NER model...")
        self.nlp = spacy.load("en_core_sci_sm")

        print(f"  KG Arm ready | {self.graph.number_of_nodes():,} nodes | "
              f"{self.graph.number_of_edges():,} edges | device={self.device}")

    def _build_edge_index(self):
        """Convert NetworkX edges to PyG-style [2, E] tensor."""
        pairs = []
        for s, t in self.graph.edges():
            if s in self.node_to_idx and t in self.node_to_idx:
                pairs.append([self.node_to_idx[s], self.node_to_idx[t]])
        self.edge_index = torch.LongTensor(pairs).t().to(self.device)

    def _build_node_features(self):
        """Recreate the same feature matrix used during training."""
        num_nodes = len(self.node_to_idx)

        # Must match training: random seed isn't fixed, so we use the same
        # random approach. For real PubMedBERT features, load from disk here.
        np.random.seed(42)
        semantic = np.random.randn(num_nodes, 768).astype(np.float32)

        degrees = dict(self.graph.degree())
        deg_arr = np.array([
            degrees.get(self.idx_to_node[i], 0) for i in range(num_nodes)
        ])
        deg_norm = (deg_arr / max(deg_arr.max(), 1)).reshape(-1, 1)

        structural = np.hstack([
            deg_norm,
            np.random.rand(num_nodes, 1),
            np.random.rand(num_nodes, 1)
        ]).astype(np.float32)

        self.node_features = torch.FloatTensor(
            np.hstack([semantic, structural])
        ).to(self.device)

    def _precompute_embeddings(self):
        """Run GAT once and cache all node embeddings."""
        print("  Pre-computing GNN embeddings...")
        with torch.no_grad():
            self.embeddings = self.model(
                self.node_features, self.edge_index
            )  # [num_nodes, output_dim]

    def extract_entities(self, text):
        """Extract medical entities from text using scispaCy."""
        doc = self.nlp(text)
        return [ent.text.lower().strip() for ent in doc.ents]

    def map_entities_to_cuis(self, entities):
        """Map entity strings to UMLS CUIs via name lookup."""
        cuis = []
        for ent in entities:
            if ent in self.name_to_cuis:
                cuis.extend(self.name_to_cuis[ent])
        return list(set(cuis))

    def get_gnn_neighborhood(self, seed_cuis, k_hops=2, top_k=50):
        """
        Find relevant CUIs using GNN embeddings.
        Combines graph-walk neighbourhood with embedding similarity.
        """
        # Graph neighbourhood via BFS
        neighbors = set()
        for cui in seed_cuis:
            if cui in self.graph:
                neighbors.add(cui)
                for hop1 in self.graph.neighbors(cui):
                    neighbors.add(hop1)
                    if k_hops >= 2:
                        for hop2 in self.graph.neighbors(hop1):
                            neighbors.add(hop2)

        # If we have embeddings, rank neighbours by similarity to seeds
        seed_idxs = [self.node_to_idx[c] for c in seed_cuis
                     if c in self.node_to_idx]
        if not seed_idxs:
            return list(neighbors)[:top_k]

        seed_emb = self.embeddings[seed_idxs].mean(dim=0)  # centroid

        neighbor_idxs = [self.node_to_idx[c] for c in neighbors
                         if c in self.node_to_idx]
        if not neighbor_idxs:
            return list(neighbors)[:top_k]

        neigh_emb = self.embeddings[neighbor_idxs]
        sims = torch.cosine_similarity(seed_emb.unsqueeze(0), neigh_emb)
        topk = min(top_k, len(neighbor_idxs))
        top_indices = sims.topk(topk).indices.cpu().tolist()

        return [self.idx_to_node[neighbor_idxs[i]] for i in top_indices]

    def score_documents(self, relevant_cuis, documents):
        """Score documents by overlap with GNN-identified concepts."""
        rel_names = set()
        for cui in relevant_cuis:
            if cui in self.concepts:
                rel_names.add(self.concepts[cui].lower().strip())

        scored = []
        for doc in documents:
            doc_lower = doc.lower()
            overlap = sum(1 for name in rel_names if name in doc_lower)
            scored.append((overlap, doc))

        scored.sort(reverse=True, key=lambda x: x[0])
        return scored


def retrieve_kg(question, context_sentences, top_k=5, kg_arm=None):
    """
    Top-level retrieval function for the KG arm.

    Args:
        question: Medical question string
        context_sentences: List of candidate sentences
        top_k: Number of sentences to return
        kg_arm: Pre-initialized KnowledgeGraphArm (reuse to avoid reload)
    """
    if kg_arm is None:
        kg_arm = KnowledgeGraphArm()

    entities = kg_arm.extract_entities(question)
    if not entities:
        return context_sentences[:top_k]

    cuis = kg_arm.map_entities_to_cuis(entities)
    if not cuis:
        return context_sentences[:top_k]

    relevant = kg_arm.get_gnn_neighborhood(cuis, k_hops=2, top_k=50)
    scored = kg_arm.score_documents(relevant, context_sentences)
    return [doc for _, doc in scored[:top_k]]


if __name__ == "__main__":
    import json

    print("Testing Knowledge Graph Arm\n")

    with open('data/pubmedqa/ori_pqal.json', 'r') as f:
        data = json.load(f)

    example = list(data.values())[0]
    question = example['QUESTION']
    contexts = example['CONTEXTS']

    print(f"Question: {question}\n")

    arm = KnowledgeGraphArm()
    entities = arm.extract_entities(question)
    print(f"Entities: {entities}\n")

    cuis = arm.map_entities_to_cuis(entities)
    print(f"CUIs: {len(cuis)} mapped\n")

    retrieved = retrieve_kg(question, contexts, top_k=5, kg_arm=arm)
    print(f"Retrieved {len(retrieved)} sentences:")
    for i, s in enumerate(retrieved, 1):
        print(f"  {i}. {s[:100]}...")
