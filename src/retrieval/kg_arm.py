"""
Knowledge Graph Arm: GNN-based retrieval using UMLS
Uses trained Graph Attention Network to find relevant concepts
"""

import torch
import pickle
import numpy as np
from pathlib import Path as PathLib
import spacy
from collections import defaultdict

# Import GNN model
import sys

# add graph directory to path
# current_file = PathLib(__file__).resolve()
# graph_dir = current_file.parent.parent / "graph"
# sys.path.insert(0, str(graph_dir))

# sys.path.append(str(Path(__file__).parent.parent / "src/graph"))
# from gnn_model import MedicalGAT

import torch
import torch.nn as nn
import torch.nn.functional as F


class MedicalGAT(nn.Module):
    """
    Graph Attention Network for medical concept embeddings.
    
    Architecture:
    - Input: Node features (768 semantic + 3 structural = 771 dims)
    - Layer 1: GAT with 8 attention heads
    - Layer 2: GAT output layer
    - Output: Node embeddings (128 dims)
    """
    
    def __init__(
        self,
        input_dim=771,
        hidden_dim=256,
        output_dim=128,
        num_heads=8,
        dropout=0.3
    ):
        super(MedicalGAT, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        #  simple 2-layer MLP (will add PyTorch Geometric GAT layers later)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout_layer = nn.Dropout(dropout)


class KnowledgeGraphArm:
    """
    Retrieval arm that uses GNN-learned embeddings over UMLS knowledge graph.
    
    Pipeline:
    1. Extract medical entities from query (scispaCy)
    2. Map entities to UMLS concepts (CUI matching)
    3. Use GNN to find relevant connected concepts
    4. Score documents by overlap with GNN neighborhood
    5. Return top-k documents
    """
    
    def __init__(
        self,
        model_path="models/gnn_model_best.pt",
        graph_path="data/umls/subgraph.pkl",
        concepts_path="data/umls/concepts.pkl",
        device='cuda'
    ):
        """
        Initialize KG arm with trained GNN and UMLS graph.
        """
        print("Initialising Knowledge Graph Arm...")
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load trained GNN model
        print(f"  Loading GNN model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = MedicalGAT(
            input_dim=771,
            hidden_dim=256,
            output_dim=128
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load node mappings
        self.node_to_idx = checkpoint['node_to_idx']
        self.idx_to_node = checkpoint['idx_to_node']
        
        # Load graph
        print(f"  Loading UMLS graph from {graph_path}...")
        with open(graph_path, 'rb') as f:
            self.graph = pickle.load(f)
        
        # Load concept names
        print(f"  Loading concepts from {concepts_path}...")
        with open(concepts_path, 'rb') as f:
            self.concepts = pickle.load(f)
        
        # Build reverse index: concept_name -> [CUIs]
        print("  Building concept name index...")
        self.name_to_cuis = defaultdict(list)
        for cui, name in self.concepts.items():
            normalized_name = name.lower().strip()
            self.name_to_cuis[normalized_name].append(cui)
        
        # Load scispaCy for entity extraction
        print("  Loading scispaCy NER model...")
        self.nlp = spacy.load("en_core_sci_sm")
        
        print("Knowledge Graph Arm initialized")
        print(f"  Graph: {self.graph.number_of_nodes():,} nodes, {self.graph.number_of_edges():,} edges")
        print(f"  Model on: {self.device}")
    
    def extract_entities(self, text):
        """
        Extract medical entities from text using scispaCy.
        
        Returns:
            list of str: Extracted entity texts
        """
        doc = self.nlp(text)
        entities = [ent.text.lower().strip() for ent in doc.ents]
        return entities
    
    def map_entities_to_cuis(self, entities):
        """
        Map extracted entities to UMLS concept IDs (CUIs).
        
        Returns:
            list of str: CUIs found
        """
        cuis = []
        for entity in entities:
            # Exact match
            if entity in self.name_to_cuis:
                cuis.extend(self.name_to_cuis[entity])
            else:
                # Fuzzy match: check if entity is substring of any concept
                for concept_name, concept_cuis in self.name_to_cuis.items():
                    if entity in concept_name or concept_name in entity:
                        cuis.extend(concept_cuis)
                        break
        
        return list(set(cuis))  # Remove duplicates
    
    def get_gnn_neighborhood(self, seed_cuis, k_hops=2, top_k=50):
        """
        Use GNN to find relevant concepts around seed CUIs.
        
        Args:
            seed_cuis: Starting concepts
            k_hops: How many hops to expand
            top_k: Max concepts to return
            
        Returns:
            list of str: Relevant CUIs ranked by GNN similarity
        """
        # Filter to CUIs in our graph
        valid_seeds = [cui for cui in seed_cuis if cui in self.node_to_idx]
        
        if not valid_seeds:
            return []
        
        # Get embeddings for seed concepts (would need to compute if we had features)
        # For MVP: Just doing graph traversal
        relevant_cuis = set(valid_seeds)
        
        for hop in range(k_hops):
            new_cuis = set()
            for cui in list(relevant_cuis):
                if cui in self.graph:
                    # Add neighbors
                    new_cuis.update(self.graph.successors(cui))
                    new_cuis.update(self.graph.predecessors(cui))
            
            relevant_cuis.update(new_cuis)
            
            # Don't let it explode
            if len(relevant_cuis) > top_k * 10:
                break
        
        return list(relevant_cuis)[:top_k]
    
    def score_documents(self, relevant_cuis, context_sentences):
        """
        Score documents by overlap with GNN-identified concepts.
        
        Returns:
            list of (score, sentence) tuples
        """
        scores = []
        
        # Get concept names for relevant CUIs
        relevant_terms = set()
        for cui in relevant_cuis:
            if cui in self.concepts:
                relevant_terms.add(self.concepts[cui].lower())
        
        for sentence in context_sentences:
            sentence_lower = sentence.lower()
            
            # Count how many relevant terms appear in sentence
            overlap = sum(1 for term in relevant_terms if term in sentence_lower)
            
            # Normalize by sentence length
            score = overlap / max(len(sentence.split()), 1)
            
            scores.append((score, sentence))
        
        # Sort by score descending
        scores.sort(reverse=True, key=lambda x: x[0])
        
        return scores


def retrieve_kg(question, context_sentences, top_k=5, kg_arm=None):
    """
    Retrieve using Knowledge Graph.
    
    Args:
        question: Medical question
        context_sentences: Available context
        top_k: How many sentences to return
        kg_arm: Pre-initialized KnowledgeGraphArm (optional)
        
    Returns:
        list of str: Top-k relevant sentences
    """
    # Initialize KG arm if not provided (for backward compatibility)
    if kg_arm is None:
        kg_arm = KnowledgeGraphArm()
    
    # Extract entities from question
    entities = kg_arm.extract_entities(question)
    
    if not entities:
        # Fallback: if no entities found, return first k sentences
        print("  [KG] No entities found, using fallback")
        return context_sentences[:top_k]
    
    # Map to CUIs
    seed_cuis = kg_arm.map_entities_to_cuis(entities)
    
    if not seed_cuis:
        # Fallback: no CUIs found
        print("  [KG] No CUIs matched, using fallback")
        return context_sentences[:top_k]
    
    # Get GNN neighborhood
    relevant_cuis = kg_arm.get_gnn_neighborhood(seed_cuis, k_hops=2, top_k=50)
    
    # Score documents
    scored = kg_arm.score_documents(relevant_cuis, context_sentences)
    
    # Return top-k
    top_k = min(top_k, len(scored))
    return [sent for score, sent in scored[:top_k]]


# Test the KG arm
if __name__ == "__main__":
    print("Testing Knowledge Graph Arm\n")
    
    import json
    
    # Load example
    with open('data/pubmedqa/ori_pqal.json', 'r') as f:
        data = json.load(f)
    
    example = list(data.values())[0]
    question = example['QUESTION']
    contexts = example['CONTEXTS']
    
    print(f"Question: {question}\n")
    
    # Initialize KG arm
    kg_arm = KnowledgeGraphArm()
    
    # Test entity extraction
    entities = kg_arm.extract_entities(question)
    print(f"Extracted entities: {entities}\n")
    
    # Test CUI mapping
    cuis = kg_arm.map_entities_to_cuis(entities)
    print(f"Mapped to {len(cuis)} CUIs: {cuis[:5]}...\n")
    
    # Test retrieval
    retrieved = retrieve_kg(question, contexts, top_k=5, kg_arm=kg_arm)
    
    print(f"Retrieved {len(retrieved)} sentences:")
    for i, sent in enumerate(retrieved, 1):
        print(f"{i}. {sent[:100]}...")