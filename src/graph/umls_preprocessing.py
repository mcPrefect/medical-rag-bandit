"""
UMLS Preprocessing: Extract Clinically-Relevant Subgraph
Input: Full UMLS (37GB uncompressed)
Output: Subgraph for GP queries 
"""

import os
import pickle
import networkx as nx
from collections import defaultdict
from pathlib import Path


# Top 50 common conditions for GP queries
SEED_DISEASES = [
    "hypertension", "diabetes mellitus", "pneumonia", "urinary tract infection",
    "chronic obstructive pulmonary disease", "asthma", "depression", "anxiety",
    "migraine", "gastroesophageal reflux disease", "osteoarthritis", "back pain",
    "hypothyroidism", "atrial fibrillation", "heart failure", "coronary artery disease",
    "hyperlipidemia", "obesity", "insomnia", "allergic rhinitis",
    "skin infection", "upper respiratory infection", "bronchitis", "sinusitis",
    "otitis media", "gastroenteritis", "irritable bowel syndrome", "hemorrhoids",
    "eczema", "psoriasis", "acne", "dermatitis", "urticaria",
    "anemia", "thyroid disease", "gout", "osteoporosis", "fibromyalgia",
    "chronic fatigue syndrome", "vertigo", "benign prostatic hyperplasia",
    "urinary incontinence", "peptic ulcer", "diverticulitis", "constipation",
    "diarrhea", "nausea", "vomiting", "abdominal pain"
]


def load_concepts(mrconso_path):
    """
    Load UMLS concepts from MRCONSO.RRF
    
    MRCONSO format (pipe-delimited):
    CUI|LAT|TS|LUI|STT|SUI|ISPREF|AUI|SAUI|SCUI|SDUI|SAB|TTY|CODE|STR|SRL|SUPPRESS|CVF|
    
    Returns:
        dict: {CUI: concept_name}
    """
    print(f"Loading concepts from {mrconso_path}...")
    
    concepts = {}
    concept_names = defaultdict(list)
    
    with open(mrconso_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print(f"  Processed {i:,} lines, found {len(concepts):,} concepts")
            
            fields = line.strip().split('|')
            if len(fields) < 15:
                continue
            
            cui = fields[0]           # Concept Unique Identifier
            language = fields[1]      # LAT (language)
            concept_name = fields[14] # STR (string/name)
            
            # Only English concepts
            if language != 'ENG':
                continue
            
            # Store concept
            if cui not in concepts:
                concepts[cui] = concept_name
            
            # Track all names for this concept (for matching)
            concept_names[cui].append(concept_name.lower())
    
    print(f"Loaded {len(concepts):,} unique concepts")
    return concepts, concept_names


def find_seed_concepts(concepts, concept_names, seed_terms):
    """
    Find CUIs for seed disease terms.
    
    Returns:
        set: CUIs matching seed terms
    """
    print(f"\nFinding CUIs for {len(seed_terms)} seed diseases...")
    
    seed_cuis = set()
    
    for seed in seed_terms:
        seed_lower = seed.lower()
        for cui, names in concept_names.items():
            # Exact match or contains
            if seed_lower in names or any(seed_lower in name for name in names):
                seed_cuis.add(cui)
    
    print(f"Found {len(seed_cuis)} seed CUIs")
    return seed_cuis


def load_relationships(mrrel_path):
    """
    Load UMLS relationships from MRREL.RRF
    
    MRREL format (pipe-delimited):
    CUI1|AUI1|STYPE1|REL|CUI2|AUI2|STYPE2|RELA|RUI|SRUI|SAB|SL|RG|DIR|SUPPRESS|CVF|
    
    Returns:
        list: [(cui1, cui2, rel_type), ...]
    """
    print(f"\nLoading relationships from {mrrel_path}...")
    
    relationships = []
    relevant_reltypes = {'RO', 'RN', 'RB', 'RL'}  # Clinically relevant types
    
    with open(mrrel_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print(f"  Processed {i:,} lines, found {len(relationships):,} relationships")
            
            fields = line.strip().split('|')
            if len(fields) < 8:
                continue
            
            cui1 = fields[0]      # Source concept
            rel = fields[3]       # Relationship type
            cui2 = fields[4]      # Target concept
            rela = fields[7]      # Specific relationship (e.g., "treats", "causes")
            
            # Only keep clinically relevant relationships
            if rel in relevant_reltypes:
                relationships.append((cui1, cui2, rela if rela else rel))
    
    print(f"Loaded {len(relationships):,} relationships")
    return relationships


def build_subgraph(seed_cuis, relationships, k_hops=2):
    """
    Build k-hop subgraph around seed concepts.
    
    Args:
        seed_cuis: Starting concepts
        relationships: All UMLS relationships
        k_hops: How many hops to expand
        
    Returns:
        NetworkX graph
    """
    print(f"\nBuilding {k_hops}-hop subgraph from {len(seed_cuis)} seeds")
    
    # Build full graph first
    G = nx.DiGraph()
    for cui1, cui2, rel_type in relationships:
        G.add_edge(cui1, cui2, rel_type=rel_type)
    
    print(f"Full graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    
    # Extract subgraph via k-hop expansion
    relevant_nodes = set(seed_cuis)
    
    for hop in range(k_hops):
        print(f"  Hop {hop + 1}: {len(relevant_nodes):,} nodes")
        new_nodes = set()
        
        for node in list(relevant_nodes):
            if node in G:
                # Add neighbors
                new_nodes.update(G.successors(node))
                new_nodes.update(G.predecessors(node))
        
        relevant_nodes.update(new_nodes)
    
    # Create subgraph
    subgraph = G.subgraph(relevant_nodes).copy()
    
    print(f"Subgraph: {subgraph.number_of_nodes():,} nodes, {subgraph.number_of_edges():,} edges")
    return subgraph


def preprocess_umls(umls_dir, output_dir):
    """
    Main preprocessing pipeline.
    
    Args:
        umls_dir: Path to extracted UMLS directory
        output_dir: Where to save processed files
    """
    print("UMLS PREPROCESSING PIPELINE")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # File paths 
    mrconso_path = Path(umls_dir) / "META" / "MRCONSO.RRF"
    mrrel_path = Path(umls_dir) / "META" / "MRREL.RRF"
    
    if not mrconso_path.exists():
        print(f"ERROR: {mrconso_path} not found!")
        return
    
    # Step 1: Load concepts
    concepts, concept_names = load_concepts(mrconso_path)
    
    # Step 2: Find seed CUIs
    seed_cuis = find_seed_concepts(concepts, concept_names, SEED_DISEASES)
    
    # Step 3: Load relationships
    relationships = load_relationships(mrrel_path)
    
    # Step 4: Build subgraph
    subgraph = build_subgraph(seed_cuis, relationships, k_hops=2)
    
    # Step 5: Save outputs
    print("\nSaving preprocessed data")
    
    # Save subgraph
    with open(output_dir / "subgraph.pkl", 'wb') as f:
        pickle.dump(subgraph, f)
    print(f"  Saved: {output_dir / 'subgraph.pkl'}")
    
    # Save concept names
    with open(output_dir / "concepts.pkl", 'wb') as f:
        pickle.dump(concepts, f)
    print(f"  Saved: {output_dir / 'concepts.pkl'}")
    
    # Save seed CUIs for reference
    with open(output_dir / "seed_cuis.txt", 'w') as f:
        for cui in sorted(seed_cuis):
            f.write(f"{cui}\t{concepts.get(cui, 'Unknown')}\n")
    print(f"  Saved: {output_dir / 'seed_cuis.txt'}")
    
    print("PREPROCESSING COMPLETE!")

    print(f"\nOutput files in: {output_dir}")
    print(f"  - subgraph.pkl ({subgraph.number_of_nodes():,} nodes, {subgraph.number_of_edges():,} edges)")
    print(f"  - concepts.pkl ({len(concepts):,} concepts)")
    print(f"  - seed_cuis.txt ({len(seed_cuis)} seeds)")


# Example usage
if __name__ == "__main__":
    umls_dir = "/home/demouser/medical-rag-bandit/data/umls/extracted/2025AB"
    output_dir = "/home/demouser/medical-rag-bandit/data/umls"
    
    
    preprocess_umls(umls_dir, output_dir)

#    /home/demouser/medical-rag-bandit/data/umls/extracted/2025AB/META/MRCONSO.RRF