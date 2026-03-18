"""
Compute PubMedBERT embeddings for UMLS concept nodes.

Reads concept names from concepts.pkl, encodes them in batches
with PubMedBERT, computes structural graph features, and saves
the combined 771-dim feature matrix to disk.
"""

import pickle
import numpy as np
import torch
import networkx as nx
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import time
import math


def compute_pubmedbert_embeddings(concepts, idx_to_node, batch_size=128,
                                  device='cuda'):
    """
    Encode concept names with PubMedBERT, mean-pool last hidden state.

    Args:
        concepts: dict of {CUI: concept_name}
        idx_to_node: dict of {idx: CUI}
        batch_size: GPU batch size
        device: cuda or cpu

    Returns:
        np.ndarray of shape [num_nodes, 768]
    """
    model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    print(f"Loading PubMedBERT from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    num_nodes = len(idx_to_node)
    embeddings = np.zeros((num_nodes, 768), dtype=np.float32)

    # Build text for each node in index order
    texts = []
    for i in range(num_nodes):
        cui = idx_to_node[i]
        name = concepts.get(cui, "unknown concept")
        texts.append(name)

    num_batches = math.ceil(num_nodes / batch_size)
    print(f"Encoding {num_nodes:,} concepts in {num_batches:,} batches "
          f"(batch_size={batch_size})...")

    t0 = time.time()
    with torch.no_grad():
        for b in range(num_batches):
            start = b * batch_size
            end = min(start + batch_size, num_nodes)
            batch_texts = texts[start:end]

            encoded = tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=64, return_tensors='pt'
            ).to(device)

            output = model(**encoded)
            # Mean pool over token dimension, respecting attention mask
            mask = encoded['attention_mask'].unsqueeze(-1).float()
            pooled = (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            embeddings[start:end] = pooled.cpu().numpy()

            if b % 500 == 0 and b > 0:
                elapsed = time.time() - t0
                rate = b / elapsed
                eta = (num_batches - b) / rate
                print(f"  Batch {b:,}/{num_batches:,} | "
                      f"{elapsed:.0f}s elapsed | ~{eta:.0f}s remaining")

    elapsed = time.time() - t0
    print(f"Encoding done in {elapsed:.1f}s")
    return embeddings


def compute_structural_features(graph, idx_to_node):
    """
    Compute degree, approximate betweenness, and pagerank for each node.

    Returns:
        np.ndarray of shape [num_nodes, 3]
    """
    num_nodes = len(idx_to_node)
    print(f"\nComputing structural features for {num_nodes:,} nodes...")

    # Degree (fast)
    print("  Computing degree centrality...")
    degrees = dict(graph.degree())
    deg_arr = np.array([degrees.get(idx_to_node[i], 0)
                        for i in range(num_nodes)], dtype=np.float32)
    deg_arr /= max(deg_arr.max(), 1)

    # Betweenness (expensive on large graphs, use sampling)
    print("  Computing approximate betweenness centrality (sampled)...")
    t0 = time.time()
    try:
        betweenness = nx.betweenness_centrality(graph, k=min(50, num_nodes)) # was 500 but took too long
        bet_arr = np.array([betweenness.get(idx_to_node[i], 0)
                            for i in range(num_nodes)], dtype=np.float32)
        max_bet = bet_arr.max()
        if max_bet > 0:
            bet_arr /= max_bet
        print(f"    Done in {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"    Betweenness failed ({e}), using degree as proxy")
        bet_arr = deg_arr.copy()

    # PageRank
    print("  Computing PageRank...")
    t0 = time.time()
    try:
        pagerank = nx.pagerank(graph, max_iter=50, tol=1e-4)
        pr_arr = np.array([pagerank.get(idx_to_node[i], 0)
                           for i in range(num_nodes)], dtype=np.float32)
        max_pr = pr_arr.max()
        if max_pr > 0:
            pr_arr /= max_pr
        print(f"    Done in {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"    PageRank failed ({e}), using zeros")
        pr_arr = np.zeros(num_nodes, dtype=np.float32)

    structural = np.column_stack([deg_arr, bet_arr, pr_arr])
    print(f"  Structural features shape: {structural.shape}")
    return structural


def main(data_dir="data/umls", output_dir="data/umls", batch_size=128):
    data_path = Path(data_dir)
    output_path = Path(output_dir)

    # Load graph and concepts
    print("Loading graph and concepts...")
    with open(data_path / "subgraph.pkl", 'rb') as f:
        graph = pickle.load(f)
    with open(data_path / "concepts.pkl", 'rb') as f:
        concepts = pickle.load(f)

    # Build node index (same ordering as train_gnn uses)
    node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    num_nodes = len(node_to_idx)
    print(f"Graph: {num_nodes:,} nodes, {graph.number_of_edges():,} edges")

    # Compute PubMedBERT embeddings
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    semantic = compute_pubmedbert_embeddings(
        concepts, idx_to_node, batch_size=batch_size, device=device
    )

    # Compute structural features
    structural = compute_structural_features(graph, idx_to_node)

    # Combine: [num_nodes, 771]
    features = np.hstack([semantic, structural]).astype(np.float32)
    print(f"\nFinal feature matrix: {features.shape}")

    # Save features and node mappings
    np.save(output_path / "node_features.npy", features)
    with open(output_path / "node_to_idx.pkl", 'wb') as f:
        pickle.dump(node_to_idx, f)
    with open(output_path / "idx_to_node.pkl", 'wb') as f:
        pickle.dump(idx_to_node, f)

    print(f"\nSaved to {output_path}:")
    print(f"  node_features.npy  ({features.nbytes / 1e9:.2f} GB)")
    print(f"  node_to_idx.pkl")
    print(f"  idx_to_node.pkl")


if __name__ == "__main__":
    main()
