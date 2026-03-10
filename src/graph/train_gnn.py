"""
Train Graph Attention Network on UMLS Medical Knowledge Graph
Task: Link Prediction 
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
import time

from gnn_model import MedicalGAT, compute_link_prediction_loss


def load_preprocessed_data(data_dir="data/umls"):
    """
    Load preprocessed UMLS subgraph.
    
    Returns:
        graph: NetworkX graph
        concepts: Dict of concept names
    """
    print("Loading preprocessed UMLS data...")
    
    data_path = Path(data_dir)
    
    with open(data_path / "subgraph.pkl", 'rb') as f:
        graph = pickle.load(f)
    
    with open(data_path / "concepts.pkl", 'rb') as f:
        concepts = pickle.load(f)
    
    print(f"Loaded graph: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")
    print(f"Loaded {len(concepts):,} concept names")
    
    return graph, concepts


def create_node_features(graph, concepts, feature_dim=771):
    """
    Create node features for GNN.
    
    For now its Simple random features (768 dims) + graph stats (3 dims)
    Later I will replace with PubMedBERT embeddings
    
    Returns:
        torch.Tensor: [num_nodes, feature_dim]
    """
    print("\nCreating node features...")
    
    num_nodes = graph.number_of_nodes()
    
    # Create node ID mapping (CUI -> integer index)
    node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    
    # Initialise features
    # For MVP: Random features (will replace with PubMedBERT later)
    semantic_features = np.random.randn(num_nodes, 768).astype(np.float32)
    
    # Add graph structural features
    import networkx as nx
    degrees = dict(graph.degree())
    degree_features = np.array([degrees[idx_to_node[i]] for i in range(num_nodes)])
    degree_features = (degree_features / degree_features.max()).reshape(-1, 1)  # Normalize
    
    # Simple structural features for now
    structural_features = np.hstack([
        degree_features,
        np.random.rand(num_nodes, 1),  # Placeholder for betweenness
        np.random.rand(num_nodes, 1)   # Placeholder for pagerank
    ]).astype(np.float32)
    
    # Combine
    features = np.hstack([semantic_features, structural_features])
    
    print(f"Created features: {features.shape}")
    return torch.FloatTensor(features), node_to_idx, idx_to_node


def split_edges(graph, train_ratio=0.7, val_ratio=0.15):
    """
    Split edges into train/val/test sets.
    
    Returns:
        train_edges, val_edges, test_edges
    """
    print("\nSplitting edges...")
    
    edges = list(graph.edges())
    np.random.shuffle(edges)
    
    num_edges = len(edges)
    num_train = int(num_edges * train_ratio)
    num_val = int(num_edges * val_ratio)
    
    train_edges = edges[:num_train]
    val_edges = edges[num_train:num_train + num_val]
    test_edges = edges[num_train + num_val:]
    
    print(f"Train: {len(train_edges):,} edges")
    print(f"Val: {len(val_edges):,} edges")
    print(f"Test: {len(test_edges):,} edges")
    
    return train_edges, val_edges, test_edges


def edges_to_tensor(edges, node_to_idx):
    """Convert edge list to PyTorch tensor format."""
    edge_index = []
    for src, tgt in edges:
        if src in node_to_idx and tgt in node_to_idx:
            edge_index.append([node_to_idx[src], node_to_idx[tgt]])
    
    if not edge_index:
        return torch.zeros((2, 0), dtype=torch.long)
    
    return torch.LongTensor(edge_index).t()


def create_negative_samples(num_nodes, positive_edges, num_samples):
    """
    Create negative edge samples (random node pairs with no edge).
    """
    positive_set = set(map(tuple, positive_edges.t().tolist()))
    negative_edges = []
    
    while len(negative_edges) < num_samples:
        src = np.random.randint(0, num_nodes)
        tgt = np.random.randint(0, num_nodes)
        
        if src != tgt and (src, tgt) not in positive_set:
            negative_edges.append([src, tgt])
    
    return torch.LongTensor(negative_edges).t()


def evaluate_model(model, x, pos_edges, neg_edges):
    """
    Evaluate model using AUC score.
    """
    model.eval()
    
    with torch.no_grad():
        z = model(x)
        
        # Positive edge scores
        pos_scores = model.decode_all_edges(z, pos_edges).cpu().numpy()
        
        # Negative edge scores
        neg_scores = model.decode_all_edges(z, neg_edges).cpu().numpy()
        
        # Compute AUC
        labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
        scores = np.concatenate([pos_scores, neg_scores])
        
        auc = roc_auc_score(labels, scores)
    
    return auc


def train_gnn(
    data_dir="data/umls",
    output_dir="models",
    num_epochs=100,
    learning_rate=0.001,
    hidden_dim=256,
    output_dim=128,
    patience=10,
    target_auc=0.75
):
    """
    Main training loop.
    
    Args:
        data_dir: Where preprocessed UMLS data is
        output_dir: Where to save trained model
        num_epochs: Max training epochs
        learning_rate: Learning rate
        hidden_dim: Hidden layer size
        output_dim: Output embedding size
        patience: Early stopping patience
        target_auc: Stop if validation AUC exceeds this
    """
    print("GNN TRAINING PIPELINE")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    graph, concepts = load_preprocessed_data(data_dir)
    
    # Create features
    x, node_to_idx, idx_to_node = create_node_features(graph, concepts)
    num_nodes = x.shape[0]
    
    # Split edges
    train_edges, val_edges, test_edges = split_edges(graph)
    
    # Convert to tensors
    train_pos = edges_to_tensor(train_edges, node_to_idx)
    val_pos = edges_to_tensor(val_edges, node_to_idx)
    test_pos = edges_to_tensor(test_edges, node_to_idx)
    
    # Create negative samples
    train_neg = create_negative_samples(num_nodes, train_pos, len(train_edges))
    val_neg = create_negative_samples(num_nodes, val_pos, len(val_edges))
    test_neg = create_negative_samples(num_nodes, test_pos, len(test_edges))
    
    print(f"\nDataset prepared:")
    print(f"  Nodes: {num_nodes:,}")
    print(f"  Train edges: {train_pos.shape[1]:,} pos, {train_neg.shape[1]:,} neg")
    print(f"  Val edges: {val_pos.shape[1]:,} pos, {val_neg.shape[1]:,} neg")
    print(f"  Test edges: {test_pos.shape[1]:,} pos, {test_neg.shape[1]:,} neg")
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Move data to GPU
    x = x.to(device)
    train_pos = train_pos.to(device)
    train_neg = train_neg.to(device)
    val_pos = val_pos.to(device)
    val_neg = val_neg.to(device)
    test_pos = test_pos.to(device)
    test_neg = test_neg.to(device)
    
    # Initialise model on GPU
    model = MedicalGAT(
        input_dim=771,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Target: Val AUC ≥ {target_auc}")
    
    # Training loop
    best_val_auc = 0
    epochs_without_improvement = 0
    
    print("TRAINING")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train
        model.train()
        optimizer.zero_grad()
        
        loss = compute_link_prediction_loss(model, x, train_pos, train_neg)
        loss.backward()
        optimizer.step()
        
        # Evaluate every 5 epochs
        if epoch % 5 == 0:
            val_auc = evaluate_model(model, x, val_pos, val_neg)
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Val AUC: {val_auc:.4f} | Time: {epoch_time:.2f}s")
            
            # Save best model
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                epochs_without_improvement = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_auc': val_auc,
                    'node_to_idx': node_to_idx,
                    'idx_to_node': idx_to_node
                }, output_path / "gnn_model_best.pt")
                
                print(f"  Best model saved (AUC: {val_auc:.4f})")
                
                # Check if target reached
                if val_auc >= target_auc:
                    print(f"\nTarget AUC reached! ({val_auc:.4f} ≥ {target_auc})")
                    break
            else:
                epochs_without_improvement += 1
            
            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping (no improvement for {patience} epochs)")
                break
    
    # Final evaluation on test set
    print("FINAL EVALUATION")
    
    # Load best model
    checkpoint = torch.load(output_path / "gnn_model_best.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_auc = evaluate_model(model, x, test_pos, test_neg)
    
    print(f"\nBest Val AUC: {best_val_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    if test_auc >= target_auc:
        print(f"\nSUCCESS: Test AUC {test_auc:.4f} ≥ target {target_auc}")
    else:
        print(f"\nBelow target: Test AUC {test_auc:.4f} < target {target_auc}")
    
    print(f"Model saved to: {output_path / 'gnn_model_best.pt'}")
    
    return model, test_auc


if __name__ == "__main__":
    # Train GNN
    model, test_auc = train_gnn(
        data_dir="data/umls",
        output_dir="models",
        num_epochs=100,
        target_auc=0.75
    )