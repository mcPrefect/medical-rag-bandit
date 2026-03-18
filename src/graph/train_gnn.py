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
import json

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
    print("\nLoading node features...")
    features_path = Path("data/umls/node_features.npy")

    if features_path.exists():
        print(f"  Loading pre-computed features from {features_path}")
        features = np.load(features_path)
        with open("data/umls/node_to_idx.pkl", 'rb') as f:
            node_to_idx = pickle.load(f)
        with open("data/umls/idx_to_node.pkl", 'rb') as f:
            idx_to_node = pickle.load(f)
    else:
        print("  No pre-computed features found, using random")
        num_nodes = graph.number_of_nodes()
        node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}
        idx_to_node = {idx: node for node, idx in node_to_idx.items()}
        semantic = np.random.randn(num_nodes, 768).astype(np.float32)
        degrees = dict(graph.degree())
        deg_arr = np.array([degrees[idx_to_node[i]] for i in range(num_nodes)])
        deg_norm = (deg_arr / max(deg_arr.max(), 1)).reshape(-1, 1)
        structural = np.hstack([deg_norm, np.random.rand(num_nodes, 1), np.random.rand(num_nodes, 1)]).astype(np.float32)
        features = np.hstack([semantic, structural])

    print(f"  Features shape: {features.shape}")
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


def evaluate_model(model, x, edge_index, pos_edges, neg_edges):
    """
    Evaluate model using AUC score.
    """
    model.eval()
    
    with torch.no_grad():
        z = model(x, edge_index)
        
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
    hidden_dim=32,
    output_dim=128,
    num_heads=8,
    patience=10,
    target_auc=0.75
):
    """
    Main training loop.
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

    # The message-passing edge index is the training positives
    # (don't leak val/test edges into message passing)
    msg_edge_index = train_pos
    
    # Initialise model on GPU
    model = MedicalGAT(
        input_dim=771,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_heads=num_heads
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print(f"Target: Val AUC >= {target_auc}\n")

    best_val_auc = 0
    no_improve = 0
    history = []

    for epoch in range(num_epochs):
        t0 = time.time()

        # Train step
        model.train()
        optimizer.zero_grad()
        loss = compute_link_prediction_loss(
            model, x, msg_edge_index, train_pos, train_neg
        )
        loss.backward()
        optimizer.step()

        # Eval every 5 epochs
        if epoch % 5 == 0:
            val_auc = evaluate_model(model, x, msg_edge_index, val_pos, val_neg)
            elapsed = time.time() - t0

            history.append({
                "epoch": epoch,
                "loss": round(loss.item(), 4),
                "val_auc": round(val_auc, 4)
            })

            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | "
                  f"Val AUC: {val_auc:.4f} | {elapsed:.2f}s")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                no_improve = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_auc': val_auc,
                    'node_to_idx': node_to_idx,
                    'idx_to_node': idx_to_node,
                    'config': {
                        'input_dim': 771,
                        'hidden_dim': hidden_dim,
                        'output_dim': output_dim,
                        'num_heads': num_heads
                    }
                }, output_path / "gnn_model_best.pt")
                print(f"  -> Best model saved (AUC: {val_auc:.4f})")

                if val_auc >= target_auc:
                    print(f"\n  Target AUC reached! ({val_auc:.4f} >= {target_auc})")
                    break
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"\nEarly stopping (no improvement for {patience} eval rounds)")
                break

    # Final evaluation on test set
    print("\nFINAL EVALUATION")
    checkpoint = torch.load(output_path / "gnn_model_best.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_auc = evaluate_model(model, x, msg_edge_index, test_pos, test_neg)

    print(f"Best Val AUC: {best_val_auc:.4f}")
    print(f"Test AUC:     {test_auc:.4f}")

    if test_auc >= target_auc:
        print(f"\nSUCCESS: Test AUC {test_auc:.4f} >= target {target_auc}")
    else:
        print(f"\nBelow target: {test_auc:.4f} < {target_auc}")
        print("Consider: more epochs, PubMedBERT features, or tuning heads/hidden_dim")

    # Save training history
    with open(output_path / "gnn_training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nModel saved to: {output_path / 'gnn_model_best.pt'}")
    print(f"History saved to: {output_path / 'gnn_training_history.json'}")
    return model, test_auc


if __name__ == "__main__":
    model, test_auc = train_gnn(
        data_dir="data/umls",
        output_dir="models",
        num_epochs=100,
        target_auc=0.75
    )
