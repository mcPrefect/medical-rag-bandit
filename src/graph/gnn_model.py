"""
Graph Attention Network for Medical Knowledge Graph
Task: Learn embeddings for UMLS concepts via link prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class MedicalGAT(nn.Module):
    """
    2-layer GAT for medical concept embeddings.

    Architecture:
        Layer 1: GATConv(in_dim, 32, heads=8) -> 256 dims
        Layer 2: GATConv(256, out_dim, heads=1, concat=False) -> 128 dims

    Input: node features (768 semantic + 3 structural = 771 dims)
    Output: node embeddings (128 dims)
    """
    
    def __init__(
        self,
        input_dim=771,
        hidden_dim=32,
        output_dim=128,
        num_heads=8,
        dropout=0.3
    ):
        super().__init__()

        self.dropout = dropout

        # Layer 1: multi-head attention
        self.gat1 = GATConv(
            input_dim, hidden_dim, heads=num_heads, dropout=dropout
        )
        # Layer 2: single-head output
        self.gat2 = GATConv(
            hidden_dim * num_heads, output_dim, heads=1,
            concat=False, dropout=dropout
        )
        
    def forward(self, x, edge_index):
        """
        Forward pass through network.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        
        return x
    
    def predict_edge(self, z_i, z_j):
        """
        Predict probability of edge between nodes i and j.
        
        Args:
            z_i: Embedding of node i [output_dim]
            z_j: Embedding of node j [output_dim]
            
        Returns:
            Edge probability [0, 1]
        """
        # Dot product + sigmoid
        return torch.sigmoid((z_i * z_j).sum())
    
    def decode_all_edges(self, z, edge_index):
        """
        Predict probabilities for all edges in edge_index.
        
        Args:
            z: All node embeddings [num_nodes, output_dim]
            edge_index: Edges to predict [2, num_edges]
            
        Returns:
            Edge probabilities [num_edges]
        """
        # Get embeddings for source and target nodes
        z_i = z[edge_index[0]]  # Source nodes
        z_j = z[edge_index[1]]  # Target nodes
        
        # Dot product for each edge
        scores = (z_i * z_j).sum(dim=1)
        
        return torch.sigmoid(scores)


def compute_link_prediction_loss(model, x, edge_index, pos_edge_index, neg_edge_index):
    """
    Compute binary cross-entropy loss for link prediction.
    
    Args:
        model: GNN model
        x: Node features
        pos_edge_index: Positive (real) edges
        neg_edge_index: Negative (non-existent) edges
        
    Returns:
        Total loss
    """
    # Get embeddings
    z = model(x, edge_index)
    
    # Positive edge scores (should be high)
    pos_scores = model.decode_all_edges(z, pos_edge_index)
    pos_loss = -torch.log(pos_scores + 1e-15).mean()
    
    # Negative edge scores (should be low)
    neg_scores = model.decode_all_edges(z, neg_edge_index)
    neg_loss = -torch.log(1 - neg_scores + 1e-15).mean()
    
    return pos_loss + neg_loss


# Test the model
if __name__ == "__main__":
    print("Testing Medical GAT Model")
    
    # Create dummy data
    num_nodes = 100
    input_dim = 771
    num_edges = 500
    
    # Random node features
    x = torch.randn(num_nodes, input_dim)
    
    # Random edges
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Initialise model
    model = MedicalGAT(input_dim=771, hidden_dim=256, output_dim=128)
    
    print(f"Model initialized:")
    print(f"  Input dim: {model.input_dim}")
    print(f"  Hidden dim: {model.hidden_dim}")
    print(f"  Output dim: {model.output_dim}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    embeddings = model(x)
    print(f"\nForward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {embeddings.shape}")
    
    # Test edge prediction
    pos_edge_index = edge_index[:, :250]  # First 250 as positive
    neg_edge_index = edge_index[:, 250:]  # Rest as negative
    
    loss = compute_link_prediction_loss(model, x, pos_edge_index, neg_edge_index)
    print(f"\nLink prediction loss: {loss.item():.4f}")
    
    print("Model test complete!")