"""
Graph Attention Network for Medical Knowledge Graph
Task: Learn embeddings for UMLS concepts via link prediction
"""

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
        
    def forward(self, x, edge_index=None):
        """
        Forward pass through network.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges] (optional for now)
            
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        # Layer 1
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout_layer(x)
        
        # Layer 2
        x = self.fc2(x)
        
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


def compute_link_prediction_loss(model, x, pos_edge_index, neg_edge_index):
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
    z = model(x)
    
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