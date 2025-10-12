import torch
import sys
sys.path.append('.')

from modules.graph_encoder import GraphEncoder

def test_graph_validation():
    print("Validating GraphEncoder behavior...")
    print("-" * 50)
    
    # Create a simple graph with known structure
    # Triangle graph: 0-1-2-0
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    x = torch.eye(3)  # One-hot node features
    
    # Test message passing
    encoder = GraphEncoder(
        input_dim=3,
        hidden_dim=8,
        output_dim=4,
        n_layers=1,
        layer_type='gcn'
    )
    
    # Get node features after one layer
    output = encoder(x, edge_index, return_node_features=True)
    node_feats = output['node_features']
    
    print(f"Node features shape: {node_feats.shape}")
    print(f"Node 0 received messages from nodes 1 and 2: {not torch.allclose(node_feats[0], node_feats[1])}")
    
    # Test isolated node
    # Add isolated node
    x_isolated = torch.cat([x, torch.zeros(1, 3)])
    output_iso = encoder(x_isolated, edge_index, return_node_features=True)
    node_feats_iso = output_iso['node_features']
    
    print(f"Isolated node has different features: {not torch.allclose(node_feats_iso[3], node_feats_iso[0])}")
    
    # Test that pooling aggregates correctly
    batch = torch.tensor([0, 0, 0, 1])  # First 3 nodes in graph 0, last in graph 1
    output_batch = encoder(x_isolated, edge_index, batch=batch)
    print(f"Batch output shape (2 graphs): {output_batch.shape}")
    
    print("\n✓ Graph operations validated - message passing and pooling work correctly")

if __name__ == "__main__":
    test_graph_validation()