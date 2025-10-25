import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from typing import Optional, Tuple


class GraphEncoder(nn.Module):
    def __init__(self,
                 node_dim: int = 64,               # Node feature dimension
                 hidden_dim: int = 128,             # Hidden dimension
                 output_dim: int = 256,             # Output dimension
                 n_layers: int = 3,                 # Number of GNN layers
                 architecture: str = 'gcn',         # 'gcn', 'gat', or 'graphsage'
                 n_heads: int = 8,                  # Number of attention heads (for GAT)
                 dropout: float = 0.1,              # Dropout rate
                 aggregation: str = 'mean',         # Node aggregation: 'mean', 'max', 'sum'
                 **kwargs):                         # REQUIRED: Accept extra kwargs
        super().__init__()
        
        # Log unknown parameters
        if kwargs:
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.architecture = architecture
        self.aggregation = aggregation
        
        # Input projection
        self.input_projection = nn.Linear(node_dim, hidden_dim)
        
        # Architecture-specific layers
        self.gnn_layers = nn.ModuleList()
        
        for i in range(n_layers):
            input_dim = hidden_dim if i > 0 else hidden_dim
            output_dim_layer = hidden_dim if i < n_layers - 1 else output_dim
            
            if architecture == 'gcn':
                self.gnn_layers.append(
                    GCNLayer(input_dim, output_dim_layer, dropout)
                )
            elif architecture == 'gat':
                # GAT with multiple heads
                self.gnn_layers.append(
                    GATLayer(input_dim, output_dim_layer // n_heads, n_heads, dropout)
                )
            elif architecture == 'graphsage':
                self.gnn_layers.append(
                    GraphSAGELayer(input_dim, output_dim_layer, dropout)
                )
            else:
                raise ValueError(f"Unknown architecture: {architecture}")
        
        # Node-level output projection
        self.node_projection = nn.Linear(
            output_dim if architecture != 'gat' else output_dim,
            output_dim
        )
        
        # Graph-level readout layers
        self.graph_projection = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self,
                node_features: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> dict:
        """
        Forward pass of the graph encoder.
        
        Args:
            node_features: Node features (num_nodes, node_dim)
            edge_index: Edge indices (2, num_edges)
            edge_attr: Optional edge attributes (num_edges, edge_dim)
            batch: Optional batch assignment vector (num_nodes,)
                   Maps each node to its respective graph in the batch
        
        Returns:
            Dictionary with keys:
                - 'node_embeddings': Node-level embeddings (num_nodes, output_dim)
                - 'graph_embedding': Graph-level embedding (batch_size, output_dim)
                - 'pooled_features': Intermediate pooled features
        """
        num_nodes = node_features.size(0)
        device = node_features.device
        
        # Input projection
        x = self.input_projection(node_features)
        
        # Apply GNN layers
        for i, layer in enumerate(self.gnn_layers):
            x = layer(x, edge_index, edge_attr)
            
            # Apply activation and dropout (except last layer)
            if i < self.n_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.1, training=self.training)
        
        # Node-level output
        node_embeddings = self.node_projection(x)
        node_embeddings = self.layer_norm(node_embeddings)
        
        # Graph-level pooling
        if batch is None:
            # Single graph case
            batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
        
        # Aggregate nodes by graph
        graph_embedding = self._graph_pool(node_embeddings, batch)
        
        # Graph-level projection
        graph_embedding = self.graph_projection(graph_embedding)
        
        return {
            'node_embeddings': node_embeddings,
            'graph_embedding': graph_embedding,
            'pooled_features': x  # Features before final projection
        }
    
    def _graph_pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Pool node features to graph-level representation."""
        batch_size = int(batch.max().item()) + 1
        pooled = torch.zeros(batch_size, x.size(-1), device=x.device)
        
        if self.aggregation == 'mean':
            # Mean pooling per graph
            for i in range(batch_size):
                mask = (batch == i)
                if mask.any():
                    pooled[i] = x[mask].mean(dim=0)
        
        elif self.aggregation == 'max':
            # Max pooling per graph
            for i in range(batch_size):
                mask = (batch == i)
                if mask.any():
                    pooled[i] = x[mask].max(dim=0)[0]
        
        elif self.aggregation == 'sum':
            # Sum pooling per graph
            for i in range(batch_size):
                mask = (batch == i)
                if mask.any():
                    pooled[i] = x[mask].sum(dim=0)
        
        return pooled


class GCNLayer(nn.Module):
    """Graph Convolutional Network layer."""
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """GCN forward pass with normalized adjacency."""
        # Add self-loops
        num_nodes = x.size(0)
        device = x.device
        
        # Create adjacency matrix
        row, col = edge_index
        edge_weight = torch.ones(edge_index.size(1), device=device)
        
        # Add self-loops
        self_loops = torch.arange(num_nodes, device=device)
        row = torch.cat([row, self_loops])
        col = torch.cat([col, self_loops])
        edge_weight = torch.cat([edge_weight, torch.ones(num_nodes, device=device)])
        
        # Compute normalization
        deg = torch.zeros(num_nodes, device=device)
        deg = deg.scatter_add(0, row, edge_weight)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # Normalize edge weights
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        
        # Message passing
        out = torch.zeros_like(x)
        for i in range(edge_weight.size(0)):
            out[row[i]] += edge_weight[i] * x[col[i]]
        
        # Linear transformation
        out = self.linear(out)
        out = self.dropout(out)
        
        return out


class GATLayer(nn.Module):
    """Graph Attention Network layer."""
    def __init__(self, in_dim: int, out_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        
        # Multi-head attention
        self.W = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        self.a = nn.Parameter(torch.zeros(2 * out_dim, n_heads))
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """GAT forward pass with multi-head attention."""
        # Linear transformation
        h = self.W(x).view(-1, self.n_heads, self.out_dim)
        
        # Attention mechanism
        row, col = edge_index
        
        # Compute attention coefficients
        h_i = h[row]  # (num_edges, n_heads, out_dim)
        h_j = h[col]  # (num_edges, n_heads, out_dim)
        
        # Concatenate and compute attention
        cat_ij = torch.cat([h_i, h_j], dim=-1)  # (num_edges, n_heads, 2*out_dim)
        e = torch.sum(cat_ij * self.a.unsqueeze(0), dim=-1)  # (num_edges, n_heads)
        e = self.leaky_relu(e)
        
        # Softmax per node
        num_nodes = x.size(0)
        alpha = torch.zeros(num_nodes, num_nodes, self.n_heads, device=x.device)
        alpha[row, col] = e
        
        # Row-wise softmax
        alpha = F.softmax(alpha, dim=1)
        alpha = self.dropout(alpha)
        
        # Apply attention
        out = torch.zeros_like(h)
        for i in range(num_nodes):
            neighbors = (row == i)
            if neighbors.any():
                neighbor_indices = col[neighbors]
                attention = alpha[i, neighbor_indices].unsqueeze(-1)
                out[i] = (attention * h[neighbor_indices]).sum(dim=0)
        
        # Concatenate or average heads
        out = out.view(-1, self.n_heads * self.out_dim)
        
        return out


class GraphSAGELayer(nn.Module):
    """GraphSAGE layer with neighbor aggregation."""
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """GraphSAGE forward pass with mean aggregation."""
        num_nodes = x.size(0)
        device = x.device
        row, col = edge_index
        
        # Aggregate neighbor features
        neighbor_feats = torch.zeros_like(x)
        neighbor_counts = torch.zeros(num_nodes, device=device)
        
        # Mean aggregation
        for i in range(edge_index.size(1)):
            neighbor_feats[row[i]] += x[col[i]]
            neighbor_counts[row[i]] += 1
        
        # Avoid division by zero
        neighbor_counts = neighbor_counts.clamp(min=1)
        neighbor_feats = neighbor_feats / neighbor_counts.unsqueeze(-1)
        
        # Concatenate self and neighbor features
        combined = torch.cat([x, neighbor_feats], dim=-1)
        
        # Transform and normalize
        out = self.linear(combined)
        out = F.normalize(out, p=2, dim=-1)
        out = self.dropout(out)
        
        return out