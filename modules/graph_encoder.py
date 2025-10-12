import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, aggr='mean'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.aggr = aggr
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            
    def forward(self, x, edge_index, edge_weight=None):
        # x: (num_nodes, in_features)
        # edge_index: (2, num_edges)
        # edge_weight: (num_edges,) optional
        
        # Linear transformation
        x = torch.matmul(x, self.weight)
        
        # Message passing
        row, col = edge_index
        
        if edge_weight is not None:
            # Weighted message passing
            messages = x[col] * edge_weight.unsqueeze(-1)
        else:
            messages = x[col]
        
        # Aggregate messages
        out = torch.zeros_like(x)
        if self.aggr == 'mean':
            out.index_add_(0, row, messages)
            # Count number of neighbors for each node
            degree = torch.zeros(x.size(0), device=x.device)
            degree.index_add_(0, row, torch.ones(row.size(0), device=x.device))
            degree = degree.clamp(min=1)
            out = out / degree.unsqueeze(-1)
        elif self.aggr == 'sum':
            out.index_add_(0, row, messages)
        elif self.aggr == 'max':
            out, _ = torch_scatter.scatter_max(messages, row, dim=0, dim_size=x.size(0))
        
        if self.bias is not None:
            out = out + self.bias
            
        return out


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, n_heads=8, dropout=0.1):
        super().__init__()
        assert out_features % n_heads == 0
        
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.d_k = out_features // n_heads
        
        self.W = nn.Linear(in_features, out_features)
        self.a = nn.Parameter(torch.FloatTensor(n_heads, 2 * self.d_k))
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)
        
    def forward(self, x, edge_index):
        # x: (num_nodes, in_features)
        # edge_index: (2, num_edges)
        
        # Linear transformation
        h = self.W(x)  # (num_nodes, out_features)
        h = h.view(-1, self.n_heads, self.d_k)  # (num_nodes, n_heads, d_k)
        
        # Attention mechanism
        row, col = edge_index
        
        # Get source and target node features
        h_i = h[row]  # (num_edges, n_heads, d_k)
        h_j = h[col]  # (num_edges, n_heads, d_k)
        
        # Concatenate source and target features
        h_cat = torch.cat([h_i, h_j], dim=-1)  # (num_edges, n_heads, 2*d_k)
        
        # Compute attention scores
        e = torch.sum(h_cat * self.a, dim=-1)  # (num_edges, n_heads)
        e = F.leaky_relu(e, 0.2)
        
        # Softmax over neighbors
        # This is simplified - proper implementation would use segment_softmax
        e = F.softmax(e, dim=0)
        e = self.dropout(e)
        
        # Apply attention
        out = torch.zeros_like(h)
        for head in range(self.n_heads):
            messages = h_j[:, head] * e[:, head].unsqueeze(-1)
            out[:, head].index_add_(0, row, messages)
            
        out = out.view(-1, self.out_features)
        
        return out


class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=128, n_layers=3,
                 layer_type='gcn', n_heads=8, dropout=0.1, pooling='mean'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.layer_type = layer_type
        self.pooling = pooling
        
        # Build layers
        self.layers = nn.ModuleList()
        
        for i in range(n_layers):
            if i == 0:
                in_dim = input_dim
            else:
                in_dim = hidden_dim
                
            if i == n_layers - 1:
                out_dim = output_dim
            else:
                out_dim = hidden_dim
                
            if layer_type == 'gcn':
                layer = GraphConvLayer(in_dim, out_dim)
            elif layer_type == 'gat':
                layer = GraphAttentionLayer(in_dim, out_dim, n_heads)
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
                
            self.layers.append(layer)
            
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
        # Pooling layers
        if pooling == 'attention':
            self.pool_attention = nn.Linear(output_dim, 1)
            
    def forward(self, x, edge_index, edge_attr=None, batch=None, return_node_features=False):
        """
        x: Node features (num_nodes, input_dim)
        edge_index: COO edge indices (2, num_edges)
        edge_attr: Edge features (num_edges, edge_dim) - optional
        batch: Batch assignment vector (num_nodes,) - optional
        """
        
        # Apply graph convolution layers
        for i, layer in enumerate(self.layers):
            if self.layer_type == 'gcn' and edge_attr is not None and i == 0:
                # Use edge weights if available
                if edge_attr.dim() > 1:
                    edge_weight = edge_attr.mean(dim=-1)  # Simple reduction
                else:
                    edge_weight = edge_attr
                x = layer(x, edge_index, edge_weight)
            else:
                x = layer(x, edge_index)
                
            if i < self.n_layers - 1:
                x = self.activation(x)
                x = self.dropout(x)
        
        # Store node features
        node_features = x
        
        # Graph-level pooling
        if batch is None:
            # Single graph
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        if self.pooling == 'mean':
            out = self.global_mean_pool(x, batch)
        elif self.pooling == 'max':
            out = self.global_max_pool(x, batch)
        elif self.pooling == 'sum':
            out = self.global_sum_pool(x, batch)
        elif self.pooling == 'attention':
            out = self.global_attention_pool(x, batch)
        
        if return_node_features:
            return {
                'graph_embedding': out,
                'node_features': node_features,
                'batch': batch
            }
        else:
            return out
            
    def global_mean_pool(self, x, batch):
        """Global mean pooling over nodes in each graph"""
        num_graphs = batch.max().item() + 1
        out = torch.zeros(num_graphs, x.size(1), device=x.device)
        count = torch.zeros(num_graphs, device=x.device)
        
        for i in range(num_graphs):
            mask = batch == i
            if mask.any():
                out[i] = x[mask].mean(dim=0)
                
        return out
        
    def global_max_pool(self, x, batch):
        """Global max pooling over nodes in each graph"""
        num_graphs = batch.max().item() + 1
        out = torch.zeros(num_graphs, x.size(1), device=x.device)
        
        for i in range(num_graphs):
            mask = batch == i
            if mask.any():
                out[i], _ = x[mask].max(dim=0)
                
        return out
        
    def global_sum_pool(self, x, batch):
        """Global sum pooling over nodes in each graph"""
        num_graphs = batch.max().item() + 1
        out = torch.zeros(num_graphs, x.size(1), device=x.device)
        
        for i in range(num_graphs):
            mask = batch == i
            if mask.any():
                out[i] = x[mask].sum(dim=0)
                
        return out
        
    def global_attention_pool(self, x, batch):
        """Global attention pooling over nodes in each graph"""
        scores = self.pool_attention(x).squeeze(-1)
        
        num_graphs = batch.max().item() + 1
        out = torch.zeros(num_graphs, x.size(1), device=x.device)
        
        for i in range(num_graphs):
            mask = batch == i
            if mask.any():
                graph_scores = scores[mask]
                graph_scores = F.softmax(graph_scores, dim=0)
                out[i] = (x[mask] * graph_scores.unsqueeze(-1)).sum(dim=0)
                
        return out


class EdgeConvLayer(nn.Module):
    """EdgeConv layer for dynamic graph construction"""
    def __init__(self, in_features, out_features, k=20):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Linear(2 * in_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )
        
    def forward(self, x):
        # x: (batch_size, num_points, in_features)
        batch_size, num_points, num_features = x.size()
        
        # Compute pairwise distances
        x_expanded = x.unsqueeze(2).expand(-1, -1, num_points, -1)
        x_tiled = x.unsqueeze(1).expand(-1, num_points, -1, -1)
        dists = torch.sum((x_expanded - x_tiled) ** 2, dim=-1)
        
        # Get k-nearest neighbors
        _, indices = torch.topk(dists, self.k, dim=-1, largest=False)
        
        # Gather neighbor features
        x_neighbors = torch.gather(x_tiled, 2, indices.unsqueeze(-1).expand(-1, -1, -1, num_features))
        
        # Edge features
        x_concat = torch.cat([x.unsqueeze(2).expand_as(x_neighbors), x_neighbors], dim=-1)
        
        # Apply convolution
        out = self.conv(x_concat)
        
        # Max pooling
        out, _ = torch.max(out, dim=2)
        
        return out