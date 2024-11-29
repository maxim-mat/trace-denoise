import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv
import torch.nn as nn


class GraphNodeEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, heads=8, dropout_prob=0.1):
        super().__init__()
        # Define GNN layers for each edge type using HeteroConv
        self.conv1 = HeteroConv({
            ('transition', 'transition_to_place', 'place'): GATConv(
                embedding_dim, hidden_dim, heads=heads, dropout=dropout_prob, add_self_loops=False),
            ('place', 'place_to_transition', 'transition'): GATConv(
                embedding_dim, hidden_dim, heads=heads, dropout=dropout_prob, add_self_loops=False)
        }, aggr='sum')

        self.conv2 = HeteroConv({
            ('transition', 'transition_to_place', 'place'): GATConv(
                hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout_prob, add_self_loops=False),
            ('place', 'place_to_transition', 'transition'): GATConv(
                hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout_prob, add_self_loops=False)
        }, aggr='sum')

        # Define output layers for each node type
        self.output_layer = nn.ModuleDict({
            'transition': nn.Linear(hidden_dim, output_dim),
            'place': nn.Linear(hidden_dim, output_dim)
        })

    def forward(self, x_dict, edge_index_dict):
        # x_dict: node features for each node type
        # edge_index_dict: edge indices for each edge type

        # First convolution layer with activation
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {node_type: F.relu(x) for node_type, x in x_dict.items()}

        # Second convolution layer with activation
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {node_type: F.relu(x) for node_type, x in x_dict.items()}

        # Apply output layers to each node type
        out_dict = {node_type: self.output_layer[node_type](x) for node_type, x in x_dict.items()}

        return out_dict
