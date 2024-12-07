import torch_geometric as pyg
from torch_geometric.nn import GINConv, GATConv
import torch.nn as nn
import torch.nn.functional as F
import torch


class GraphNodeEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, num_nodes, heads=8, dropout_prob=0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.conv1 = GATConv(embedding_dim, hidden_dim, heads=heads, dropout=dropout_prob)
        self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False, dropout=dropout_prob)
        # self.conv2 = GINConv(
        #     nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        # self.conv3 = GINConv(
        #     nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        # self.conv4 = GINConv(
        #     nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        # self.conv5 = GINConv(
        #     nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.ReLU(), nn.Linear(output_dim, output_dim)))
        self.output_layer = nn.Linear(output_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.embedding(x)
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.output_layer(x)
        return x
