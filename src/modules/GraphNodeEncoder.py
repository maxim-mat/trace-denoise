import torch_geometric as pyg
from torch_geometric.nn import GINConv
import torch.nn as nn
import torch.nn.functional as F
import torch


class GraphNodeEncoder(nn.Module):
    def __init__(self, num_nodes, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.conv1 = GINConv(
            nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv2 = GINConv(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv3 = GINConv(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv4 = GINConv(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv5 = GINConv(
            nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.ReLU(), nn.Linear(output_dim, output_dim)))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.embedding(x).squeeze(1)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        x = self.conv5(x, edge_index)
        return x
