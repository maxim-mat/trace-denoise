import torch_geometric as pyg
from torch_geometric.nn import GCNConv, SAGEConv, GINConv
import torch.nn as nn
import torch


class GraphEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        nn1 = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 32))
        self.conv1 = GINConv(nn1)
        nn2 = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 16))
        self.conv2 = GINConv(nn2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x
