import torch.nn as nn
from torch_geometric.nn import GINEConv

class EdgeGINEConv(nn.Module):
    """
    GINEConv wrapper: local message passing that uses edge_attr.
    """

    def __init__(self, in_dim, out_dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = out_dim

        mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.conv = GINEConv(mlp)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        # GINEConv assumes edge_attr is not None (can be zeros if needed)
        x = self.conv(x, edge_index, edge_attr)
        x = self.dropout(x)
        return x
