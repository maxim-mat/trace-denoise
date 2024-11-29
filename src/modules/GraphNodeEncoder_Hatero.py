import torch_geometric as pyg
from torch_geometric.nn import GINConv
import torch.nn as nn
import torch.nn.functional as F
import torch

# Assuming HateroData is defined elsewhere in the project or provided as a custom module.
from hatero_data import HateroData


class HaterogenicGraphNodeEncoder(nn.Module):
    def __init__(self, num_nodes, embedding_dim, hidden_dim, output_dim, dropout_prob=0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)

        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout_prob),
                nn.Linear(hidden_dim, hidden_dim)
            )
        )

        self.conv2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout_prob),
                nn.Linear(hidden_dim, hidden_dim)
            )
        )

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        # Ensure the input is an instance of HateroData
        if not isinstance(data, HateroData):
            raise TypeError("Expected input data to be of type HateroData.")

        # Extract features and edge indices from HateroData
        x, edge_index = data.node_features, data.edge_index

        # Pass through embedding layer
        x = self.embedding(x)

        # Apply GIN convolution layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Output layer for final node representations
        x = self.output_layer(x)

        return x
