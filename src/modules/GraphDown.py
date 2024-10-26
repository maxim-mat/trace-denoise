import torch.nn as nn
import torch_geometric

from src.modules.DoubleConv import DoubleConv
from src.modules.GraphEncoder import GraphEncoder


class GraphDown(nn.Module):
    def __init__(self, in_channels, out_channels, graph_data: torch_geometric.data.Data, node_embed_dim,
                 graph_hidden_dim, emb_dim=256):
        super().__init__()
        self.graph_data = graph_data

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

        self.graph_encoder = GraphEncoder(num_nodes=graph_data.num_nodes, embedding_dim=node_embed_dim,
                                          hidden_dim=graph_hidden_dim, output_dim=out_channels)

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t).unsqueeze(2).repeat(1, 1, x.shape[2])
        graph_emb = self.graph_encoder(self.graph_data)
        return x + emb + graph_emb
