import torch
import torch.nn as nn
import torch_geometric

from modules.DoubleConv import DoubleConv
from modules.GraphEncoder import GraphEncoder


class GraphUp(nn.Module):
    def __init__(self, in_channels, out_channels, graph_data: torch_geometric.data.Data, node_embed_dim,
                 graph_hidden_dim, emb_dim=256):
        super().__init__()
        self.graph_data = graph_data

        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
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

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t).unsqueeze(2).repeat(1, 1, x.shape[2])
        graph_emb = self.graph_encoder(self.graph_data).unsqueeze(2).repeat(1, 1, x.shape[2])
        return x + emb + graph_emb
