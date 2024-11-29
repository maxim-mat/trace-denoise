import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import to_hetero

from src.modules.DoubleConv import DoubleConv
from src.modules.Up import Up
from src.modules.Down import Down
from src.modules.SelfAttention import SelfAttention
from src.modules.CrossAttention import CrossAttention
from src.modules.GraphNodeEncoder import GraphNodeEncoder


class ConditionalUnetAttentionGraphDenoiser(nn.Module):
    def __init__(self, in_ch, out_ch, max_input_dim, graph_data: torch_geometric.data.Data, node_embed_dim,
                 graph_hidden_dim, metadata=None, time_dim=128, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.max_input_dim = max_input_dim
        self.graph_data = graph_data
        self.metadata = metadata

        self.graph_node_encoder = GraphNodeEncoder(embedding_dim=node_embed_dim,
                                                   hidden_dim=graph_hidden_dim, output_dim=graph_hidden_dim)
        # if self.metadata is not None:
        #     self.graph_node_encoder = to_hetero(self.graph_node_encoder, self.metadata)

        self.inc = DoubleConv(in_ch, 64)
        self.down1 = Down(64, 128, emb_dim=time_dim)
        self.ca1 = CrossAttention(128, max_input_dim // 2)
        self.sa1 = SelfAttention(128, max_input_dim // 2)
        self.down2 = Down(128, 256, emb_dim=time_dim)
        self.ca2 = nn.MultiheadAttention(256, num_heads=2)
        self.sa2 = SelfAttention(256, max_input_dim // 4)
        self.down3 = Down(256, 256, emb_dim=time_dim)
        self.ca3 = nn.MultiheadAttention(256, num_heads=2)
        self.sa3 = SelfAttention(256, max_input_dim // 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128, emb_dim=time_dim)
        self.ca4 = nn.MultiheadAttention(128, num_heads=2)
        self.sa4 = SelfAttention(128, max_input_dim // 4)
        self.up2 = Up(256, 64, emb_dim=time_dim)
        self.ca5 = nn.MultiheadAttention(64, num_heads=2)
        self.sa5 = SelfAttention(64, max_input_dim // 2)
        self.up3 = Up(128, 64, emb_dim=time_dim)
        self.ca6 = nn.MultiheadAttention(64, num_heads=2)
        self.sa6 = SelfAttention(64, max_input_dim)

        self.inc_cond = DoubleConv(in_ch, 64)
        self.down1_cond = Down(64, 128, emb_dim=time_dim)
        self.sa1_cond = SelfAttention(128, max_input_dim // 2)
        self.down2_cond = Down(128, 256, emb_dim=time_dim)
        self.sa2_cond = SelfAttention(256, max_input_dim // 4)
        self.down3_cond = Down(256, 256, emb_dim=time_dim)
        self.sa3_cond = SelfAttention(256, max_input_dim // 8)

        self.bot1_cond = DoubleConv(256, 512)
        self.bot2_cond = DoubleConv(512, 512)
        self.bot3_cond = DoubleConv(512, 256)

        self.up1_cond = Up(512, 128, emb_dim=time_dim)
        self.sa4_cond = SelfAttention(128, max_input_dim // 4)
        self.up2_cond = Up(256, 64, emb_dim=time_dim)
        self.sa5_cond = SelfAttention(64, max_input_dim // 2)
        self.up3_cond = Up(128, 64, emb_dim=time_dim)
        self.sa6_cond = SelfAttention(64, max_input_dim)
        self.outc = nn.Conv1d(64, out_ch, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def _forward_uncond(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        del x4
        del x3
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        del x2
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        del x1
        x = self.sa6(x)
        x = self.outc(x)
        return x

    def _forward_cond(self, x, y, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        graph_node_embedding = self.graph_node_encoder(self.graph_data.x_dict, self.graph_data.edge_index_dict)
        unrefined_traces = torch.argmax(y, dim=1)
        unrefined_nodes = graph_node_embedding[unrefined_traces]

        y1 = self.inc_cond(y)
        x1 = self.inc(x)
        ca1 = self.ca1(query=x1 + y1, key=unrefined_nodes, value=unrefined_nodes)
        x2 = self.down1(x1 + y1, t) * ca1

        y2 = self.down1_cond(x1 + y1, t)
        y2 = self.sa1_cond(y2)
        x2 = self.sa1(x2)
        x3 = self.down2(x2 + y2, t)
        x3 = self.sa2(x3)
        y3 = self.down2_cond(x2 + y2, t)
        y3 = self.sa2_cond(y3)
        x4 = self.down3(x3 + y3, t)
        x4 = self.sa3(x4)
        y4 = self.down3_cond(x3 + y3, t)
        y4 = self.sa3_cond(y4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        y4 = self.bot1_cond(y4)
        y4 = self.bot2_cond(y4)
        y4 = self.bot3_cond(y4)

        y = self.up1(x4 + y4, y3, t)
        x = self.up1(x4 + y4, x3, t)
        x = self.sa4(x)
        y = self.sa4(y)
        x_next = self.up2(x + y, x2, t)
        y_next = self.up2(x + y, y2, t)
        y_next = self.sa5(y_next)
        x_next = self.sa5(x_next)
        x = self.up3(x_next + y_next, x1, t)
        y = self.up3(x_next + y_next, y1, t)
        y = self.sa6(y)
        x = self.sa6(x)
        x = self.outc(x + y)

        return x

    def forward(self, x, t, y=None):
        if y is not None:
            return self._forward_cond(x, y, t)
        else:
            return self._forward_uncond(x, t)
