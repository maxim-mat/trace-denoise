import torch
import torch.nn as nn
import torch_geometric

from modules.DoubleConv import DoubleConv
from modules.SelfAttention import SelfAttention
from modules.GraphUp import GraphUp
from modules.GraphDown import GraphDown
from modules.Down import Down
from modules.Up import Up
from modules.GraphEncoder import GraphEncoder


class ConditionalUnetGraphDenoiser(nn.Module):
    def __init__(self, in_ch, out_ch, max_input_dim, num_nodes, embedding_dim, hidden_dim, time_dim=128, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.max_input_dim = max_input_dim

        # generated output u-net layers
        self.inc = DoubleConv(in_ch, 64)
        self.down1 = Down(64, 128, emb_dim=time_dim)
        self.sa1 = SelfAttention(128, max_input_dim // 2)
        self.down2 = Down(128, 256, emb_dim=time_dim)
        self.sa2 = SelfAttention(256, max_input_dim // 4)
        self.down3 = Down(256, 256, emb_dim=time_dim)
        self.sa3 = SelfAttention(256, max_input_dim // 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128, emb_dim=time_dim)
        self.sa4 = SelfAttention(128, max_input_dim // 4)
        self.up2 = Up(256, 64, emb_dim=time_dim)
        self.sa5 = SelfAttention(64, max_input_dim // 2)
        self.up3 = Up(128, 64, emb_dim=time_dim)
        self.sa6 = SelfAttention(64, max_input_dim)

        # guidance sk trace u-net layers
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

        self.genc1 = GraphEncoder(num_nodes, embedding_dim, hidden_dim, output_dim=64)
        self.genc2 = GraphEncoder(num_nodes, embedding_dim, hidden_dim, output_dim=128)
        self.genc3 = GraphEncoder(num_nodes, embedding_dim, hidden_dim, output_dim=256)

        self.genc1_cond = GraphEncoder(num_nodes, embedding_dim, hidden_dim, output_dim=64)
        self.genc2_cond = GraphEncoder(num_nodes, embedding_dim, hidden_dim, output_dim=128)
        self.genc3_cond = GraphEncoder(num_nodes, embedding_dim, hidden_dim, output_dim=256)

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

    def _forward_uncond_no_graph(self, x, t):
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
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        x = self.outc(x)
        return x

    def _forward_cond_no_graph(self, x, y, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        y1 = self.inc_cond(y)
        x1 = self.inc(x)
        x2 = self.down1(x1 + y1, t)
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

    def _forward_uncond_graph(self, x, g, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        batch_size = x.size(0)

        x1 = self.inc(x)
        g1 = self.genc1(g).view(1, -1, 1).repeat(batch_size, 1, x1.size(2))
        x2 = self.down1(x1 + g1, t)
        x2 = self.sa1(x2)
        g2 = self.genc2(g).view(1, -1, 1).repeat(batch_size, 1, x2.size(2))
        x3 = self.down2(x2 + g2, t)
        x3 = self.sa2(x3)
        g3 = self.genc3(g).view(1, -1, 1).repeat(batch_size, 1, x3.size(2))
        x4 = self.down3(x3 + g3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        x = self.outc(x)

        return x

    def _forward_cond_graph(self, x, y, g, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        batch_size = x.size(0)

        y1 = self.inc_cond(y)
        x1 = self.inc(x)
        g1 = self.genc1(g).view(1, -1, 1).repeat(batch_size, 1, x1.size(2))
        g1_cond = self.genc1_cond(g).view(1, -1, 1).repeat(batch_size, 1, y1.size(2))
        x2 = self.down1(x1 + y1 + g1, t)
        y2 = self.down1_cond(x1 + y1 + g1_cond, t)
        y2 = self.sa1_cond(y2)
        x2 = self.sa1(x2)
        g2 = self.genc2(g).view(1, -1, 1).repeat(batch_size, 1, x2.size(2))
        g2_cond = self.genc2_cond(g).view(1, -1, 1).repeat(batch_size, 1, y2.size(2))
        x3 = self.down2(x2 + y2 + g2, t)
        y3 = self.down2_cond(x2 + y2 + g2_cond, t)
        y3 = self.sa2_cond(y3)
        x3 = self.sa2(x3)
        g3 = self.genc3(g).view(1, -1, 1).repeat(batch_size, 1, x3.size(2))
        g3_cond = self.genc3_cond(g).view(1, -1, 1).repeat(batch_size, 1, y3.size(2))
        x4 = self.down3(x3 + y3 + g3, t)
        x4 = self.sa3(x4)
        y4 = self.down3_cond(x3 + y3 + g3_cond, t)
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

    def forward(self, x, t, y=None, g=None):
        if g is not None:
            if y is not None:
                return self._forward_cond_graph(x, y, g, t)
            else:
                return self._forward_uncond_graph(x, g, t)
        else:
            if y is not None:
                return self._forward_cond_no_graph(x, y, t)
            else:
                return self._forward_uncond_no_graph(x, t)
