import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modules.DoubleConv import DoubleConv
from src.modules.DoubleConv2d import DoubleConv2d
from src.modules.Up import Up
from src.modules.Up2d import Up2d
from src.modules.Down import Down
from src.modules.Down2d import Down2d
from src.modules.SelfAttention import SelfAttention
from src.modules.CrossAttention import CrossAttention


class ConditionalUnetMatrixDenoiser(nn.Module):
    def __init__(self, in_ch, out_ch, max_input_dim, transition_dim, transition_matrix=None, time_dim=128,
                 device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.max_input_dim = max_input_dim
        self.transition_dim = transition_dim
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.transition_matrix = torch.randn(
            (transition_dim, transition_dim)
        ) if transition_matrix is None else transition_matrix
        self.transition_matrix = self.transition_matrix.unsqeeze(0).unsqueeze(0)

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

        self.inc_mat = DoubleConv2d(1, 64)
        self.down1_mat = Down2d(64, 128, emb_dim=time_dim)
        self.down2_mat = Down2d(128, 256, emb_dim=time_dim)
        self.down3_mat = Down2d(256, 256, emb_dim=time_dim)
        self.bot1_mat = DoubleConv2d(256, 512)
        self.bot2_mat = DoubleConv2d(512, 512)
        self.bot3_mat = DoubleConv2d(512, 256)
        self.up1_mat = Up2d(512, 128, emb_dim=time_dim)
        self.up2_mat = Up2d(256, 64, emb_dim=time_dim)
        self.up3_mat = Up2d(128, 64, emb_dim=time_dim)
        self.up_out_mat = Up2d(64, 1, emb_dim=time_dim)

        self.ca1 = CrossAttention(128, max_input_dim // 2)
        self.ca2 = CrossAttention(256, max_input_dim // 4)
        self.ca3 = CrossAttention(256, max_input_dim // 8)
        self.ca4 = CrossAttention(128, max_input_dim // 4)
        self.ca5 = CrossAttention(64, max_input_dim // 2)
        self.ca6 = CrossAttention(64, max_input_dim)

        # self.outc_mat_linear = nn.Linear(64 * max_input_dim, 64)
        # self.outc_mat = nn.Linear(64, self.transition_dim * self.transition_dim)
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
        x = self.sa4(x)
        m = self.up1_mat(x4, x3, t)
        m = self.sa4_mat(m)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        m = self.up2_mat(m, x2, t)
        m = self.sa5_mat(m)
        x = self.up3(x, x1, t)
        # x = self.sa6(x)
        m = self.up3_mat(m, x1, t)
        # m = self.sa6_mat(m)
        m = m.view(m.size(0), -1)
        m = F.relu(self.outc_mat_linear(m))
        m = self.outc_mat(m)
        x = self.outc(x)
        return x, m

    def _forward_cond(self, x, y, m, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        y1 = self.inc_cond(y)
        x1 = self.inc(x)
        m1 = self.inc_mat(m)
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
        m = self.up1_mat(x4 + y4, x3 + y3, t)
        m = self.sa4_mat(m)
        x_next = self.up2(x + y, x2, t)
        y_next = self.up2(x + y, y2, t)
        y_next = self.sa5(y_next)
        x_next = self.sa5(x_next)
        m = self.up2_mat(m, x2 + y2, t)
        m = self.sa5_mat(m)
        x = self.up3(x_next + y_next, x1, t)
        y = self.up3(x_next + y_next, y1, t)
        # y = self.sa6(y)
        # x = self.sa6(x)
        m = self.up3_mat(m, x1 + y1, t)
        # m = self.sa6_mat(m)
        m = m.view(m.size(0), -1)
        m = F.relu(self.outc_mat_linear(m))
        m = self.outc_mat(m)
        x = self.outc(x + y)

        return x, m

    def forward(self, x, t, y=None):
        if y is not None:
            return self._forward_cond(x, y, t)
        else:
            return self._forward_uncond(x, t)
