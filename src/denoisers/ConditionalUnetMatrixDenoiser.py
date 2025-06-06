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
        self.alpha = nn.Parameter(torch.tensor(0.0)).to(device)
        self.transition_matrix = transition_matrix if transition_matrix is not None \
            else nn.Parameter(torch.randn(1, in_ch + 1, self.transition_dim, self.transition_dim).to(device))

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

        self.inc_mat = DoubleConv2d(in_ch + 1, 64)
        self.down1_mat = Down2d(64, 128, emb_dim=time_dim, down_rate=8)
        self.sa1_mat = SelfAttention(128, (self.transition_dim // 8) ** 2)
        self.down2_mat = Down2d(128, 256, emb_dim=time_dim)
        self.sa2_mat = SelfAttention(256, (self.transition_dim // 16) ** 2)
        self.down3_mat = Down2d(256, 256, emb_dim=time_dim)
        self.sa3_mat = SelfAttention(256, (self.transition_dim // 32) ** 2)
        self.bot1_mat = DoubleConv2d(256, 512)
        self.bot2_mat = DoubleConv2d(512, 512)
        self.bot3_mat = DoubleConv2d(512, 256)
        self.up1_mat = Up2d(512, 128, emb_dim=time_dim)
        self.sa4_mat = SelfAttention(128, (self.transition_dim // 16) ** 2)
        self.up2_mat = Up2d(256, 64, emb_dim=time_dim)
        self.sa5_mat = SelfAttention(64, (self.transition_dim // 8) ** 2)
        self.up3_mat = Up2d(128, 64, emb_dim=time_dim, scale_factor=8)
        self.sa6_mat = SelfAttention(64, self.transition_dim ** 2)

        self.casm1 = CrossAttention(128, max_input_dim // 2)
        self.casm2 = CrossAttention(256, max_input_dim // 4)
        self.casm3 = CrossAttention(256, max_input_dim // 8)
        self.casm4 = CrossAttention(128, max_input_dim // 4)
        self.casm5 = CrossAttention(64, max_input_dim // 2)
        self.casm6 = CrossAttention(64, max_input_dim)

        self.casm1_cond = CrossAttention(128, max_input_dim // 2)
        self.casm2_cond = CrossAttention(256, max_input_dim // 4)
        self.casm3_cond = CrossAttention(256, max_input_dim // 8)
        self.casm4_cond = CrossAttention(128, max_input_dim // 4)
        self.casm5_cond = CrossAttention(64, max_input_dim // 2)
        self.casm6_cond = CrossAttention(64, max_input_dim)

        self.cams1 = CrossAttention(128, (self.transition_dim // 8) ** 2)
        self.cams2 = CrossAttention(256, (self.transition_dim // 16) ** 2)
        self.cams3 = CrossAttention(256, (self.transition_dim // 32) ** 2)
        self.cams4 = CrossAttention(128, (self.transition_dim // 16) ** 2)
        self.cams5 = CrossAttention(64, (self.transition_dim // 8) ** 2)
        self.cams6 = CrossAttention(64, self.transition_dim ** 2)

        # self.outc_mat_linear = nn.Linear(64 * max_input_dim, 64)
        # self.outc_mat = nn.Linear(64, self.transition_dim * self.transition_dim)
        self.outc_mat = nn.Conv2d(64, out_ch + 1, kernel_size=1)
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

    def _forward_uncond(self, x, m, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        batch_dim = x.shape[0]

        x1 = self.inc(x)
        m1 = self.inc_mat(m)

        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        m2 = self.down1_mat(m1)
        m2 = m2.view(1, m2.shape[1], -1)
        m2 = self.sa1_mat(m2)
        m2_ca = self.cams1(m2, x2, x2).view(m2.size(0), m2.size(1),
                                            self.transition_dim // 8, self.transition_dim // 8)
        x2_ca = self.casm1(x2, m2, m2)

        x3 = self.down2(x2_ca, t)
        x3 = self.sa2(x3)
        m3 = self.down2_mat(m2_ca)
        m3 = m3.view(1, m3.shape[1], -1)
        m3 = self.sa2_mat(m3)
        m3_ca = self.cams2(m3, x3, x3).view(m3.size(0), m3.size(1),
                                            self.transition_dim // 16, self.transition_dim // 16)
        x3_ca = self.casm2(x3, m3, m3)

        x4 = self.down3(x3_ca, t)
        x4 = self.sa3(x4)
        m4 = self.down2_mat(m3_ca)
        m4 = m4.view(1, m4.shape[1], -1)
        m4 = self.sa3_mat(m4)
        m4_ca = self.cams3(m4, x4, x4).view(m4.size(0), m4.size(1),
                                            self.transition_dim // 32, self.transition_dim // 32)
        x4_ca = self.casm3(x4, m4, m4)

        x4 = self.bot1(x4_ca)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        m4 = self.bot1_mat(m4_ca)
        m4 = self.bot2_mat(m4)
        m4 = self.bot3_mat(m4)

        x = self.up1(x4, x3_ca, t)
        x = self.sa4(x)
        m = self.up1_mat(m4, m3_ca, t)
        m = m.view(1, m.shape[1], -1)
        m = self.sa4_mat(m)
        m_ca = self.cams4(m, x, x).view(m.size(0), m.size(1),
                                        self.transition_dim // 16, self.transition_dim // 16)
        x_ca = self.casm4(x, m, m)

        x_next = self.up2(x_ca, x2_ca, t)
        x_next = self.sa5(x_next)
        m_next = self.up2_mat(m_ca, m2_ca, t)
        m_next = m_next.view(1, m_next.shape[1], -1)
        m_next = self.sa5_mat(m_next)
        m_next_ca = self.cams4(m_next, x_next, x_next).view(m.size(0), m.size(1),
                                                            self.transition_dim // 8, self.transition_dim // 8)
        x_next_ca = self.casm4(x_next, m_next, m_next)

        x = self.up3(x_next_ca, x1, t)
        x = self.sa6(x)
        m = self.up3_mat(m_next_ca, m1, t)

        m = self.outc_mat(m)
        x = self.outc(x)

        return x, m

    def _forward_cond(self, x, y, m, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        batch_dim = x.shape[0]

        y1 = self.inc_cond(y)
        x1 = self.inc(x)
        m1 = self.inc_mat(m)

        x2 = self.down1(x1 + y1, t)
        x2 = self.sa1(x2)
        y2 = self.down1_cond(x1 + y1, t)
        y2 = self.sa1_cond(y2)
        m2 = self.down1_mat(m1, t)
        m2 = m2.view(batch_dim, m2.shape[1], -1)  # down handles extending to batch dim
        m2 = self.sa1_mat(m2)
        m2_ca = self.cams1(m2, x2 + y2, x2 + y2).view(m2.size(0), m2.size(1),
                                                      self.transition_dim // 8, self.transition_dim // 8)
        x2_ca = self.casm1(x2 + y2, m2, m2)
        y2_ca = self.casm1_cond(x2 + y2, m2, m2)

        x3 = self.down2(x2_ca + y2_ca, t)
        x3 = self.sa2(x3)
        y3 = self.down2_cond(x2_ca + y2_ca, t)
        y3 = self.sa2_cond(y3)
        m3 = self.down2_mat(m2_ca, t)
        m3 = m3.view(batch_dim, m3.shape[1], -1)
        m3 = self.sa2_mat(m3)
        m3_ca = self.cams2(m3, x3 + y3, x3 + y3).view(m3.size(0), m3.size(1),
                                                      self.transition_dim // 16, self.transition_dim // 16)
        x3_ca = self.casm2(x3 + y3, m3, m3)
        y3_ca = self.casm2_cond(x3 + y3, m3, m3)

        x4 = self.down3(x3_ca + y3_ca, t)
        x4 = self.sa3(x4)
        y4 = self.down3_cond(x3_ca + y3_ca, t)
        y4 = self.sa3_cond(y4)
        m4 = self.down3_mat(m3_ca, t)
        m4 = m4.view(batch_dim, m4.shape[1], -1)
        m4 = self.sa3_mat(m4)
        m4_ca = self.cams3(m4, x4 + y4, x4 + y4).view(m4.size(0), m4.size(1),
                                                      self.transition_dim // 32, self.transition_dim // 32)
        x4_ca = self.casm3(x4 + y4, m4, m4)
        y4_ca = self.casm3_cond(x4 + y4, m4, m4)

        x4 = self.bot1(x4_ca)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        y4 = self.bot1_cond(y4_ca)
        y4 = self.bot2_cond(y4)
        y4 = self.bot3_cond(y4)
        m4 = self.bot1_mat(m4_ca)
        m4 = self.bot2_mat(m4)
        m4 = self.bot3_mat(m4)

        x = self.up1(x4 + y4, x3_ca, t)
        x = self.sa4(x)
        y = self.up1(x4 + y4, y3_ca, t)
        y = self.sa4_cond(y)
        m = self.up1_mat(m4, m3_ca, t)
        m = m.view(batch_dim, m.shape[1], -1)
        m = self.sa4_mat(m)
        m_ca = self.cams4(m, x + y, x + y).view(m.size(0), m.size(1),
                                                self.transition_dim // 16, self.transition_dim // 16)
        x_ca = self.casm4(x + y, m, m)
        y_ca = self.casm4_cond(x + y, m, m)

        x_next = self.up2(x_ca + y_ca, x2_ca, t)
        x_next = self.sa5(x_next)
        y_next = self.up2_cond(x_ca + y_ca, y2_ca, t)
        y_next = self.sa5_cond(y_next)
        m_next = self.up2_mat(m_ca, m2_ca, t)
        m_next = m_next.view(batch_dim, m_next.shape[1], -1)
        m_next = self.sa5_mat(m_next)
        m_next_ca = self.cams5(m_next, x_next + y_next, x_next + y_next).view(m_next.size(0), m_next.size(1),
                                                                              self.transition_dim // 8,
                                                                              self.transition_dim // 8)
        x_next_ca = self.casm5(x_next + y_next, m_next, m_next)
        y_next_ca = self.casm5_cond(x_next + y_next, m_next, m_next)

        x = self.up3(x_next_ca + y_next_ca, x1, t)
        x = self.sa6(x)
        y = self.up3(x_next_ca + y_next_ca, y1, t)
        y = self.sa6(y)
        m = self.up3_mat(m_next_ca, m1.repeat(5, 1, 1, 1), t)

        m = self.outc_mat(m)
        x = self.outc(x + y)

        return x, m

    def forward(self, x, t, y=None):
        if y is not None:
            return self._forward_cond(x, y, self.transition_matrix, t)
        else:
            return self._forward_uncond(x, self.transition_matrix, t)
