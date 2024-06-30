import torch
import torch.nn as nn


class SimpleDenoiser(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, time_dim, device):
        super(SimpleDenoiser, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True).float()
        self.fc = nn.Linear(hidden_dim, output_dim).float()
        self.time_dim = time_dim
        self.device = device
        self.embed_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                time_dim,
                input_dim
            ),
        ).float()

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        emb = self.embed_layer(t).unsqueeze(1).repeat(1, x.shape[1], 1)
        combined = (x + emb).float()
        out, _ = self.lstm(combined)
        out = self.fc(out)
        return out
