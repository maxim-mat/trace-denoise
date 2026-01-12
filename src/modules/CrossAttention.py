import torch.nn as nn
import torch


class CrossAttention(nn.Module):
    def __init__(self, channels, size, n_heads=2, q_dim=None, kv_dim=None, init_gate=0.0, emb_dim=256):
        super(CrossAttention, self).__init__()
        self.q_dim = q_dim if q_dim is not None else channels
        self.kv_dim = kv_dim if kv_dim is not None else channels
        self.gate = nn.Parameter(torch.tensor(float(init_gate)))
        self.channels = channels
        self.size = size
        self.ln_q = nn.LayerNorm([self.q_dim])
        self.ln_kv = nn.LayerNorm([self.kv_dim])
        self.q_proj = nn.Linear(self.q_dim, self.channels)
        self.k_proj = nn.Linear(self.kv_dim, self.channels)
        self.v_proj = nn.Linear(self.kv_dim, self.channels)
        self.out_proj = nn.Linear(self.channels, self.q_dim)
        self.mha = nn.MultiheadAttention(self.channels, num_heads=n_heads, batch_first=True)
        self.ff_cross = nn.Sequential(
            nn.LayerNorm([self.q_dim]),
            nn.Linear(self.q_dim, self.q_dim),
            nn.GELU(),
            nn.Linear(self.q_dim, self.q_dim),
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                self.q_dim
            ),
        )

    def forward(self, query, key, value, t):
        query = query.swapaxes(1, 2)
        key = key.swapaxes(1, 2)
        value = value.swapaxes(1, 2)

        query_ln = self.ln_q(query)
        key_ln = self.ln_kv(key)
        value_ln = self.ln_kv(value)

        queryln_proj = self.q_proj(query_ln)
        key_proj = self.k_proj(key_ln)
        value_proj = self.v_proj(value_ln)

        attention_value, _ = self.mha(queryln_proj, key_proj, value_proj)
        attention_value = self.out_proj(attention_value) * nn.Softplus()(self.gate) + query  # Residual connection
        attention_value = self.ff_cross(attention_value) + attention_value  # Residual after feed-forward

        emb = self.emb_layer(t).unsqueeze(2).repeat(1, 1, query.shape[1])

        return attention_value.swapaxes(2, 1).view(-1, self.q_dim, self.size) + emb
