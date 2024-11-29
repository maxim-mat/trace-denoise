import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self, channels, size):
        super(CrossAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 2, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_cross = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, query, key, value):
        query = query.view(-1, self.channels, self.size).swapaxes(1, 2)
        key = key.view(-1, self.channels, self.size).swapaxes(1, 2)
        value = value.view(-1, self.channels, self.size).swapaxes(1, 2)

        query_ln = self.ln(query)
        key_ln = self.ln(key)
        value_ln = self.ln(value)

        attention_value, _ = self.mha(query_ln, key_ln, value_ln)
        attention_value = attention_value + query  # Residual connection
        attention_value = self.ff_cross(attention_value) + attention_value  # Residual after feed-forward

        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size)