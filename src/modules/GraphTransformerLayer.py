import torch.nn as nn


class GraphTransformerLayer(nn.Module):
    """
    Global attention over nodes with graph-wise masking.
    Uses torch.nn.MultiheadAttention on all nodes at once,
    but masks attention between different graphs in the batch.
    """

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,  # expects [L, B, C]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch):
        """
        x: [N, dim]
        batch: [N] with graph indices
        """
        N, dim = x.size()
        device = x.device

        # Treat all nodes as one sequence (L = N, B = 1)
        h = x.unsqueeze(1)  # [N, 1, dim] -> L=N, B=1

        # Build attention mask so nodes from different graphs cannot attend
        # attn_mask[i, j] = True => position j is masked for query i
        batch_i = batch.unsqueeze(0)        # [1, N]
        batch_j = batch.unsqueeze(1)        # [N, 1]
        attn_mask = (batch_i != batch_j)    # [N, N], True where different graph

        h_out, _ = self.mha(
            h, h, h,
            attn_mask=attn_mask.to(device)
        )  # [N, 1, dim]
        h_out = h_out.squeeze(1)  # [N, dim]
        h_out = self.dropout(h_out)
        return h_out
