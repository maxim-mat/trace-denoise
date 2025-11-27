class GPSLayer(nn.Module):
    """
    One GPS-style block:
      - local GINEConv branch
      - global Transformer attention branch
      - fusion + FFN
      - virtual node update (per-graph)
    """

    def __init__(
            self,
            dim,
            edge_dim,
            num_heads=8,
            dropout=0.1,
    ):
        super().__init__()
        # Local branch
        self.local_mp = EdgeGINEConv(
            in_dim=dim,
            out_dim=dim,
            hidden_dim=dim,
            dropout=dropout,
        )

        # Global branch
        self.global_attn = GraphTransformerLayer(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Fusion + norm
        self.norm1 = nn.LayerNorm(dim)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

        # Virtual node GRU (global context)
        self.v_gru = nn.GRUCell(dim, dim)
        self.v_proj = nn.Linear(dim, dim)  # broadcast virtual node back to nodes

    def forward(self, x, edge_index, edge_attr, batch, v):
        """
        x: [N, dim]
        edge_index: [2, E]
        edge_attr: [E, edge_dim] or None (then zeros)
        batch: [N] graph indices
        v: [B, dim] virtual node embeddings (B = num_graphs)
        """
        # Local
        x_local = self.local_mp(x, edge_index, edge_attr)

        # Global
        x_global = self.global_attn(x, batch)

        # Fuse branches + residual + norm (pre-norm style)
        h = x + x_local + x_global
        h = self.norm1(h)

        # Update virtual node: pool over nodes per graph
        g = global_mean_pool(h, batch)  # [B, dim]
        v_new = self.v_gru(g, v)        # [B, dim]

        # Broadcast virtual node back to nodes
        v_broadcast = self.v_proj(v_new[batch])  # [N, dim]
        h = h + v_broadcast

        # FFN + residual + norm
        h_ffn = self.ffn(h)
        h = self.norm2(h + h_ffn)

        return h, v_new
