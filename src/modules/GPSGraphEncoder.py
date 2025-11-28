import torch
import torch.nn as nn
from torch_geometric.nn import (
    GINEConv,
    global_mean_pool,
)
from torch_geometric.utils import add_self_loops
from modules.GPSLayer import GPSLayer


class GPSGraphEncoder(nn.Module):
    """
    Full GPS-style encoder:
      - Node embedding (discrete node IDs or raw features)
      - Edge embedding (optional)
      - Optional positional encodings
      - L GPSLayers
      - Global pooling to graph embedding

    Expects PyG Data with:
      - data.x (LongTensor node IDs OR float features)
      - data.edge_index
      - optional data.edge_attr
      - data.batch for batched graphs
      - optional data.pos_enc (positional encodings per node)
    """

    def __init__(
            self,
            num_node_embeddings: int = None,
            node_in_dim: int = None,
            edge_in_dim: int = None,
            pos_enc_dim: int = None,
            hidden_dim: int = 128,
            num_layers: int = 6,
            num_heads: int = 8,
            dropout: float = 0.1,
            out_dim: int = 128,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Node encoder: either embedding (for ID) or linear (for features)
        if num_node_embeddings is not None:
            self.node_encoder = nn.Embedding(num_node_embeddings, hidden_dim)
        elif node_in_dim is not None:
            self.node_encoder = nn.Linear(node_in_dim, hidden_dim)
        else:
            raise ValueError("Either num_node_embeddings or node_in_dim must be provided.")

        # Edge encoder (can be None -> zeros)
        if edge_in_dim is not None:
            self.edge_encoder = nn.Linear(edge_in_dim, hidden_dim)
        else:
            self.edge_encoder = None

        # Positional encoding projection (optional)
        if pos_enc_dim is not None:
            self.pos_encoder = nn.Linear(pos_enc_dim, hidden_dim)
        else:
            self.pos_encoder = None

        # Virtual node initial embedding (learned, shared across graphs)
        self.virtual_init = nn.Parameter(torch.randn(hidden_dim))

        # Stack of GPS layers
        self.layers = nn.ModuleList([
            GPSLayer(
                dim=hidden_dim,
                edge_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Output MLP on graph-level embedding
        self.graph_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def encode_nodes(self, data):
        x = data.x

        # Node encoding:
        # - if x is LongTensor (node IDs) -> embedding
        # - else -> assume float features -> linear
        if isinstance(self.node_encoder, nn.Embedding):
            if x.dim() == 2:
                x = x.squeeze(-1)
            x = self.node_encoder(x)
        else:
            x = self.node_encoder(x)

        # Positional encodings (optional)
        if hasattr(data, "pos_enc") and self.pos_encoder is not None:
            pe = self.pos_encoder(data.pos_enc)
            x = x + pe

        return x

    def encode_edges(self, data, num_edges):
        if self.edge_encoder is None:
            # If nothing, use zeros as edge_attr
            return torch.zeros(
                num_edges, self.hidden_dim,
                device=data.edge_index.device,
            )

        edge_attr = getattr(data, "edge_attr", None)
        if edge_attr is None:
            return torch.zeros(
                num_edges, self.hidden_dim,
                device=data.edge_index.device,
            )
        return self.edge_encoder(edge_attr)

    def forward(self, data):
        """
        Returns:
          - graph_repr: [B, out_dim]
        """
        device = data.edge_index.device

        # Node features
        x = self.encode_nodes(data)  # [N, hidden_dim]
        batch = data.batch if data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Ensure self-loops exist (helps attention / stability)
        edge_index, _ = add_self_loops(data.edge_index, num_nodes=x.size(0))
        num_edges = edge_index.size(1)

        # Edge features
        edge_attr = self.encode_edges(data, num_edges)

        # Virtual node initialization per graph
        num_graphs = int(batch.max().item()) + 1
        v0 = self.virtual_init.unsqueeze(0).expand(num_graphs, -1).to(device)  # [B, hidden_dim]

        v = v0
        for layer in self.layers:
            x, v = layer(x, edge_index, edge_attr, batch, v)

        # Graph-level representation: mean over nodes + virtual node
        g_nodes = global_mean_pool(x, batch)  # [B, hidden_dim]
        g = g_nodes + v  # combine node-pooled and virtual node

        out = self.graph_mlp(g)  # [B, out_dim]
        return out
