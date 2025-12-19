import torch
import torch.nn as nn
from torch_geometric.nn import (
    GINConv, SAGEConv, GCNConv, GATv2Conv,
    global_add_pool, global_mean_pool, HeteroConv
)
from modules.GPSLayer import GPSLayer


def build_gnn_layer(kind: str, in_dim: int, out_dim: int, dropout: float, n_heads: int = 4):
    """Factory for arbitrary GNN layers."""
    kind = kind.lower()

    if kind == "gin":
        mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
        return GINConv(mlp)

    elif kind == "graphsage":
        return SAGEConv(in_dim, out_dim)

    elif kind == "gcn":
        return GCNConv(in_dim, out_dim)

    elif kind == "gat":
        heads = n_heads
        assert out_dim % heads == 0
        return GATv2Conv(in_dim, out_dim // heads, heads=heads, dropout=dropout)

    elif kind == "gps":
        return GPSLayer(
            dim=in_dim,
            edge_dim=in_dim,
            num_heads=n_heads,
            dropout=dropout,
        )

    else:
        raise ValueError(f"Unknown conv type: {kind}")


class HeteroGraphEncoder(nn.Module):
    """
    Heterogeneous graph encoder mirroring GraphEncoder but operating on HeteroData.

    - Per-node-type embedding tables (indices -> embeddings).
    - Per-layer HeteroConv with one conv per edge type using build_gnn_layer.
    - Global pooling over all node types, aggregated to a single graph representation.
    """
    def __init__(
        self,
        metadata,
        num_nodes_dict,
        embedding_dim,
        hidden_dim,
        output_dim,
        num_layers: int = 5,
        dropout_prob: float = 0.1,
        pooling="add",
        conv_type: str = "gin",
        attention_heads: int = 4,  # relevant only for gat/gps
        aggr: str = "sum",         # per-edge-type aggregation inside HeteroConv
    ):
        """
        Args:
            metadata: (node_types, edge_types) as returned by HeteroData.metadata().
            num_nodes_dict: dict {node_type: num_nodes} for embedding tables.
            embedding_dim: node embedding dimension (shared across node types).
            hidden_dim: hidden dimension of GNN layers.
            output_dim: final graph-level embedding dimension.
            num_layers: number of HeteroConv layers.
            dropout_prob: dropout in GNN layers (where applicable).
            pooling: 'add', 'mean', or None for graph-level pooling.
            conv_type: 'gin' | 'graphsage' | 'gcn' | 'gat' | 'gps'.
            attention_heads: for 'gat' and 'gps' layers.
            aggr: aggregation inside HeteroConv ('sum', 'mean', etc.).
        """
        super().__init__()

        node_types, edge_types = metadata
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim

        # Node-type specific embeddings
        self.embeddings = nn.ModuleDict()
        for ntype in node_types:
            if ntype not in num_nodes_dict:
                raise ValueError(f"num_nodes_dict missing entry for node type '{ntype}'")
            self.embeddings[ntype] = nn.Embedding(num_nodes_dict[ntype], embedding_dim)

        # Pooling over node types
        if pooling == "add":
            self.pool_func = global_add_pool
        elif pooling == "mean":
            self.pool_func = global_mean_pool
        elif pooling is None:
            self.pool_func = None
        else:
            raise ValueError(f"Invalid pooling='{pooling}'")

        # Build stacked HeteroConv layers
        self.convs = nn.ModuleList()
        for layer_idx in range(num_layers):
            in_dim = embedding_dim if layer_idx == 0 else hidden_dim

            convs_dict = nn.ModuleDict()
            for edge_type in edge_types:
                convs_dict["__".join(edge_type)] = build_gnn_layer(
                    conv_type,
                    in_dim,
                    hidden_dim,
                    dropout_prob,
                    n_heads=attention_heads,
                )

            # Wrap in HeteroConv: maps {edge_type: conv} -> hetero message passing
            hetero_conv = HeteroConv(
                {
                    edge_type: convs_dict["__".join(edge_type)]
                    for edge_type in edge_types
                },
                aggr=aggr,
            )
            self.convs.append(hetero_conv)

        # Graph-level MLP head
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def _embed_nodes(self, data):
        """
        Create a dict of embeddings per node type from integer node IDs in data[ntype].x.
        Expects x to be shape [N, 1] or [N] with integer indices.
        """
        x_dict = {}
        for ntype in self.node_types:
            x = data[ntype].x
            if x.dim() == 2 and x.size(-1) == 1:
                x = x.squeeze(-1)
            x_dict[ntype] = self.embeddings[ntype](x)
        return x_dict

    def forward(self, data):
        """
        Args:
            data: torch_geometric.data.HeteroData with:
                - data[ntype].x (integer ids for embeddings)
                - data[ntype].batch (optional, for graph-level pooling)
                - data[edge_type].edge_index
                - optional data[edge_type].edge_attr for gps/gat/etc.

        Returns:
            Tensor of shape [batch_size, output_dim] if pooling is not None,
            otherwise a dict {node_type: [N_ntype, hidden_dim]} of node embeddings.
        """
        # Initial per-type embeddings
        x_dict = self._embed_nodes(data)

        # Heterogeneous message passing
        for conv in self.convs:
            # HeteroConv expects x_dict, edge_index_dict, and optionally edge_attr_dict
            x_dict = conv(x_dict, data.edge_index_dict)

        # No pooling: concatenate all node embeddings across node types
        if self.pool_func is None:
            xs = []
            for ntype in self.node_types:  # deterministic order from metadata
                xs.append(x_dict[ntype])  # [N_nt, hidden_dim]
            return torch.cat(xs, dim=0)  # [sum_t N_t, hidden_dim]

        # Graph-level pooling over all node types, then aggregate to a single representation
        pooled_per_type = []
        batch_size = None

        for ntype in self.node_types:
            x_ntype = x_dict[ntype]
            batch_ntype = getattr(data[ntype], "batch", None)

            if batch_ntype is None:
                # Single-graph case: create a dummy batch of zeros
                batch_ntype = torch.zeros(
                    x_ntype.size(0), dtype=torch.long, device=x_ntype.device
                )

            pooled_ntype = self.pool_func(x_ntype, batch_ntype)
            pooled_per_type.append(pooled_ntype)

            if batch_size is None:
                batch_size = pooled_ntype.size(0)

        # Stack [num_types, batch_size, hidden_dim] -> [batch_size, num_types, hidden_dim]
        stacked = torch.stack(pooled_per_type, dim=1)

        # Aggregate over node types (sum). You could change to mean or concat if preferred.
        graph_repr = stacked.sum(dim=1)  # [batch_size, hidden_dim]

        graph_repr = self.out(graph_repr)
        return graph_repr
