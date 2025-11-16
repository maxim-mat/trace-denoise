import torch_geometric as pyg
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool
import torch.nn as nn
import torch.nn.functional as F
import torch


class GraphEncoder(nn.Module):
    def __init__(self, num_nodes, embedding_dim, hidden_dim, output_dim, num_layers=5, dropout_prob=0.1, pooling='add'):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.pool_func = global_add_pool if pooling == 'add' else global_mean_pool
        self.convs = nn.ModuleList(
            [GINConv(
                nn.Sequential(
                    nn.Linear(embedding_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_prob),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU()
                )
            ) if i == 0 else
              GINConv(
                  nn.Sequential(
                      nn.Linear(hidden_dim, hidden_dim),
                      nn.BatchNorm1d(hidden_dim),
                      nn.ReLU(),
                      nn.Dropout(p=dropout_prob),
                      nn.Linear(hidden_dim, hidden_dim),
                      nn.BatchNorm1d(hidden_dim),
                      nn.ReLU()
                  )
              ) for i in range(num_layers)]
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.embedding(x).squeeze(1)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = self.pool_func(x, batch=torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        x = self.out(x)
        return x
