import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, LayerNorm
from torch_geometric.nn.aggr import AttentionalAggregation


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int, edge_dim: int, dropout: float):
        super().__init__()
        self.gat = GATv2Conv(in_dim, out_dim, heads=heads, edge_dim=edge_dim, add_self_loops=True, dropout=dropout)
        self.ln = LayerNorm(out_dim * heads, affine=True)
        self.proj = None
        if in_dim != out_dim * heads:
            self.proj = nn.Linear(in_dim, out_dim * heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        h = self.gat(x, edge_index, edge_attr)
        if self.proj is not None:
            x = self.proj(x)
        x = x + h
        x = self.ln(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class GATv2Encoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, heads1: int = 4, heads2: int = 4, dropout: float = 0.3):
        super().__init__()
        # Update edge_dim to 6 to accommodate new features
        self.block1 = ResidualBlock(in_dim, hidden, heads1, edge_dim=6, dropout=dropout)
        self.block2 = ResidualBlock(hidden * heads1, hidden, heads2, edge_dim=6, dropout=dropout)
        # final heads=1, keep dim hidden
        self.conv3 = GATv2Conv(hidden * heads2, 64, heads=1, edge_dim=6, add_self_loops=True, dropout=dropout)
        self.ln3 = LayerNorm(64, affine=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        x = self.block1(x, edge_index, edge_attr)
        x = self.block2(x, edge_index, edge_attr)
        x = self.conv3(x, edge_index, edge_attr)
        x = self.ln3(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class ShotHead(nn.Module):
    def __init__(self, in_dim: int = 64):
        super().__init__()
        gate_nn = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.pool = AttentionalAggregation(gate_nn)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x, batch):
        hg = self.pool(x, batch)
        logit = self.mlp(hg)
        return logit.view(-1, 1)


class ReceiverHead(nn.Module):
    def __init__(self, in_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.mlp(x)


class MultiTaskGATv2(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, heads: int = 4, dropout: float = 0.3, edge_dim: int = 6):
        super().__init__()
        self.encoder = GATv2Encoder(in_dim, hidden=hidden, heads1=heads, heads2=heads, dropout=dropout)
        self.shot_head = ShotHead(64)
        self.receiver_head = ReceiverHead(64)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        h = self.encoder(x, edge_index, edge_attr)
        shot_logit = self.shot_head(h, batch)
        receiver_logit = self.receiver_head(h)
        return receiver_logit, shot_logit


# Add new separate models for single task training
class SingleTaskGATv2Shot(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, heads: int = 4, dropout: float = 0.3, edge_dim: int = 6):
        super().__init__()
        self.encoder = GATv2Encoder(in_dim, hidden=hidden, heads1=heads, heads2=heads, dropout=dropout)
        self.shot_head = ShotHead(64)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        h = self.encoder(x, edge_index, edge_attr)
        shot_logit = self.shot_head(h, batch)
        return shot_logit


class SingleTaskGATv2Receiver(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, heads: int = 4, dropout: float = 0.3, edge_dim: int = 6):
        super().__init__()
        self.encoder = GATv2Encoder(in_dim, hidden=hidden, heads1=heads, heads2=heads, dropout=dropout)
        self.receiver_head = ReceiverHead(64)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        h = self.encoder(x, edge_index, edge_attr)
        receiver_logit = self.receiver_head(h)
        return receiver_logit
