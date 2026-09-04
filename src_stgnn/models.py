"""
Model Architectures for NFL Big Data Bowl 2024 tackle prediction task

This module defines neural network architectures for predicting tackle
locations in NFL plays. It includes two main model types: SportsTransformer and
TheZooArchitecture, along with a shared LightningModule wrapper for training.

Classes:
    SportsTransformer: Generalized Transformer-based model for sports tracking data
    TheZooArchitecture: Baseline approach based on the winning solution of the NFL Big Data Bowl 2020
    LitModel: LightningModule wrapper for shared training functionality
"""

from typing import Any

import torch
from lightning import LightningModule
from torch import Tensor, nn, squeeze
from torch.optim import AdamW

from datasets import TEMPORAL_MODEL_TYPES

torch.set_float32_matmul_precision("medium")


class SportsTransformer(nn.Module):
    """
    Transformer model that treats all 22 players as a sequence for tackle prediction.
    """

    def __init__(
        self,
        feature_len: int,
        model_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        """
        Initialize the SportsTransformer.

        Args:
            feature_len (int): Number of input features per player.
            model_dim (int): Dimension of the model's internal representations.
            num_layers (int): Number of transformer encoder layers.
            dropout (float): Dropout rate for regularization.
        """
        super().__init__()
        dim_feedforward = model_dim * 4
        num_heads = min(16, max(2, 2 * round(model_dim / 64)))  # attention is better optimized for even number of heads

        self.hyperparams = {
            "model_dim": model_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "dim_feedforward": dim_feedforward,
        }

        # Normalize input features
        self.feature_norm_layer = nn.BatchNorm1d(feature_len)

        # Embed input features to model dimension
        self.feature_embedding_layer = nn.Sequential(
            nn.Linear(feature_len, model_dim),
            nn.ReLU(),
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
        )

        # Transformer Encoder
        # This component applies multiple layers of self-attention and feed-forward networks
        # to process player data in a permutation-equivariant manner.

        # Key properties:
        # 1. Player-order equivariance: The output maintains the same shape and player order as the input.
        # 2. Contextual feature extraction: Transforms initial player features into rich, context-aware representations.
        # 3. Inter-player relationships: Captures complex interactions between players across the field.
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        # Pool across player dimension
        # We pool because this task is a single value across all players, you don't need to pool for all tasks.
        self.player_pooling_layer = nn.AdaptiveAvgPool1d(1)

        # Task-specific Decoder to predict tackle location.
        # self.decoder = nn.Sequential(
        #     nn.Linear(model_dim, model_dim // 4),
        #     nn.ReLU(),
        #     nn.Linear(model_dim // 4, 2),
        # )

        self.decoder = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim // 4),
            nn.ReLU(),
            nn.LayerNorm(model_dim // 4),
            nn.Linear(model_dim // 4, 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the SportsTransformer.

        Args:
            x (Tensor): Input tensor of shape [batch_size, num_players, feature_len].

        Returns:
            Tensor: Predicted tackle location of shape [batch_size, 2].
        """
        # x: [B: batch_size, P: # of players, F: feature_len]
        B, P, F = x.size()

        # Normalize features
        x = self.feature_norm_layer(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B,P,F] -> [B,P,F]

        # Embed features
        x = self.feature_embedding_layer(x)  # [B,P,F] -> [B,P,M: model_dim]

        # Apply transformer encoder
        x = self.transformer_encoder(x)  # [B,P,M] -> [B,P,M]

        # Pool over player dimension
        x = squeeze(self.player_pooling_layer(x.permute(0, 2, 1)), -1)  # [B,M,P] -> [B,M]

        # Decode to predict tackle location
        x = self.decoder(x)  # [B,M] -> [B,2]

        return x


class TheZooArchitecture(nn.Module):
    """
    TheZooArchitecture represents a baseline model to compare against the SportsTransformer.
    It was the winning solution for the 2020 Big Data Bowl designed to predict run game yardage gained with an
    innovative (at the time) approach to solving player-equivariance problem. At a high level, the approach requires
    generating a set of pairwise interaction vectors between offense (10) and defense (11) players, applying
    feedforward layers to each interaction embedding independently, and then pooling across interaction
    dimensions to get to a final output.

    Based on: https://github.com/juancamilocampos/nfl-big-data-bowl-2020/blob/master/1st_place_zoo_solution_v2.ipynb
    """

    # 10 offensive players and 11 defensive players
    NUM_OFFENSE = 10
    NUM_DEFENSE = 11

    def __init__(
        self,
        feature_len: int,
        model_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        """
        Initialize the TheZooArchitecture.

        Args:
            feature_len (int): Number of input features in each interaction vector.
            model_dim (int): Dimension of the model.
            num_layers (int): Number of convolutional layers.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.hyperparams = {
            "model_dim": model_dim,
            "num_layers": num_layers,
        }

        # Normalize input features
        self.feature_norm_layer = nn.BatchNorm2d(feature_len)

        # Embed input features to model dimension
        self.feature_embedding_layer = nn.Sequential(
            nn.Linear(feature_len, model_dim),
            nn.ReLU(),
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
        )

        # Feedforward Layer block 1 across all interaction vectors
        # Notably, this "CNN" is just a hacky way to apply a Linear or Feedforward layer across all interaction vectors.
        # It is not doing any convolution between neighboring players (as that would violate permutation equivariance).
        self.ff_block1 = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(in_channels=model_dim, out_channels=model_dim, kernel_size=(1, 1), stride=(1, 1)),
                    nn.ReLU(),
                )
                for _ in range(num_layers)
            ]
        )

        # Feedforward Layer block 2 after pooling across offensive players
        self.ff_block2 = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv1d(in_channels=model_dim, out_channels=model_dim, kernel_size=1, stride=1),
                    nn.ReLU(),
                    nn.BatchNorm1d(model_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.output_layer = nn.Sequential(
            *(
                [
                    nn.Sequential(
                        nn.Linear(model_dim, model_dim),
                        nn.ReLU(),
                        nn.BatchNorm1d(model_dim),
                    )
                    for _ in range(max(0, num_layers - 2))
                ]
                + [
                    nn.Dropout(dropout),
                    nn.Linear(model_dim, model_dim // 4),
                    nn.ReLU(),
                    nn.LayerNorm(model_dim // 4),
                    nn.Linear(model_dim // 4, 2),
                ]
            )
        )

        # Pooling layers for collapsing offensive and defensive dimensions
        # Created in __init__ (not forward()) so fvcore can trace FLOPs
        self.pool_offense_max = nn.MaxPool2d((1, self.NUM_OFFENSE))
        self.pool_offense_avg = nn.AvgPool2d((1, self.NUM_OFFENSE))
        self.pool_defense_max = nn.MaxPool1d(self.NUM_DEFENSE)
        self.pool_defense_avg = nn.AvgPool1d(self.NUM_DEFENSE)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the TheZooArchitecture.

        Args:
            x (Tensor): Input tensor of shape [B, O, D, F].

        Returns:
            Tensor: Output tensor of shape [B, 2].
        """
        # x: [B: batch_size, O: offense, D: defense, F: feature_len]
        B, O, D, F = x.size()  # B=Batch, O=Offense, D=Defense, F=Feature

        # Normalize features
        x = self.feature_norm_layer(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # [B,O,D,F] -> [B,O,D,F]
        # Embed features
        x = self.feature_embedding_layer(x)  # [B,O,D,F] -> [B,O,D,M: model_dim]

        # apply first block, pool and collapse offensive dimension
        x = self.ff_block1(x.permute(0, 3, 2, 1))  # [B,O,D,M] -> [B,M,D,O]
        # Zoo Authors mentioned using a weighted sum of max and avg pooling helped most (experimentally verified hparam)
        x = self.pool_offense_max(x) * 0.3 + self.pool_offense_avg(x) * 0.7  # [B,M,D,O] -> [B,M,D,1]
        x = x.squeeze(-1)  # [B,M,D,1] -> [B,M,D]

        # apply second block, pool and collapse defensive dimension
        x = self.ff_block2(x)  # [B,M,D] -> [B,M,D]
        x = self.pool_defense_max(x) * 0.3 + self.pool_defense_avg(x) * 0.7  # [B,M,D] -> [B,M,1]
        x = x.squeeze(-1)  # [B,M,1] -> [B,M]

        # apply decoder
        x = self.output_layer(x)  # [B,M] -> [B,2]
        assert x.shape == (B, 2)
        return x


def _build_decoder(model_dim: int, dropout: float) -> nn.Sequential:
    """Shared task decoder mapping a pooled [B, model_dim] vector to [B, 2] (x, y).

    Mirrors the head used by SportsTransformer so the temporal models differ from the
    baselines only in how they encode/aggregate, not in the prediction head.
    """
    return nn.Sequential(
        nn.Linear(model_dim, model_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(model_dim, model_dim // 4),
        nn.ReLU(),
        nn.LayerNorm(model_dim // 4),
        nn.Linear(model_dim // 4, 2),
    )


EDGE_FEATURE_DIM = 6  # [dx, dy, distance, dvx, dvy, closing_speed] per ordered pair


def _knn_adj(pos: Tensor, side: Tensor, k: int, cross: bool, same: bool) -> Tensor:
    '''
    knn topology help feature to construct knn adjacency matrix

    Args:
        pos: (B, N, 2) player coords
        side: (B, N) team indicator (+1 = off, -1 = def)
        k: number of nearest neighbors to connect
        cross: allow opponent connections
        same: allow teammate connections

    Return:
        adj: (B, N, N) boolean adjacency matrix
            adj[b, i, j] = True if player i is connected to player j
    '''
    B, N, _ = pos.shape
    diff = pos.unsqueeze(2) - pos.unsqueeze(1)  # pairwise distance diff between player i and j
    dist2 = (diff ** 2).sum(-1)     # (B, N, N) Squared Euclidean distance matrix
    INF = float('inf')

    same_team = (side.unsqueeze(2) * side.unsqueeze(1)) > 0
    opp_team  = ~same_team

    adj = torch.zeros(B, N, N, dtype=torch.bool, device=pos.device) # initialize adjacency matrix

    if same and cross:
        # knn_hybrid: k nearest same + k nearest cross

        # sets non-teammate distances to infinity (can't be kNN)
        d_same = dist2.masked_fill(~same_team, INF)

        # diagonal set to infinity (can't be kNN of self)
        d_same.diagonal(dim1=1, dim2=2).fill_(INF)

        # find indices of kNNs and write the edges into adj (connect player to kNNs)
        adj.scatter_(2, d_same.topk(k, dim=-1, largest=False).indices, True)    # (B, N, N)

        # repeat for opposite-team neighbors
        d_cross = dist2.masked_fill(~opp_team, INF)
        adj.scatter_(2, d_cross.topk(k, dim=-1, largest=False).indices, True)

    elif same:
        # same-team kNN only
        d_same = dist2.masked_fill(~same_team, INF)
        d_same.diagonal(dim1=1, dim2=2).fill_(INF)
        adj.scatter_(2, d_same.topk(k, dim=-1, largest=False).indices, True)

    elif cross:
        # opponent kNN only
        d_cross = dist2.masked_fill(~opp_team, INF)
        adj.scatter_(2, d_cross.topk(k, dim=-1, largest=False).indices, True)

    else:
        # plain knn — no team filter
        d = dist2.clone()
        d.diagonal(dim1=1, dim2=2).fill_(INF)
        adj.scatter_(2, d.topk(k, dim=-1, largest=False).indices, True)

    return adj

def _add_hub(adj: Tensor, bc: Tensor) -> Tensor:
    """
    Force ball carrier to act as hub node.
    Every player has an edge to/from the ball carrier"""
    b, n, _ = adj.shape
    bcb = (bc > 0.5).unsqueeze(1).expand(b, n, n)
    # returns original edges and edges to/from ball carrier
    return adj | bcb | bcb.transpose(1, 2) 


def build_adjacency_torch(side: Tensor, bc: Tensor, pos: Tensor, knn_k: int, topology: str) -> Tensor:
    """Batched (B, N, N) boolean adjacency from per-player `side` (+1/-1) and `bc` (0/1).

    Mirrors src/graphs.build_adjacency but in torch, and always adds self-loops so no
    node has a fully-masked attention row (and so `full` matches a standard Transformer,
    whose self-attention includes the token itself).
    """
    b, n = side.shape
    if topology == "full":
        adj = torch.ones(b, n, n, dtype=torch.bool, device=side.device)
    elif topology == "bipartite":
        adj = (side.unsqueeze(2) * side.unsqueeze(1)) < 0  # opposite teams
    elif topology == "hub":
        bcb = bc > 0.5
        adj = bcb.unsqueeze(2) | bcb.unsqueeze(1)  # either endpoint is the ball carrier

    # knn specific topology
    elif topology == "knn_same":
        adj = _knn_adj(pos, side, k=knn_k, cross=False, same=True)  # same team
    elif topology == "knn_cross":
        adj = _knn_adj(pos, side, k=knn_k, cross=True,  same=False) # opp team
    elif topology == "knn_hybrid":
        adj = _knn_adj(pos, side, k=knn_k//2, cross=True,  same=True)  # k same team & k opp team
    elif topology == "knn":
        adj = _knn_adj(pos, side, k=knn_k, cross=False, same=False) # normal knn

    # knn with ball carrier connection guaranteed
    elif topology == "knn_same_hub":
        adj = _add_hub(_knn_adj(pos, side, k=knn_k, cross=False, same=True),  bc)
    elif topology == "knn_cross_hub":
        adj = _add_hub(_knn_adj(pos, side, k=knn_k, cross=True,  same=False), bc)
    elif topology == "knn_hybrid_hub":
        adj = _add_hub(_knn_adj(pos, side, k=knn_k//2, cross=True,  same=True),  bc)
    elif topology == "knn_hub":
        adj = _add_hub(_knn_adj(pos, side, k=knn_k, cross=False, same=False), bc)

    else:
        raise ValueError(f"unknown topology {topology!r}")
    eye = torch.eye(n, dtype=torch.bool, device=side.device).unsqueeze(0)
    return adj | eye


def edge_features_torch(pos: Tensor, vel: Tensor) -> Tensor:
    """(B, N, N, 6) physical edge features: relative position, distance, relative velocity, closing speed."""
    delta = pos.unsqueeze(1) - pos.unsqueeze(2)  # [b,i,j] = pos_j - pos_i  -> (B,N,N,2)
    relv = vel.unsqueeze(1) - vel.unsqueeze(2)
    dist = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)  # (B,N,N,1)
    closing = -(delta * relv).sum(-1, keepdim=True) / dist.clamp(min=1e-6)  # positive = converging
    return torch.cat([delta, dist, relv, closing], dim=-1)


class GATLayer(nn.Module):
    """Masked multi-head attention with an optional additive edge-feature bias + FFN.

    This is exactly a Transformer encoder layer plus (a) an adjacency mask (-inf on
    non-edges) and (b) a per-head scalar bias projected from the edge features. With
    `full` topology and no edge features it reduces to standard self-attention.
    """

    def __init__(self, model_dim: int, num_heads: int, dropout: float, edge_dim: int = 0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.q = nn.Linear(model_dim, model_dim)
        self.k = nn.Linear(model_dim, model_dim)
        self.v = nn.Linear(model_dim, model_dim)
        self.o = nn.Linear(model_dim, model_dim)
        self.edge_proj = nn.Linear(edge_dim, num_heads) if edge_dim > 0 else None
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.ff = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(model_dim * 4, model_dim)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, h: Tensor, adj: Tensor, edge_feats: Tensor | None) -> Tensor:
        b, n, _ = h.shape
        q = self.q(h).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)  # B,H,N,d
        k = self.k(h).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(h).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        logits = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)  # B,H,N,N
        if self.edge_proj is not None and edge_feats is not None:
            logits = logits + self.edge_proj(edge_feats).permute(0, 3, 1, 2)  # per-head edge bias
        logits = logits.masked_fill(~adj.unsqueeze(1), float("-inf"))
        attn = self.drop(torch.softmax(logits, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(b, n, -1)  # B,N,M
        h = self.norm1(h + self.drop(self.o(out)))
        h = self.norm2(h + self.drop(self.ff(h)))
        return h


class WindowedTransformer(nn.Module):
    """
    Non-recurrent temporal control. Each player's T-frame window is flattened into a
    single (T*feature_len)-dim token, then processed by the standard SportsTransformer
    self-attention over the 22 players. Holds the interaction module identical to the
    baseline transformer and varies only the per-player encoder (flatten vs. recurrence),
    which isolates "temporal information via attention" from recurrence itself.

    Input: [batch, T, 22, feature_len] -> [batch, 2]
    """

    def __init__(
        self,
        feature_len: int,
        model_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.3,
        window_length: int = 5,
    ):
        super().__init__()
        self.window_length = window_length
        # Reuse the full single-frame transformer with a fattened per-player feature dim.
        self.transformer = SportsTransformer(
            feature_len=feature_len * window_length,
            model_dim=model_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.hyperparams = {**self.transformer.hyperparams, "window_length": window_length}

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, P, F] -> [B, P, T*F] (concatenate each player's frames into one token)
        B, T, P, F = x.size()
        x = x.permute(0, 2, 1, 3).reshape(B, P, T * F)
        return self.transformer(x)


class PureGRU(nn.Module):
    """
    Per-player GRU over the T-frame window, mean-pooled across players, then decoded.
    No cross-player attention -- isolates recurrence WITHOUT interaction modeling.
    `num_layers` sets the GRU depth (the GRU is this model's primary processor).

    Input: [batch, T, 22, feature_len] -> [batch, 2]
    """

    def __init__(
        self,
        feature_len: int,
        model_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.3,
        window_length: int = 5,
    ):
        super().__init__()
        self.window_length = window_length
        self.hyperparams = {
            "model_dim": model_dim,
            "num_layers": num_layers,
            "window_length": window_length,
        }
        self.feature_norm_layer = nn.BatchNorm1d(feature_len)
        self.gru = nn.GRU(
            input_size=feature_len,
            hidden_size=model_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder = _build_decoder(model_dim, dropout)

    def forward(self, x: Tensor) -> Tensor:
        B, T, P, F = x.size()
        # Normalize features across all (batch, time, player) positions.
        x = self.feature_norm_layer(x.reshape(-1, F)).reshape(B, T, P, F)
        # Per-player sequence: [B, T, P, F] -> [B, P, T, F] -> [B*P, T, F]
        x = x.permute(0, 2, 1, 3).reshape(B * P, T, F)
        _, h_n = self.gru(x)  # h_n: [num_layers, B*P, model_dim]
        x = h_n[-1].reshape(B, P, -1)  # last-layer hidden state per player -> [B, P, model_dim]
        x = x.mean(dim=1)  # permutation-invariant mean-pool over players -> [B, model_dim]
        return self.decoder(x)


class HybridTS(nn.Module):
    """
    Time -> Space. A shared single-layer GRU encodes each player's T-frame trajectory
    into one embedding; self-attention then models interaction across the 22 players.
    The GRU plays the role of the per-player encoder (analogous to the linear embedding
    in WindowedTransformer), so `num_layers` controls the attention depth and the two
    models are comparable up to "flatten vs. recurrence" for the encoder.

    Input: [batch, T, 22, feature_len] -> [batch, 2]
    """

    def __init__(
        self,
        feature_len: int,
        model_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.3,
        window_length: int = 5,
    ):
        super().__init__()
        self.window_length = window_length
        num_heads = min(16, max(2, 2 * round(model_dim / 64)))
        dim_feedforward = model_dim * 4
        self.hyperparams = {
            "model_dim": model_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "dim_feedforward": dim_feedforward,
            "window_length": window_length,
        }
        self.feature_norm_layer = nn.BatchNorm1d(feature_len)
        # Single-layer GRU per-player temporal encoder.
        self.gru = nn.GRU(input_size=feature_len, hidden_size=model_dim, num_layers=1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.player_pooling_layer = nn.AdaptiveAvgPool1d(1)
        self.decoder = _build_decoder(model_dim, dropout)

    def forward(self, x: Tensor) -> Tensor:
        B, T, P, F = x.size()
        x = self.feature_norm_layer(x.reshape(-1, F)).reshape(B, T, P, F)
        x = x.permute(0, 2, 1, 3).reshape(B * P, T, F)
        _, h_n = self.gru(x)
        x = h_n[-1].reshape(B, P, -1)  # [B, P, model_dim] trajectory embeddings
        x = self.transformer_encoder(x)  # attention across players
        x = squeeze(self.player_pooling_layer(x.permute(0, 2, 1)), -1)  # pool players -> [B, model_dim]
        return self.decoder(x)


class HybridST(nn.Module):
    """
    Space -> Time. Self-attention models interaction across the 22 players at EACH
    frame (shared weights, set->set, no pooling), then a shared single-layer GRU
    integrates each player's sequence of contextualized embeddings over time. Same
    components as HybridTS (one GRU + `num_layers` attention); differs ONLY in ordering.

    Input: [batch, T, 22, feature_len] -> [batch, 2]
    """

    def __init__(
        self,
        feature_len: int,
        model_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.3,
        window_length: int = 5,
    ):
        super().__init__()
        self.window_length = window_length
        num_heads = min(16, max(2, 2 * round(model_dim / 64)))
        dim_feedforward = model_dim * 4
        self.hyperparams = {
            "model_dim": model_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "dim_feedforward": dim_feedforward,
            "window_length": window_length,
        }
        self.feature_norm_layer = nn.BatchNorm1d(feature_len)
        self.feature_embedding_layer = nn.Sequential(
            nn.Linear(feature_len, model_dim),
            nn.ReLU(),
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
        )
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.gru = nn.GRU(input_size=model_dim, hidden_size=model_dim, num_layers=1, batch_first=True)
        self.player_pooling_layer = nn.AdaptiveAvgPool1d(1)
        self.decoder = _build_decoder(model_dim, dropout)

    def forward(self, x: Tensor) -> Tensor:
        B, T, P, F = x.size()
        x = self.feature_norm_layer(x.reshape(-1, F)).reshape(B, T, P, F)
        # Embed + attend across players within each frame (fold time into the batch dim).
        x = self.feature_embedding_layer(x.reshape(B * T, P, F))  # [B*T, P, model_dim]
        x = self.transformer_encoder(x)  # attention across players, per frame
        # Per-player temporal integration: [B*T, P, M] -> [B, T, P, M] -> [B*P, T, M]
        x = x.reshape(B, T, P, -1).permute(0, 2, 1, 3).reshape(B * P, T, -1)
        _, h_n = self.gru(x)
        x = h_n[-1].reshape(B, P, -1)  # [B, P, model_dim]
        x = squeeze(self.player_pooling_layer(x.permute(0, 2, 1)), -1)  # pool players -> [B, model_dim]
        return self.decoder(x)

class STGNN_TS(nn.Module):
    """
    Time -> Space with GAT. A shared single-layer GRU encodes each player's T-frame
    trajectory into one embedding; stacked GATLayers then model interaction across
    the 22 embeddings with a structured adjacency and optional edge features.
    Drop-in replacement for HybridTS with GATLayer in place of plain TransformerEncoder.

    Input: [batch, T, 22, feature_len] -> [batch, 2]
    """

    def __init__(
        self,
        feature_len: int,
        model_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.3,
        window_length: int = 15,
        topology: str = "full",
        edge_features: bool = True,
        knn_k: int = 8,
    ):
        super().__init__()
        self.window_length = window_length
        self.topology = topology
        self.use_edge_features = edge_features
        self.knn_k = knn_k
        num_heads = min(16, max(2, 2 * round(model_dim / 64)))
        self.hyperparams = {
            "model_dim": model_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "window_length": window_length,
            "topology": topology,
            "edge_features": int(edge_features),
            "knn_k": knn_k,
        }
        self.feature_norm_layer = nn.BatchNorm1d(feature_len)
        self.gru = nn.GRU(input_size=feature_len, hidden_size=model_dim, num_layers=1, batch_first=True)
        edge_dim = EDGE_FEATURE_DIM if edge_features else 0
        self.gat_layers = nn.ModuleList([GATLayer(model_dim, num_heads, dropout, edge_dim) for _ in range(num_layers)])
        self.player_pooling_layer = nn.AdaptiveAvgPool1d(1)
        self.decoder = _build_decoder(model_dim, dropout)

    def forward(self, x: Tensor) -> Tensor:
        B, T, P, F = x.size()
        # Extract pos/vel/side/bc from the LAST frame for adjacency (most recent snapshot)
        last = x[:, -1, :, :]  # [B, P, F]
        pos, vel, side, bc = last[..., 0:2], last[..., 2:4], last[..., 6], last[..., 7]

        x = self.feature_norm_layer(x.reshape(-1, F)).reshape(B, T, P, F)
        # GRU per player over time
        x = x.permute(0, 2, 1, 3).reshape(B * P, T, F)
        _, h_n = self.gru(x)
        x = h_n[-1].reshape(B, P, -1)  # [B, P, model_dim]

        # Build adjacency and edge features once from the last frame
        adj = build_adjacency_torch(side, bc, pos, self.knn_k, self.topology)
        edge_feats = edge_features_torch(pos, vel) if self.use_edge_features else None

        for layer in self.gat_layers:
            x = layer(x, adj, edge_feats)

        x = squeeze(self.player_pooling_layer(x.permute(0, 2, 1)), -1)  # [B, model_dim]
        return self.decoder(x)


class STGNN_ST(nn.Module):
    """
    Space -> Time with GAT. GATLayers model interaction across the 22 players at EACH
    frame (shared weights, no pooling), then a shared single-layer GRU integrates each
    player's sequence of contextualized embeddings over time.
    Drop-in replacement for HybridST with GATLayer in place of plain TransformerEncoder.

    Input: [batch, T, 22, feature_len] -> [batch, 2]
    """

    def __init__(
        self,
        feature_len: int,
        model_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.3,
        window_length: int = 15,
        topology: str = "full",
        edge_features: bool = True,
        knn_k: int = 8,
    ):
        super().__init__()
        self.window_length = window_length
        self.topology = topology
        self.use_edge_features = edge_features
        self.knn_k = knn_k
        num_heads = min(16, max(2, 2 * round(model_dim / 64)))
        self.hyperparams = {
            "model_dim": model_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "window_length": window_length,
            "topology": topology,
            "edge_features": int(edge_features),
            "knn_k": knn_k,
        }
        self.feature_norm_layer = nn.BatchNorm1d(feature_len)
        self.feature_embedding_layer = nn.Sequential(
            nn.Linear(feature_len, model_dim),
            nn.ReLU(),
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
        )
        edge_dim = EDGE_FEATURE_DIM if edge_features else 0
        self.gat_layers = nn.ModuleList([GATLayer(model_dim, num_heads, dropout, edge_dim) for _ in range(num_layers)])
        self.gru = nn.GRU(input_size=model_dim, hidden_size=model_dim, num_layers=1, batch_first=True)
        self.player_pooling_layer = nn.AdaptiveAvgPool1d(1)
        self.decoder = _build_decoder(model_dim, dropout)

    def forward(self, x: Tensor) -> Tensor:
        B, T, P, F = x.size()

        x = self.feature_norm_layer(x.reshape(-1, F)).reshape(B, T, P, F)

        # Build adjacency per frame — fold time into batch, extract pos/vel/side/bc
        x_flat = x.reshape(B * T, P, F)
        pos  = x_flat[..., 0:2]
        vel  = x_flat[..., 2:4]
        side = x_flat[..., 6]
        bc   = x_flat[..., 7]
        adj = build_adjacency_torch(side, bc, pos, self.knn_k, self.topology)  # [B*T, P, P]
        edge_feats = edge_features_torch(pos, vel) if self.use_edge_features else None

        # Embed then GAT across players at each frame
        h = self.feature_embedding_layer(x_flat)  # [B*T, P, model_dim]
        for layer in self.gat_layers:
            h = layer(h, adj, edge_feats)

        # Per-player temporal integration
        h = h.reshape(B, T, P, -1).permute(0, 2, 1, 3).reshape(B * P, T, -1)
        _, h_n = self.gru(h)
        h = h_n[-1].reshape(B, P, -1)  # [B, P, model_dim]
        h = squeeze(self.player_pooling_layer(h.permute(0, 2, 1)), -1)  # [B, model_dim]
        return self.decoder(h)
    

class GraphModel(nn.Module):
    """GNN over the 22 players for single-frame tackle prediction.

    A generalization of SportsTransformer: choose the edge `topology` (full / bipartite /
    hub) and whether to feed explicit physical `edge_features`. Permutation-equivariant
    (shared weights + symmetric aggregation), mean-pooled to the shared decoder head.

    Input: [batch, 22, feature_len] (raw features) -> [batch, 2].
    """

    def __init__(
        self,
        feature_len: int,
        model_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        topology: str = "bipartite",
        edge_features: bool = True,
        knn_k: int = 4
    ):
        super().__init__()
        self.knn_k = knn_k
        self.topology = topology
        self.use_edge_features = edge_features
        num_heads = min(16, max(2, 2 * round(model_dim / 64)))
        self.hyperparams = {
            "model_dim": model_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "topology": topology,
            "edge_features": int(edge_features),
            "knn_k": knn_k,
        }
        self.feature_norm_layer = nn.BatchNorm1d(feature_len)
        self.feature_embedding_layer = nn.Sequential(
            nn.Linear(feature_len, model_dim), nn.ReLU(), nn.LayerNorm(model_dim), nn.Dropout(dropout)
        )
        edge_dim = EDGE_FEATURE_DIM if edge_features else 0
        self.layers = nn.ModuleList([GATLayer(model_dim, num_heads, dropout, edge_dim) for _ in range(num_layers)])
        self.decoder = _build_decoder(model_dim, dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, 22, F] raw features; RAW_FEATURES = [x_rel, y_rel, vx, vy, ox, oy, side, is_ball_carrier]
        pos, vel, side, bc = x[..., 0:2], x[..., 2:4], x[..., 6], x[..., 7]
        adj = build_adjacency_torch(side, bc, pos, self.knn_k, self.topology)
        edge_feats = edge_features_torch(pos, vel) if self.use_edge_features else None
        h = self.feature_norm_layer(x.transpose(1, 2)).transpose(1, 2)
        h = self.feature_embedding_layer(h)
        for layer in self.layers:
            h = layer(h, adj, edge_feats)
        return self.decoder(h.mean(dim=1))


class LitModel(LightningModule):
    """
    Lightning module for training and evaluating tackle prediction models.
    """

    def __init__(
        self,
        model_type: str,
        batch_size: int,
        model_dim: int,
        num_layers: int,
        feature_len: int,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        window_length: int = 1,
        topology: str= "full",
        edge_features: bool = True,
        knn_k: int = 4
    ):
        """
        Initialize the LitModel.

        Args:
            model_type (str): Type of model ('transformer', 'zoo', or a temporal type:
                'windowed_transformer', 'pure_gru', 'hybrid_ts', 'hybrid_st').
            batch_size (int): Batch size for training and evaluation.
            model_dim (int): Dimension of the model's internal representations.
            num_layers (int): Number of layers in the model.
            feature_len (int): Number of input features per player (transformer) or per interaction (zoo).
            dropout (float): Dropout rate for regularization.
            learning_rate (float): Learning rate for the optimizer.
            window_length (int): Temporal window length T (temporal models only).
        """
        super().__init__()
        self.model_type = model_type.lower()
        model_classes = {
            "transformer": SportsTransformer,
            "zoo": TheZooArchitecture,
            "windowed_transformer": WindowedTransformer,
            "pure_gru": PureGRU,
            "hybrid_ts": HybridTS,
            "hybrid_st": HybridST,
            "stgnn_ts": STGNN_TS,
            "stgnn_st": STGNN_ST
        }
        self.model_class = model_classes.get(self.model_type, TheZooArchitecture)
        self.feature_len = feature_len
        self.window_length = window_length
        self.is_temporal = self.model_type in TEMPORAL_MODEL_TYPES

        # Initialize model with architecture-specific parameters
        STGNN_TYPES = ["stgnn_ts", "stgnn_st"]
        if self.is_temporal:
            if self.model_type in STGNN_TYPES:
                self.model = self.model_class(
                    feature_len=self.feature_len,
                    model_dim=model_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                    window_length=window_length,
                    topology=topology,
                    edge_features=edge_features,
                    knn_k=knn_k
                )
                self.example_input_array = torch.randn((batch_size, window_length, 22, self.feature_len))
            else:
                self.model = self.model_class(
                    feature_len=self.feature_len,
                    model_dim=model_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                )
                self.example_input_array = (
                    torch.randn((batch_size, 22, self.feature_len))
                    if self.model_type == "transformer"
                    else torch.randn((batch_size, 10, 11, self.feature_len))
                )

        self.learning_rate = learning_rate
        self.num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.hparams["params"] = self.num_params
        for k, v in self.model.hyperparams.items():
            self.hparams[k] = v

        self.save_hyperparameters()
        self.loss_fn = torch.nn.SmoothL1Loss()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return self.model(x)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        Perform a single training step.

        Args:
            batch (tuple[Tensor, Tensor]): Batch of input features and target locations.
            batch_idx (int): Index of the current batch.

        Returns:
            Tensor: Computed loss for the batch.
        """
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        Validation step for the model.

        Args:
            batch (tuple[Tensor, Tensor]): Batch of input and target tensors.
            batch_idx (int): Index of the current batch.

        Returns:
            Tensor: Computed loss.
        """
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        Test step for the model.

        Args:
            batch (tuple[Tensor, Tensor]): Batch of input and target tensors.
            batch_idx (int): Index of the current batch.

        Returns:
            Tensor: Computed loss.
        """
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        return loss

    def predict_step(self, batch: tuple[Tensor, Tensor], batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """
        Prediction step for the model.

        Args:
            batch (tuple[Tensor, Tensor]): Batch of input and target tensors.
            batch_idx (int): Index of the current batch.
            dataloader_idx (int): Index of the dataloader.

        Returns:
            Tensor: Predicted output tensor.
        """
        x, y = batch
        y_hat = self.model(x)
        return y_hat

    def configure_optimizers(self) -> AdamW:
        """
        Configure the optimizer for training.

        Returns:
            AdamW: Configured optimizer.
        """
        return AdamW(self.parameters(), lr=self.learning_rate)

    def get_hyperparams(self) -> dict[str, Any]:
        """
        Get the hyperparameters of the model.

        Returns:
            Dict[str, Any]: Dictionary of hyperparameters.
        """
        return self.hparams