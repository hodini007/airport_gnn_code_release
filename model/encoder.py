"""
Stage 5: Spatiotemporal encoder.

Implements Model_Architecture_Specification.docx Section 4, Steps 1-2:
a Graph Attention Network (spatial) followed by a Temporal Convolutional
Network (temporal), applied to the graph snapshot sequence built in
Stage 4 (data_processed/graph/sequence/*.pt).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class SpatialEncoder(nn.Module):
    """
    Step 1: GAT, applied independently to one graph snapshot at a time.
    Each airport's output embedding incorporates its directly connected
    neighbors, weighted by learned attention -- these attention weights
    double as an interpretability signal (methodology Stage 5).

    Defaults (hidden_dim=64, num_layers=2, heads=4) are starting points to
    tune empirically, not fixed requirements -- see
    Model_Architecture_Specification.docx Section 4.
    """

    def __init__(self, in_channels=10, hidden_channels=64, num_layers=2,
                 heads=4, edge_dim=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            # concat=False averages the attention heads instead of
            # concatenating them, so hidden size doesn't grow with head count.
            self.layers.append(
                GATConv(in_ch, hidden_channels, heads=heads, concat=False,
                         edge_dim=edge_dim, dropout=dropout, add_self_loops=True)
            )
            self.norms.append(nn.LayerNorm(hidden_channels))
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        """x: [num_nodes, in_channels]. Returns [num_nodes, hidden_channels]."""
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            residual = x if i > 0 else None  # first layer changes dimensionality, no residual
            x = layer(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = self.activation(x)
            x = self.dropout(x)
            if residual is not None:
                x = x + residual
        return x


class TemporalBlock(nn.Module):
    """One causal, dilated 1D convolution block (a single TCN layer)."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation  # pad left only -> causal (no future leakage)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=self.padding, dilation=dilation)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        """x: [batch, channels, time]."""
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]  # chomp the extra right-side padding to stay causal
        out = self.dropout(self.activation(self.norm(out)))
        residual = x if self.downsample is None else self.downsample(x)
        return self.activation(out + residual)


class TemporalEncoder(nn.Module):
    """
    Step 2: TCN. Takes one airport's spatial embeddings from the last K
    snapshots (K=12 by default -- the last 48 hours at 4-hour resolution,
    a starting default per the architecture spec) and produces one
    embedding that also encodes the recent trend, not just the current
    moment.
    """

    def __init__(self, channels=64, num_layers=3, kernel_size=3, dropout=0.1):
        super().__init__()
        blocks = []
        for i in range(num_layers):
            dilation = 2 ** i  # 1, 2, 4, ... -- widening receptive field per layer
            blocks.append(TemporalBlock(channels, channels, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*blocks)

    def forward(self, x):
        """x: [num_nodes, channels, K]. Returns [num_nodes, channels] (final time step)."""
        out = self.network(x)
        return out[:, :, -1]


class SpatialOnlyEncoder(nn.Module):
    """
    "GAT-only" ablation baseline (Methodology_Refinements_PreImplementation.docx
    Section 3 baseline table): drops the TCN entirely and uses only the most
    recent snapshot's spatial embedding, discarding the other K-1 snapshots
    of history. Tests whether the temporal module is earning its complexity,
    i.e. whether recent trend adds anything beyond the current moment.
    """

    def __init__(self, node_feat_dim=10, hidden_dim=64, gat_layers=2, gat_heads=4, edge_dim=2, dropout=0.1):
        super().__init__()
        self.spatial = SpatialEncoder(node_feat_dim, hidden_dim, gat_layers, gat_heads, edge_dim, dropout)
        self.hidden_dim = hidden_dim

    def forward(self, snapshots):
        """snapshots: list of K PyG Data objects; only the last is used."""
        latest = snapshots[-1]
        return self.spatial(latest.x, latest.edge_index, latest.edge_attr)


class TemporalOnlyEncoder(nn.Module):
    """
    "LSTM-only" ablation baseline (same section): drops the GAT spatial
    module entirely. Each airport's own K-snapshot feature history is
    projected to hidden_dim and run through the TCN with zero cross-node
    (graph) information sharing -- tests whether spatial structure adds
    anything beyond an airport's own history.

    Uses the TCN (not a literal LSTM) to stay consistent with the
    TCN-over-LSTM/Transformer decision already made for the main model
    (Model_Architecture_Specification.docx) -- "LSTM-only" is the role
    inherited from the original baseline table (temporal-only ablation),
    not the literal layer type.
    """

    def __init__(self, node_feat_dim=10, hidden_dim=64, tcn_layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.temporal = TemporalEncoder(hidden_dim, tcn_layers, dropout=dropout)
        self.hidden_dim = hidden_dim

    def forward(self, snapshots):
        per_snapshot = [self.input_proj(data.x) for data in snapshots]
        stacked = torch.stack(per_snapshot, dim=-1)  # [num_airports, hidden_dim, K]
        return self.temporal(stacked)


class AttentionTemporalEncoder(nn.Module):
    """
    Multi-head self-attention over the K time steps, replacing the TCN's
    causal convolution -- the temporal mechanism shared by STPN (Wu et al.
    2024, IEEE TKDE) and SGDAN (Guo et al. 2020), both of which use
    attention rather than convolution to aggregate an airport's own
    history. A causal mask (can't attend to the future) keeps this a fair
    comparison against the TCN, which is also causal.
    """

    def __init__(self, channels=64, num_heads=4, max_len=64, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, channels))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x):
        """x: [num_nodes, channels, K]. Returns [num_nodes, channels] (last time step)."""
        x = x.transpose(1, 2)  # [num_nodes, K, channels]
        K = x.shape[1]
        x = x + self.pos_embedding[:, :K, :]
        causal_mask = torch.triu(torch.ones(K, K, device=x.device, dtype=torch.bool), diagonal=1)
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask)
        attn_out = self.norm(attn_out + x)
        return attn_out[:, -1, :]


class SpatioTemporalAttentionEncoder(nn.Module):
    """
    "STPN/SGDAN-lite" baseline: the same GAT spatial encoder as the main
    model, but with the TCN replaced by AttentionTemporalEncoder --
    isolating the one mechanism STPN and SGDAN share (attention-based
    temporal aggregation instead of convolution) for a fast, honest check
    before committing to a full reimplementation of either paper.

    Not a literal reimplementation of either: STPN's multi-relational
    graph (geography + weather + traffic volume, github.com/Kaimaoge/STPN)
    needs weather/traffic data this project doesn't have, and SGDAN
    operates on a heterogeneous per-flight graph rather than per-airport
    snapshots. This tests their shared core idea, not their full designs.
    """

    def __init__(self, node_feat_dim=10, hidden_dim=64, gat_layers=2, gat_heads=4, edge_dim=2, attn_heads=4, dropout=0.1):
        super().__init__()
        self.spatial = SpatialEncoder(node_feat_dim, hidden_dim, gat_layers, gat_heads, edge_dim, dropout)
        self.temporal = AttentionTemporalEncoder(hidden_dim, attn_heads, dropout=dropout)
        self.hidden_dim = hidden_dim

    def forward(self, snapshots):
        per_snapshot_embeddings = [
            self.spatial(data.x, data.edge_index, data.edge_attr) for data in snapshots
        ]
        stacked = torch.stack(per_snapshot_embeddings, dim=-1)
        return self.temporal(stacked)


class AdaptiveSpatialEncoder(nn.Module):
    """
    "MSTAGCN-lite" spatial component: a learned, self-adaptive adjacency
    matrix (per-node embeddings E1, E2; edge weight = softmax(relu(E1 @
    E2^T))) instead of the fixed, EDA-derived graph -- approximating
    MSTAGCN's (Cai et al. 2022, IEEE T-ITS) "adaptive graph convolution"
    idea, which learns connectivity end-to-end rather than using a fixed
    edge weight. Uses simple learned-adjacency-weighted aggregation
    (not GAT's attention) since the adjacency itself is the learned part.
    """

    def __init__(self, num_nodes, node_feat_dim=10, hidden_dim=64, embed_dim=16, num_layers=2, dropout=0.1):
        super().__init__()
        self.node_embed_1 = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.01)
        self.node_embed_2 = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.01)
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def adaptive_adj(self):
        logits = F.relu(self.node_embed_1 @ self.node_embed_2.t())
        return F.softmax(logits, dim=-1)  # [N, N], row-normalized

    def forward(self, x):
        """x: [num_nodes, node_feat_dim] (one snapshot). Returns [num_nodes, hidden_dim]."""
        adj = self.adaptive_adj()
        h = self.input_proj(x)
        for layer, norm in zip(self.layers, self.norms):
            h = adj @ h
            h = layer(h)
            h = norm(h)
            h = self.activation(h)
            h = self.dropout(h)
        return h


class AdaptiveGraphSpatioTemporalEncoder(nn.Module):
    """
    "MSTAGCN-lite": AdaptiveSpatialEncoder (learned graph, per snapshot) +
    the existing TCN for temporal aggregation -- isolates MSTAGCN's core
    "adaptive/learned graph" idea against our TCN, checking whether a
    data-driven graph structure captures more than our fixed, EDA-derived
    one before committing to a full reimplementation.
    """

    def __init__(self, num_nodes, node_feat_dim=10, hidden_dim=64, embed_dim=16, spatial_layers=2, tcn_layers=3, dropout=0.1):
        super().__init__()
        self.spatial = AdaptiveSpatialEncoder(num_nodes, node_feat_dim, hidden_dim, embed_dim, spatial_layers, dropout)
        self.temporal = TemporalEncoder(hidden_dim, tcn_layers, dropout=dropout)
        self.hidden_dim = hidden_dim

    def forward(self, snapshots):
        per_snapshot_embeddings = [self.spatial(data.x) for data in snapshots]
        stacked = torch.stack(per_snapshot_embeddings, dim=-1)
        return self.temporal(stacked)


class SpatioTemporalEncoder(nn.Module):
    """
    Combines SpatialEncoder + TemporalEncoder into the full Stage 5
    encoder: a sequence of K graph snapshots in, one space+time-aware
    embedding per airport out.
    """

    def __init__(self, node_feat_dim=10, hidden_dim=64, gat_layers=2, gat_heads=4,
                 tcn_layers=3, edge_dim=2, dropout=0.1):
        super().__init__()
        self.spatial = SpatialEncoder(node_feat_dim, hidden_dim, gat_layers, gat_heads, edge_dim, dropout)
        self.temporal = TemporalEncoder(hidden_dim, tcn_layers, dropout=dropout)
        self.hidden_dim = hidden_dim

    def forward(self, snapshots):
        """
        snapshots: list of K PyG Data objects, oldest to newest, each using
        the SAME fixed 200-airport node ordering (Stage 4's
        data_processed/graph/sequence/airport_index.json).

        Returns: [num_airports, hidden_dim].
        """
        per_snapshot_embeddings = [
            self.spatial(data.x, data.edge_index, data.edge_attr) for data in snapshots
        ]
        stacked = torch.stack(per_snapshot_embeddings, dim=-1)  # [num_airports, hidden_dim, K]
        return self.temporal(stacked)
