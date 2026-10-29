"""
Diagnostic for the ablation-baseline surprise in scripts/12_train_model.py /
README.md: LSTM-only (no cross-airport information sharing at all) beat both
the full spatiotemporal model and GAT-only on every test metric. Two
undistinguished explanations were proposed -- (1) GAT attention has nothing
real to learn given how rich the node features already are, or (2) the full
model just needs more than 15 epochs to exploit its larger GAT compute.

This directly tests (1): load the full model's trained GAT layers and
inspect the actual attention weights learned, on real busy (well-connected)
snapshots. If attention has collapsed to near-uniform (every neighbor gets
roughly 1/degree regardless of identity), GAT has learned nothing beyond
"average my neighbors" -- consistent with explanation 1. If attention shows
real, non-trivial spread (some neighbors consistently favored over others),
GAT has learned real structure, which points toward explanation 2 instead
(the structure is there but isn't translating to better predictions yet,
e.g. due to training budget).
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.encoder import SpatioTemporalEncoder

CHECKPOINT_DIR = Path("./data_processed/checkpoints")
SEQUENCE_DIR = Path("./data_processed/graph/sequence")


def entropy(p):
    """Shannon entropy in nats of a probability vector (or batch of them)."""
    p = p.clamp_min(1e-12)
    return -(p * p.log()).sum(-1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(0)
    random_encoder = SpatioTemporalEncoder(node_feat_dim=10, hidden_dim=64, gat_layers=2, gat_heads=4, tcn_layers=3, edge_dim=2).to(device)
    random_encoder.eval()

    encoder = SpatioTemporalEncoder(node_feat_dim=10, hidden_dim=64, gat_layers=2, gat_heads=4, tcn_layers=3, edge_dim=2).to(device)
    ckpt = torch.load(CHECKPOINT_DIR / "latest.pt", map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    print(f"Loaded full-model encoder from epoch {ckpt['epoch']}", flush=True)

    snapshots = torch.load(SEQUENCE_DIR / "2025-01.pt", weights_only=False)

    # Pick the 20 best-connected snapshots in the month (highest mean in-degree)
    # -- exactly the case where GAT has the most opportunity to show real structure.
    degrees = []
    for g in snapshots:
        deg = torch.bincount(g.edge_index[1], minlength=g.x.shape[0]).float()
        degrees.append(deg.mean().item())
    busiest_idx = np.argsort(degrees)[-20:]
    print(f"Using the 20 busiest snapshots (mean in-degree {np.mean([degrees[i] for i in busiest_idx]):.2f})", flush=True)

    def measure(enc):
        layer1_entropies, layer1_uniform_entropies = [], []
        layer2_entropies, layer2_uniform_entropies = [], []
        with torch.no_grad():
            for idx in busiest_idx:
                g = snapshots[idx].to(device)
                x, edge_index, edge_attr = g.x, g.edge_index, g.edge_attr

                out, (ei1, alpha1) = enc.spatial.layers[0](
                    x, edge_index, edge_attr=edge_attr, return_attention_weights=True
                )
                alpha1 = alpha1.mean(dim=-1)  # average the 4 heads -> [num_edges]
                x1 = enc.spatial.norms[0](out)
                x1 = enc.spatial.activation(x1)

                # Layer 2 -- reuses the ORIGINAL edge_index/edge_attr, matching
                # SpatialEncoder.forward exactly (it does not reuse the
                # self-loop-augmented edge_index GATConv returns internally).
                out2, (ei2, alpha2) = enc.spatial.layers[1](
                    x1, edge_index, edge_attr=edge_attr, return_attention_weights=True,
                )
                alpha2 = alpha2.mean(dim=-1)

                for ei, alpha, ent_list, unif_list in [
                    (ei1, alpha1, layer1_entropies, layer1_uniform_entropies),
                    (ei2, alpha2, layer2_entropies, layer2_uniform_entropies),
                ]:
                    dst = ei[1]
                    # group attention weights by destination node, compute entropy of each
                    # node's incoming-edge attention distribution, and compare to the uniform
                    # baseline (1/in-degree) entropy for the SAME degree -- the fair comparison,
                    # since a node with only 1 neighbor has zero entropy regardless of intent.
                    for node in dst.unique().tolist():
                        mask = dst == node
                        w = alpha[mask]
                        if w.numel() < 2:
                            continue  # entropy is trivially 0 with <2 neighbors either way
                        w = w / w.sum()
                        ent_list.append(entropy(w).item())
                        unif_list.append(float(np.log(w.numel())))  # max possible entropy = uniform
        return (layer1_entropies, layer1_uniform_entropies), (layer2_entropies, layer2_uniform_entropies)

    def summarize(name, ents, unif_ents):
        ents = np.array(ents)
        unif_ents = np.array(unif_ents)
        ratio = ents / np.maximum(unif_ents, 1e-12)  # 1.0 = fully uniform (collapsed), 0.0 = fully peaked on one neighbor
        print(f"  {name}: {len(ents)} distributions, mean entropy ratio = {ratio.mean():.4f}  (p10={np.percentile(ratio,10):.4f} p50={np.percentile(ratio,50):.4f} p90={np.percentile(ratio,90):.4f})")
        return ratio.mean()

    print("\n=== TRAINED full model (epoch 15) ===")
    (l1e, l1u), (l2e, l2u) = measure(encoder)
    trained_l1 = summarize("Layer 1", l1e, l1u)
    trained_l2 = summarize("Layer 2", l2e, l2u)

    print("\n=== UNTRAINED (random-init, seed=0) control ===")
    (l1e, l1u), (l2e, l2u) = measure(random_encoder)
    random_l1 = summarize("Layer 1", l1e, l1u)
    random_l2 = summarize("Layer 2", l2e, l2u)

    print(f"\nLayer 1: trained ratio {trained_l1:.4f} vs. random-init ratio {random_l1:.4f} (lower = more selective; a big gap means training taught GAT real selectivity, not just architectural noise)")
    print(f"Layer 2: trained ratio {trained_l2:.4f} vs. random-init ratio {random_l2:.4f}")


if __name__ == "__main__":
    main()
