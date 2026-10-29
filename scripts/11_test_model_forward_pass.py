"""
Smoke test: verify the Stage 5/6 model (model/encoder.py, model/heads.py)
actually runs end-to-end on real data, with real shapes -- not just that
the classes are syntactically correct. Same verification discipline used
for every other stage in this project: load real files, check real shapes,
check for NaNs, don't just trust that the code "looks right."

Does NOT train anything -- this only confirms the forward pass works.
"""
import json
from pathlib import Path

import pandas as pd
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import SpatioTemporalEncoder, FlightRepresentation, MultiTaskHeads, UncertaintyWeightedLoss, CAUSE_TO_IDX

GRAPH_DIR = Path("./data_processed/graph")
ENRICHED_DIR = Path("./data_processed/monthly_enriched")

K = 12  # last 48 hours at 4-hour resolution, per the architecture spec's starting default


def main():
    torch.manual_seed(0)

    with open(GRAPH_DIR / "sequence" / "airport_index.json") as f:
        airport_order = json.load(f)["airport_order"]
    airport_to_idx = {code: i for i, code in enumerate(airport_order)}
    print(f"Loaded fixed airport order: {len(airport_order)} airports", flush=True)

    print("\n=== Testing SpatioTemporalEncoder ===", flush=True)
    graphs = torch.load(GRAPH_DIR / "sequence" / "2023-07.pt", weights_only=False)
    print(f"Loaded 2023-07: {len(graphs)} snapshots", flush=True)
    snapshot_window = graphs[20:20 + K]
    assert len(snapshot_window) == K, f"expected {K} snapshots, got {len(snapshot_window)}"
    print(f"Using snapshots {snapshot_window[0].window_start} through {snapshot_window[-1].window_start}", flush=True)

    encoder = SpatioTemporalEncoder(node_feat_dim=10, hidden_dim=64, gat_layers=2, gat_heads=4, tcn_layers=3, edge_dim=2)
    encoder.eval()
    with torch.no_grad():
        airport_embeddings = encoder(snapshot_window)
    print(f"Encoder output shape: {tuple(airport_embeddings.shape)} (expected (200, 64))", flush=True)
    assert airport_embeddings.shape == (200, 64)
    assert not torch.isnan(airport_embeddings).any(), "NaN in encoder output"
    print("No NaNs in encoder output. PASS.", flush=True)

    print("\n=== Testing FlightRepresentation + MultiTaskHeads on real flights ===", flush=True)
    flights = pd.read_parquet(
        ENRICHED_DIR / "2023-07.parquet",
        columns=["ORIGIN", "DEST", "OP_UNIQUE_CARRIER", "ROTATION_PREV_ARR_DELAY",
                 "ROTATION_TURNAROUND_MIN", "CRS_ELAPSED_TIME", "DISTANCE",
                 "ROTATION_PREV_CAUSE", "ARR_DELAY", "ARR_DEL15"],
    )
    flights = flights[flights["ORIGIN"].isin(airport_to_idx) & flights["DEST"].isin(airport_to_idx)]
    sample = flights.sample(n=16, random_state=0).reset_index(drop=True)
    print(f"Sampled {len(sample)} real flights from 2023-07", flush=True)

    carriers = sorted(flights["OP_UNIQUE_CARRIER"].unique())
    carrier_to_idx = {c: i for i, c in enumerate(carriers)}

    origin_idx = torch.tensor([airport_to_idx[o] for o in sample["ORIGIN"]], dtype=torch.long)
    dest_idx = torch.tensor([airport_to_idx[d] for d in sample["DEST"]], dtype=torch.long)
    carrier_idx = torch.tensor([carrier_to_idx[c] for c in sample["OP_UNIQUE_CARRIER"]], dtype=torch.long)
    prev_cause_idx = torch.tensor([CAUSE_TO_IDX[c] for c in sample["ROTATION_PREV_CAUSE"]], dtype=torch.long)

    numeric_cols = sample[["ROTATION_PREV_ARR_DELAY", "ROTATION_TURNAROUND_MIN", "CRS_ELAPSED_TIME", "DISTANCE"]].fillna(0.0)
    flight_numeric = torch.tensor(numeric_cols.values, dtype=torch.float32)
    flight_numeric = (flight_numeric - flight_numeric.mean(dim=0)) / (flight_numeric.std(dim=0) + 1e-6)

    origin_emb = airport_embeddings[origin_idx]
    dest_emb = airport_embeddings[dest_idx]

    flight_repr_layer = FlightRepresentation(airport_emb_dim=64, n_carriers=len(carriers), carrier_emb_dim=8, cause_emb_dim=4, flight_numeric_dim=4)
    heads = MultiTaskHeads(input_dim=flight_repr_layer.output_dim, hidden_dim=64)

    flight_repr = flight_repr_layer(origin_emb, dest_emb, carrier_idx, prev_cause_idx, flight_numeric)
    print(f"Flight representation shape: {tuple(flight_repr.shape)} (expected ({len(sample)}, {flight_repr_layer.output_dim}))", flush=True)
    assert flight_repr.shape == (len(sample), flight_repr_layer.output_dim)

    outputs = heads(flight_repr)
    for name, tensor in outputs.items():
        print(f"  {name}: shape {tuple(tensor.shape)}, has NaN: {torch.isnan(tensor).any().item()}", flush=True)
        assert not torch.isnan(tensor).any(), f"NaN in {name}"

    print("\n=== Testing the combined multi-task loss (untrained, just checking it runs) ===", flush=True)
    true_delay = torch.tensor(sample["ARR_DELAY"].fillna(0.0).values, dtype=torch.float32)
    true_risk = torch.tensor(sample["ARR_DEL15"].fillna(0.0).values, dtype=torch.float32)

    loss_a = torch.nn.functional.mse_loss(outputs["delay_minutes"], true_delay)
    loss_b = torch.nn.functional.binary_cross_entropy_with_logits(outputs["delay_risk_logit"], true_risk)
    loss_c = torch.tensor(0.0)  # would be masked cross-entropy on delayed-only flights; skipped in this smoke test

    weighted_loss = UncertaintyWeightedLoss()
    total_loss = weighted_loss(loss_a, loss_b, loss_c)
    print(f"loss_a={loss_a.item():.3f}  loss_b={loss_b.item():.3f}  total (uncertainty-weighted)={total_loss.item():.3f}", flush=True)
    assert not torch.isnan(total_loss), "NaN in combined loss"

    print("\nALL CHECKS PASSED.", flush=True)


if __name__ == "__main__":
    main()
