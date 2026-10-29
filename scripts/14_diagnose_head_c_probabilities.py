"""
Diagnostic for the SECURITY_DELAY blind spot found by scripts/13_evaluate_model.py:
argmax never predicts SECURITY_DELAY even after 8x class weighting, but that
alone doesn't tell us whether the model carries *some* signal for it that just
never wins the softmax, or genuinely none at all. That distinction decides
whether oversampling/focal loss (amplify existing signal) or accepting the
limitation (no signal to amplify) is the right next step.

For every truly-delayed test flight, records the full 5-class softmax
probability vector, then compares P(security) and P(weather) on flights whose
true cause IS that class against the population baseline.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import SpatioTemporalEncoder, FlightRepresentation, MultiTaskHeads
from model.dataset import MonthWindowSource, build_flight_tensors, load_airport_index, load_carrier_index
from model.heads import CAUSE_VOCAB

CHECKPOINT_DIR = Path("./data_processed/checkpoints")
TEST_MONTHS = [f"2025-{m:02d}" for m in range(1, 7)]
K = 12
SECURITY_IDX = CAUSE_VOCAB.index("SECURITY_DELAY")
WEATHER_IDX = CAUSE_VOCAB.index("WEATHER_DELAY")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    airport_to_idx = load_airport_index()
    carrier_to_idx = load_carrier_index()

    encoder = SpatioTemporalEncoder(node_feat_dim=10, hidden_dim=64, gat_layers=2, gat_heads=4, tcn_layers=3, edge_dim=2).to(device)
    flight_repr_layer = FlightRepresentation(airport_emb_dim=64, n_carriers=len(carrier_to_idx), carrier_emb_dim=8, cause_emb_dim=4, flight_numeric_dim=4).to(device)
    heads = MultiTaskHeads(input_dim=flight_repr_layer.output_dim, hidden_dim=64).to(device)

    ckpt_path = CHECKPOINT_DIR / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["encoder"])
    flight_repr_layer.load_state_dict(ckpt["flight_repr_layer"])
    heads.load_state_dict(ckpt["heads"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}", flush=True)

    encoder.eval()
    flight_repr_layer.eval()
    heads.eval()

    all_probs, all_true_cause = [], []

    with torch.no_grad():
        for month_key in TEST_MONTHS:
            source = MonthWindowSource(month_key, K=K)
            n_windows_used = 0
            for window_idx in range(source.n_windows()):
                result = source.get_window(window_idx)
                if result is None:
                    continue
                snapshots, flights_df = result
                if len(flights_df) == 0:
                    continue
                batch = build_flight_tensors(flights_df, airport_to_idx, carrier_to_idx, device=device)
                if batch is None:
                    continue

                snapshots = [s.to(device) for s in snapshots]
                airport_embeddings = encoder(snapshots)
                origin_emb = airport_embeddings[batch["origin_idx"]]
                dest_emb = airport_embeddings[batch["dest_idx"]]
                flight_repr = flight_repr_layer(
                    origin_emb, dest_emb, batch["carrier_idx"], batch["prev_cause_idx"], batch["flight_numeric"]
                )
                outputs = heads(flight_repr)

                has_delayed = batch["cause_label"] != -1
                if has_delayed.any():
                    probs = F.softmax(outputs["cause_logits"][has_delayed], dim=-1).cpu().numpy()
                    all_probs.append(probs)
                    all_true_cause.append(batch["cause_label"][has_delayed].cpu().numpy())

                n_windows_used += 1
            print(f"  {month_key}: {n_windows_used} windows evaluated", flush=True)

    probs = np.concatenate(all_probs)          # [N, 5]
    true_cause = np.concatenate(all_true_cause)  # [N]

    print(f"\nTotal delayed test flights: {len(true_cause):,}", flush=True)
    print("\n" + "=" * 70)
    print("Population baseline P(class) averaged over ALL delayed test flights:")
    for i, name in enumerate(CAUSE_VOCAB[:5]):
        print(f"  {name:22s} mean P = {probs[:, i].mean():.4f}")

    for target_idx, target_name in [(SECURITY_IDX, "SECURITY_DELAY"), (WEATHER_IDX, "WEATHER_DELAY")]:
        mask = true_cause == target_idx
        n = mask.sum()
        print("\n" + "-" * 70)
        print(f"On the {n:,} flights whose TRUE cause is {target_name}:")
        if n == 0:
            print("  (no examples in this test set slice)")
            continue
        p_target = probs[mask, target_idx]
        print(f"  mean P({target_name})    = {p_target.mean():.4f}  (population baseline: {probs[:, target_idx].mean():.4f})")
        print(f"  median P({target_name})  = {np.median(p_target):.4f}")
        print(f"  max P({target_name})     = {p_target.max():.4f}")
        print(f"  P({target_name}) > 0.5 on {int((p_target > 0.5).sum())}/{n} flights")
        print(f"  P({target_name}) > 0.2 on {int((p_target > 0.2).sum())}/{n} flights")
        # what does the model predict instead, on average, for these flights?
        mean_probs_when_true = probs[mask].mean(axis=0)
        print(f"  Average full probability vector on these flights:")
        for i, name in enumerate(CAUSE_VOCAB[:5]):
            marker = "  <-- true class" if i == target_idx else ""
            print(f"    {name:22s} {mean_probs_when_true[i]:.4f}{marker}")

    print("=" * 70)


if __name__ == "__main__":
    main()
