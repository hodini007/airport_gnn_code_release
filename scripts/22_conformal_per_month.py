"""
Stage 7 follow-up: is the ~1-2 point conformal undercoverage (scripts/18_
conformal_prediction.py) spread evenly across the test period, or concentrated
in specific months?

Raised directly by an external review: reporting one aggregate coverage
number across the whole 6-month test period could hide a coverage failure
that's actually concentrated in a period with more distribution drift, which
matters for judging how serious the undercoverage limitation actually is.

Procedure: identical calibration step as script 18 (same reserved 2024-07..
2024-12 slice, same nonconformity scores, same finite-sample-corrected q),
but instead of evaluating that single q against the whole concatenated test
set, this script evaluates it separately against each of the six 2025-01..
2025-06 test months, so coverage and width can be compared month-by-month.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import (
    SpatioTemporalEncoder, SpatialOnlyEncoder, TemporalOnlyEncoder,
    SpatioTemporalAttentionEncoder, AdaptiveGraphSpatioTemporalEncoder,
    FlightRepresentation, MultiTaskHeads,
)
from model.dataset import MonthWindowSource, build_flight_tensors, load_airport_index, load_carrier_index

N_SELECTED_AIRPORTS = 200

ENCODER_BUILDERS = {
    "full": lambda: SpatioTemporalEncoder(node_feat_dim=10, hidden_dim=64, gat_layers=2, gat_heads=4, tcn_layers=3, edge_dim=2),
    "gat_only": lambda: SpatialOnlyEncoder(node_feat_dim=10, hidden_dim=64, gat_layers=2, gat_heads=4, edge_dim=2),
    "lstm_only": lambda: TemporalOnlyEncoder(node_feat_dim=10, hidden_dim=64, tcn_layers=3),
    "stpn_lite": lambda: SpatioTemporalAttentionEncoder(node_feat_dim=10, hidden_dim=64, gat_layers=2, gat_heads=4, edge_dim=2, attn_heads=4),
    "mstagcn_lite": lambda: AdaptiveGraphSpatioTemporalEncoder(num_nodes=N_SELECTED_AIRPORTS, node_feat_dim=10, hidden_dim=64, embed_dim=16, spatial_layers=2, tcn_layers=3),
}

CHECKPOINT_DIR = Path("./data_processed/checkpoints")
CALIB_MONTHS = [f"2024-{m:02d}" for m in range(7, 13)]
TEST_MONTHS = [f"2025-{m:02d}" for m in range(1, 7)]
K = 12
ALPHA = 0.10  # 90% target coverage -- the headline level from script 18


def load_model(checkpoint_path, encoder_name, device):
    airport_to_idx = load_airport_index()
    carrier_to_idx = load_carrier_index()
    encoder = ENCODER_BUILDERS[encoder_name]().to(device)
    flight_repr_layer = FlightRepresentation(airport_emb_dim=64, n_carriers=len(carrier_to_idx), carrier_emb_dim=8, cause_emb_dim=4, flight_numeric_dim=4).to(device)
    heads = MultiTaskHeads(input_dim=flight_repr_layer.output_dim, hidden_dim=64).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["encoder"])
    flight_repr_layer.load_state_dict(ckpt["flight_repr_layer"])
    heads.load_state_dict(ckpt["heads"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}", flush=True)
    encoder.eval(); flight_repr_layer.eval(); heads.eval()
    return encoder, flight_repr_layer, heads, airport_to_idx, carrier_to_idx


def run_inference(encoder, flight_repr_layer, heads, airport_to_idx, carrier_to_idx, months, device):
    pred_delay, true_delay, pred_risk_prob, true_risk = [], [], [], []
    with torch.no_grad():
        for month_key in months:
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
                pred_delay.append(outputs["delay_minutes"].cpu().numpy())
                true_delay.append(batch["true_delay"].cpu().numpy())
                pred_risk_prob.append(torch.sigmoid(outputs["delay_risk_logit"]).cpu().numpy())
                true_risk.append(batch["true_risk"].cpu().numpy())
                n_windows_used += 1
            print(f"  {month_key}: {n_windows_used} windows evaluated", flush=True)
    return (
        np.concatenate(pred_delay), np.concatenate(true_delay),
        np.concatenate(pred_risk_prob), np.concatenate(true_risk),
    )


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))


def evaluate_head_a(pred, true, q):
    lower, upper = pred - q, pred + q
    covered = (true >= lower) & (true <= upper)
    return {"empirical_coverage": float(covered.mean()), "mean_interval_width_minutes": float(2 * q)}


def evaluate_head_b(pred_prob, true_risk, q):
    in_delayed = pred_prob >= (1 - q)
    in_not_delayed = (1 - pred_prob) >= (1 - q)
    set_size = in_delayed.astype(int) + in_not_delayed.astype(int)
    true_class_in_set = np.where(true_risk == 1, in_delayed, in_not_delayed)
    return {"empirical_coverage": float(true_class_in_set.mean()), "mean_set_size": float(set_size.mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--encoder", choices=list(ENCODER_BUILDERS), default="full")
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Checkpoint: {args.checkpoint}  (encoder={args.encoder})", flush=True)

    encoder, flight_repr_layer, heads, airport_to_idx, carrier_to_idx = load_model(args.checkpoint, args.encoder, device)

    print(f"\nCalibrating on {CALIB_MONTHS} ...", flush=True)
    calib_pred_delay, calib_true_delay, calib_pred_risk, calib_true_risk = run_inference(
        encoder, flight_repr_layer, heads, airport_to_idx, carrier_to_idx, CALIB_MONTHS, device
    )
    nonconformity_a = np.abs(calib_pred_delay - calib_true_delay)
    prob_true_class = np.where(calib_true_risk == 1, calib_pred_risk, 1 - calib_pred_risk)
    nonconformity_b = 1 - prob_true_class
    q_a = conformal_quantile(nonconformity_a, ALPHA)
    q_b = conformal_quantile(nonconformity_b, ALPHA)
    print(f"q_a={q_a:.3f} min, q_b={q_b:.3f} (alpha={ALPHA}, target coverage {1-ALPHA:.0%})", flush=True)

    print(f"\nEvaluating each test month separately against this single, aggregate-calibrated q ...", flush=True)
    per_month = {}
    for month in TEST_MONTHS:
        pred_delay, true_delay, pred_risk, true_risk = run_inference(
            encoder, flight_repr_layer, heads, airport_to_idx, carrier_to_idx, [month], device
        )
        a = evaluate_head_a(pred_delay, true_delay, q_a)
        b = evaluate_head_b(pred_risk, true_risk, q_b)
        per_month[month] = {"n_flights": int(len(true_delay)), "head_a": a, "head_b": b}
        print(f"  {month}: n={len(true_delay):,}  Head A coverage={a['empirical_coverage']:.4f} width={a['mean_interval_width_minutes']:.2f}  "
              f"Head B coverage={b['empirical_coverage']:.4f} set_size={b['mean_set_size']:.3f}", flush=True)

    coverages_a = [v["head_a"]["empirical_coverage"] for v in per_month.values()]
    coverages_b = [v["head_b"]["empirical_coverage"] for v in per_month.values()]
    print("\n" + "=" * 70)
    print(f"Head A coverage across months: min={min(coverages_a):.4f} max={max(coverages_a):.4f} "
          f"range={max(coverages_a)-min(coverages_a):.4f}")
    print(f"Head B coverage across months: min={min(coverages_b):.4f} max={max(coverages_b):.4f} "
          f"range={max(coverages_b)-min(coverages_b):.4f}")
    print("=" * 70)

    report = {
        "checkpoint": args.checkpoint, "encoder": args.encoder,
        "alpha": ALPHA, "target_coverage": 1 - ALPHA,
        "q_a": q_a, "q_b": q_b,
        "per_month": per_month,
        "head_a_coverage_range": {"min": min(coverages_a), "max": max(coverages_a)},
        "head_b_coverage_range": {"min": min(coverages_b), "max": max(coverages_b)},
    }
    out_path = CHECKPOINT_DIR / f"conformal_per_month_report_{args.tag}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
