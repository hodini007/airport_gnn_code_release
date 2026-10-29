"""
Stage 7: Split conformal prediction for Head A (delay regression) and Head B
(binary delay-risk), per Model_Architecture_Specification.docx Section 5.

Hand-implemented split conformal (chosen there over MAPIE/crepes, which expect
a scikit-learn-style estimator interface incompatible with a custom multi-task
PyTorch model). Procedure:

  1. Run the trained model on the reserved calibration slice (2024-07..2024-12,
     never used in training or validation -- see scripts/12_train_model.py)
     and record a nonconformity score per flight:
       Head A: |predicted_delay_minutes - true_delay_minutes|
       Head B: 1 - predicted_probability_of_the_true_class
  2. q = the finite-sample-corrected (1-alpha) quantile of those scores --
     ceil((n+1)(1-alpha))/n, clipped to 1.0 -- which is what gives split
     conformal its marginal coverage guarantee (the spec's "90th percentile"
     description is the alpha=0.10 case of this).
  3. On the held-out test set (2025-01..2025-06, untouched by calibration):
       Head A: report [pred - q, pred + q]; check whether true_delay falls inside.
       Head B: report {class : predicted_prob(class) >= 1 - q}; check whether
       the true class is in that set (can be {not_delayed}, {delayed}, both,
       or -- for alpha > 0.5 -- empty; this is expected split-conformal
       behavior, not a bug).
  4. Report empirical coverage (should sit near 1-alpha) and average interval
     width / set size for each alpha.

Head C is explicitly out of scope for this initial build -- ProtoCP-style
per-cause calibration is a noted future enhancement (Model_Architecture_
Specification.docx Section 5), not part of the initial build.

Encoder-agnostic by design (Literature_Review_Novelty_Q1_Assessment.docx
Section 7.2): run once per --encoder/--checkpoint pair, on both the proposed
full model and the LSTM-only ablation, and compare coverage/efficiency.
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

    encoder.eval()
    flight_repr_layer.eval()
    heads.eval()
    return encoder, flight_repr_layer, heads, airport_to_idx, carrier_to_idx


def run_inference(encoder, flight_repr_layer, heads, airport_to_idx, carrier_to_idx, months, device):
    """Returns per-flight arrays: predicted delay, true delay, predicted risk prob, true risk."""
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
    """Finite-sample-corrected (1-alpha) quantile giving split conformal its
    marginal coverage guarantee: ceil((n+1)(1-alpha))/n, clipped to 1.0."""
    n = len(scores)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))


def evaluate_head_a(pred, true, q):
    lower, upper = pred - q, pred + q
    covered = (true >= lower) & (true <= upper)
    return {
        "q": q,
        "empirical_coverage": float(covered.mean()),
        "mean_interval_width_minutes": float(2 * q),
    }


def evaluate_head_b(pred_prob, true_risk, q):
    # Prediction set membership: include "delayed" if P(delayed) >= 1-q;
    # include "not_delayed" if P(not_delayed) = 1-P(delayed) >= 1-q.
    in_delayed = pred_prob >= (1 - q)
    in_not_delayed = (1 - pred_prob) >= (1 - q)
    set_size = in_delayed.astype(int) + in_not_delayed.astype(int)

    true_class_in_set = np.where(true_risk == 1, in_delayed, in_not_delayed)
    return {
        "q": q,
        "empirical_coverage": float(true_class_in_set.mean()),
        "mean_set_size": float(set_size.mean()),
        "empty_set_rate": float((set_size == 0).mean()),
        "singleton_delayed_rate": float((in_delayed & ~in_not_delayed).mean()),
        "singleton_not_delayed_rate": float((in_not_delayed & ~in_delayed).mean()),
        "both_classes_rate": float((in_delayed & in_not_delayed).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--encoder", choices=list(ENCODER_BUILDERS), default="full")
    parser.add_argument("--alphas", type=float, nargs="*", default=[0.10, 0.05], help="1-alpha is the target coverage level, e.g. 0.10 -> 90%% coverage.")
    parser.add_argument("--tag", required=True, help="Label for this run in the output report filename, e.g. 'full' or 'lstm_only'.")
    parser.add_argument("--calib-months", nargs="*", default=None, help="Override calibration months (verification mode). Default: the reserved 2024-07..2024-12 slice.")
    parser.add_argument("--test-months", nargs="*", default=None, help="Override test months (verification mode). Default: the held-out 2025-01..2025-06 slice.")
    args = parser.parse_args()
    calib_months = args.calib_months if args.calib_months else CALIB_MONTHS
    test_months = args.test_months if args.test_months else TEST_MONTHS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Checkpoint: {args.checkpoint}  (encoder={args.encoder})", flush=True)

    encoder, flight_repr_layer, heads, airport_to_idx, carrier_to_idx = load_model(args.checkpoint, args.encoder, device)

    print(f"\nRunning inference on calibration months {calib_months} ...", flush=True)
    calib_pred_delay, calib_true_delay, calib_pred_risk, calib_true_risk = run_inference(
        encoder, flight_repr_layer, heads, airport_to_idx, carrier_to_idx, calib_months, device
    )
    print(f"Calibration flights: {len(calib_true_delay):,}", flush=True)

    nonconformity_a = np.abs(calib_pred_delay - calib_true_delay)
    prob_true_class = np.where(calib_true_risk == 1, calib_pred_risk, 1 - calib_pred_risk)
    nonconformity_b = 1 - prob_true_class

    print(f"\nRunning inference on test months {test_months} ...", flush=True)
    test_pred_delay, test_true_delay, test_pred_risk, test_true_risk = run_inference(
        encoder, flight_repr_layer, heads, airport_to_idx, carrier_to_idx, test_months, device
    )
    print(f"Test flights: {len(test_true_delay):,}", flush=True)

    report = {
        "checkpoint": args.checkpoint,
        "encoder": args.encoder,
        "calibration_months": calib_months,
        "test_months": test_months,
        "n_calibration_flights": int(len(calib_true_delay)),
        "n_test_flights": int(len(test_true_delay)),
        "results_by_alpha": {},
    }

    print("\n" + "=" * 70)
    for alpha in args.alphas:
        q_a = conformal_quantile(nonconformity_a, alpha)
        q_b = conformal_quantile(nonconformity_b, alpha)
        head_a_result = evaluate_head_a(test_pred_delay, test_true_delay, q_a)
        head_b_result = evaluate_head_b(test_pred_risk, test_true_risk, q_b)

        target_coverage = 1 - alpha
        print(f"\nTarget coverage: {target_coverage:.0%}  (alpha={alpha})")
        print(f"  HEAD A -- q={q_a:.3f} min | empirical coverage={head_a_result['empirical_coverage']:.4f} "
              f"| mean interval width={head_a_result['mean_interval_width_minutes']:.2f} min")
        print(f"  HEAD B -- q={q_b:.3f}     | empirical coverage={head_b_result['empirical_coverage']:.4f} "
              f"| mean set size={head_b_result['mean_set_size']:.3f} "
              f"| empty-set rate={head_b_result['empty_set_rate']:.4f}")

        report["results_by_alpha"][str(alpha)] = {
            "target_coverage": target_coverage,
            "head_a": head_a_result,
            "head_b": head_b_result,
        }
    print("=" * 70)

    out_path = CHECKPOINT_DIR / f"conformal_report_{args.tag}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
