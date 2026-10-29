"""
Stage 9-10: Proper evaluation on the held-out test set (2025-01..2025-06),
which the model has never seen during training or validation.

Computes the actual named metrics from the methodology, not just raw
training losses:
  - Head A (delay magnitude): RMSE, MAE
  - Head B (delay risk, given class imbalance): AUC-ROC, average precision
    (AUC-PR), F1/precision/recall at a 0.5 threshold
  - Head C (dominant cause, only on truly-delayed test flights): accuracy,
    macro-F1, per-class breakdown

Loads data_processed/checkpoints/best.pt by default (epoch 9 of the first
successful full-scale run) -- pass --checkpoint to evaluate a different one.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    roc_auc_score, average_precision_score, f1_score, precision_score, recall_score,
    accuracy_score, classification_report,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import (
    SpatioTemporalEncoder, SpatialOnlyEncoder, TemporalOnlyEncoder,
    SpatioTemporalAttentionEncoder, AdaptiveGraphSpatioTemporalEncoder,
    FlightRepresentation, MultiTaskHeads,
)
from model.dataset import MonthWindowSource, build_flight_tensors, load_airport_index, load_carrier_index
from model.heads import CAUSE_VOCAB

N_SELECTED_AIRPORTS = 200

ENCODER_BUILDERS = {
    "full": lambda: SpatioTemporalEncoder(node_feat_dim=10, hidden_dim=64, gat_layers=2, gat_heads=4, tcn_layers=3, edge_dim=2),
    "gat_only": lambda: SpatialOnlyEncoder(node_feat_dim=10, hidden_dim=64, gat_layers=2, gat_heads=4, edge_dim=2),
    "lstm_only": lambda: TemporalOnlyEncoder(node_feat_dim=10, hidden_dim=64, tcn_layers=3),
    "stpn_lite": lambda: SpatioTemporalAttentionEncoder(node_feat_dim=10, hidden_dim=64, gat_layers=2, gat_heads=4, edge_dim=2, attn_heads=4),
    "mstagcn_lite": lambda: AdaptiveGraphSpatioTemporalEncoder(num_nodes=N_SELECTED_AIRPORTS, node_feat_dim=10, hidden_dim=64, embed_dim=16, spatial_layers=2, tcn_layers=3),
}

CHECKPOINT_DIR = Path("./data_processed/checkpoints")
TEST_MONTHS = [f"2025-{m:02d}" for m in range(1, 7)]
K = 12


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_DIR / "best.pt"))
    parser.add_argument("--months", nargs="*", default=None, help="Override test months (default: the held-out test set, 2025-01..2025-06)")
    parser.add_argument("--encoder", choices=list(ENCODER_BUILDERS), default="full", help="Must match the --encoder the checkpoint was trained with.")
    args = parser.parse_args()
    test_months = args.months if args.months else TEST_MONTHS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Evaluating checkpoint: {args.checkpoint}", flush=True)
    print(f"Test months (held out from training and validation): {test_months}", flush=True)

    airport_to_idx = load_airport_index()
    carrier_to_idx = load_carrier_index()

    encoder = ENCODER_BUILDERS[args.encoder]().to(device)
    flight_repr_layer = FlightRepresentation(airport_emb_dim=64, n_carriers=len(carrier_to_idx), carrier_emb_dim=8, cause_emb_dim=4, flight_numeric_dim=4).to(device)
    heads = MultiTaskHeads(input_dim=flight_repr_layer.output_dim, hidden_dim=64).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["encoder"])
    flight_repr_layer.load_state_dict(ckpt["flight_repr_layer"])
    heads.load_state_dict(ckpt["heads"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}", flush=True)

    encoder.eval()
    flight_repr_layer.eval()
    heads.eval()

    all_pred_delay, all_true_delay = [], []
    all_pred_risk_prob, all_true_risk = [], []
    all_pred_cause, all_true_cause = [], []

    with torch.no_grad():
        for month_key in test_months:
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

                all_pred_delay.append(outputs["delay_minutes"].cpu().numpy())
                all_true_delay.append(batch["true_delay"].cpu().numpy())

                all_pred_risk_prob.append(torch.sigmoid(outputs["delay_risk_logit"]).cpu().numpy())
                all_true_risk.append(batch["true_risk"].cpu().numpy())

                has_delayed = batch["cause_label"] != -1
                if has_delayed.any():
                    cause_pred = outputs["cause_logits"][has_delayed].argmax(dim=-1).cpu().numpy()
                    all_pred_cause.append(cause_pred)
                    all_true_cause.append(batch["cause_label"][has_delayed].cpu().numpy())

                n_windows_used += 1
            print(f"  {month_key}: {n_windows_used} windows evaluated", flush=True)

    pred_delay = np.concatenate(all_pred_delay)
    true_delay = np.concatenate(all_true_delay)
    pred_risk_prob = np.concatenate(all_pred_risk_prob)
    true_risk = np.concatenate(all_true_risk)
    pred_cause = np.concatenate(all_pred_cause)
    true_cause = np.concatenate(all_true_cause)

    print(f"\nTotal test flights: {len(true_delay):,}  (of which delayed, for Head C: {len(true_cause):,})", flush=True)

    # --- Head A: delay magnitude ---
    rmse = float(np.sqrt(mean_squared_error(true_delay, pred_delay)))
    mae = float(mean_absolute_error(true_delay, pred_delay))

    # --- Head B: delay risk (binary) ---
    pred_risk_label = (pred_risk_prob >= 0.5).astype(int)
    auc_roc = float(roc_auc_score(true_risk, pred_risk_prob))
    auc_pr = float(average_precision_score(true_risk, pred_risk_prob))
    f1 = float(f1_score(true_risk, pred_risk_label))
    precision = float(precision_score(true_risk, pred_risk_label))
    recall = float(recall_score(true_risk, pred_risk_label))
    positive_rate = float(true_risk.mean())

    # --- Head C: dominant cause (5-class, delayed flights only) ---
    cause_accuracy = float(accuracy_score(true_cause, pred_cause))
    cause_report = classification_report(
        true_cause, pred_cause, target_names=CAUSE_VOCAB[:5], output_dict=True, zero_division=0,
    )
    macro_f1 = float(cause_report["macro avg"]["f1-score"])

    print("\n" + "=" * 60)
    print("HEAD A -- Delay magnitude (regression)")
    print(f"  RMSE: {rmse:.2f} minutes")
    print(f"  MAE:  {mae:.2f} minutes")

    print("\nHEAD B -- Delay risk (binary classification)")
    print(f"  Base rate (fraction actually delayed): {positive_rate:.3f}")
    print(f"  AUC-ROC:          {auc_roc:.4f}")
    print(f"  AUC-PR:           {auc_pr:.4f}")
    print(f"  F1 (@ 0.5):       {f1:.4f}")
    print(f"  Precision (@0.5): {precision:.4f}")
    print(f"  Recall (@0.5):    {recall:.4f}")

    print("\nHEAD C -- Dominant cause (5-class, delayed flights only)")
    print(f"  Accuracy:  {cause_accuracy:.4f}")
    print(f"  Macro-F1:  {macro_f1:.4f}")
    for cause in CAUSE_VOCAB[:5]:
        stats = cause_report.get(cause, {})
        print(f"    {cause:22s} precision={stats.get('precision', 0):.3f} recall={stats.get('recall', 0):.3f} "
              f"f1={stats.get('f1-score', 0):.3f} support={int(stats.get('support', 0))}")
    print("=" * 60)

    report = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": ckpt["epoch"],
        "test_months": test_months,
        "n_test_flights": int(len(true_delay)),
        "n_test_delayed_flights": int(len(true_cause)),
        "head_a": {"rmse_minutes": rmse, "mae_minutes": mae},
        "head_b": {
            "base_rate": positive_rate, "auc_roc": auc_roc, "auc_pr": auc_pr,
            "f1_at_0.5": f1, "precision_at_0.5": precision, "recall_at_0.5": recall,
        },
        "head_c": {"accuracy": cause_accuracy, "macro_f1": macro_f1, "per_class": cause_report},
    }
    out_path = CHECKPOINT_DIR / "test_evaluation.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
