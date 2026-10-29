"""
Training loop. One optimizer step per time window: run the encoder once
over that window's K-snapshot history, build every real flight's
representation from the resulting airport embeddings, compute the
multi-task loss, backprop.

Time-based split (Model_Architecture_Specification.docx Section 7 -- never
split randomly, since a later flight's rotation feature can depend on an
earlier one):
  Train      : 2019-01 .. 2023-12  (60 months)
  Validation : 2024-01 .. 2024-06  (6 months)
  Calibration: 2024-07 .. 2024-12  (6 months)  -- reserved for Stage 7 conformal
  Test       : 2025-01 .. 2025-06  (6 months)

Run with --months to restrict to a small verification subset before
committing to the full 60-month training range, e.g.:
    python 12_train_model.py --months 2023-07 2023-08 --epochs 1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import (
    SpatioTemporalEncoder, SpatialOnlyEncoder, TemporalOnlyEncoder,
    SpatioTemporalAttentionEncoder, AdaptiveGraphSpatioTemporalEncoder,
    FlightRepresentation, MultiTaskHeads, UncertaintyWeightedLoss,
)
from model.dataset import MonthWindowSource, build_flight_tensors, load_airport_index, load_carrier_index
from model.heads import CAUSE_CLASS_WEIGHTS

N_SELECTED_AIRPORTS = 200  # fixed by scripts/08_select_airports_and_edge_weights.py

# Methodology_Refinements_PreImplementation.docx Section 3 baseline table:
# "gat_only" drops the TCN (spatial-only ablation), "lstm_only" drops the
# GAT (temporal-only ablation, using our TCN in the role the table calls
# "LSTM" -- see model/encoder.py). "full" is the real model. "stpn_lite"
# and "mstagcn_lite" are fast, honest checks of STPN/SGDAN's and
# MSTAGCN's core distinguishing ideas (attention-based temporal
# aggregation, and a learned/adaptive graph, respectively) -- not full
# reimplementations, see model/encoder.py docstrings.
ENCODER_BUILDERS = {
    "full": lambda: SpatioTemporalEncoder(node_feat_dim=10, hidden_dim=64, gat_layers=2, gat_heads=4, tcn_layers=3, edge_dim=2),
    "gat_only": lambda: SpatialOnlyEncoder(node_feat_dim=10, hidden_dim=64, gat_layers=2, gat_heads=4, edge_dim=2),
    "lstm_only": lambda: TemporalOnlyEncoder(node_feat_dim=10, hidden_dim=64, tcn_layers=3),
    "stpn_lite": lambda: SpatioTemporalAttentionEncoder(node_feat_dim=10, hidden_dim=64, gat_layers=2, gat_heads=4, edge_dim=2, attn_heads=4),
    "mstagcn_lite": lambda: AdaptiveGraphSpatioTemporalEncoder(num_nodes=N_SELECTED_AIRPORTS, node_feat_dim=10, hidden_dim=64, embed_dim=16, spatial_layers=2, tcn_layers=3),
}

FULL_TRAIN_MONTHS = [f"{y}-{m:02d}" for y in range(2019, 2024) for m in range(1, 13)]
FULL_VAL_MONTHS = [f"2024-{m:02d}" for m in range(1, 7)]

CHECKPOINT_DIR = Path("./data_processed/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def save_checkpoint(path, epoch, model_parts, optimizer, history):
    encoder, flight_repr_layer, heads, weighted_loss = model_parts
    torch.save({
        "epoch": epoch,
        "encoder": encoder.state_dict(),
        "flight_repr_layer": flight_repr_layer.state_dict(),
        "heads": heads.state_dict(),
        "weighted_loss": weighted_loss.state_dict(),
        "optimizer": optimizer.state_dict(),
        "history": history,
    }, path)


def load_checkpoint(path, model_parts, optimizer, device):
    encoder, flight_repr_layer, heads, weighted_loss = model_parts
    ckpt = torch.load(path, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["encoder"])
    flight_repr_layer.load_state_dict(ckpt["flight_repr_layer"])
    heads.load_state_dict(ckpt["heads"])
    weighted_loss.load_state_dict(ckpt["weighted_loss"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"], ckpt["history"]


def run_epoch(month_keys, model_parts, airport_to_idx, carrier_to_idx, device, optimizer=None, K=12):
    encoder, flight_repr_layer, heads, weighted_loss = model_parts
    training = optimizer is not None
    encoder.train(training)
    flight_repr_layer.train(training)
    heads.train(training)

    # Addresses the rare-cause blind spot found in the first full
    # evaluation (macro-F1 0.412 vs. accuracy 0.653; security delay scored
    # 0.000 F1 on only 1,272 test cases) -- see model/heads.py for how
    # these weights were derived.
    cause_class_weights = torch.tensor(CAUSE_CLASS_WEIGHTS, device=device)

    total_loss_a, total_loss_b, total_loss_c, n_windows, n_flights = 0.0, 0.0, 0.0, 0, 0

    for month_key in month_keys:
        t0 = time.time()
        source = MonthWindowSource(month_key, K=K)
        month_windows, month_flights = 0, 0

        for window_idx in range(source.n_windows()):
            result = source.get_window(window_idx)
            if result is None:
                continue
            snapshots, flights_df = result
            if len(flights_df) == 0:
                continue

            batch = build_flight_tensors(flights_df, airport_to_idx, carrier_to_idx, device=device, oversample_weather=training)
            if batch is None:
                continue

            snapshots = [s.to(device) for s in snapshots]

            with torch.set_grad_enabled(training):
                airport_embeddings = encoder(snapshots)
                origin_emb = airport_embeddings[batch["origin_idx"]]
                dest_emb = airport_embeddings[batch["dest_idx"]]
                flight_repr = flight_repr_layer(
                    origin_emb, dest_emb, batch["carrier_idx"], batch["prev_cause_idx"], batch["flight_numeric"]
                )
                outputs = heads(flight_repr)

                loss_a = F.mse_loss(outputs["delay_minutes"], batch["true_delay"])
                loss_b = F.binary_cross_entropy_with_logits(outputs["delay_risk_logit"], batch["true_risk"])

                # F.cross_entropy(..., ignore_index=-1) silently returns NaN
                # (0/0) when EVERY label in the batch is the ignore index --
                # i.e. a 4-hour window with zero delayed flights, which is
                # rare but, across ~11,000 windows in a full training run,
                # effectively guaranteed to happen at least once. That single
                # NaN poisons every parameter in the model permanently via
                # backward(), which is exactly what happened in the first
                # full-scale run (every epoch showed NaN from epoch 1 on).
                # Skip Head C's contribution entirely for windows with no
                # delayed flights instead of computing an undefined 0/0 mean.
                has_delayed = batch["cause_label"] != -1
                if has_delayed.any():
                    loss_c = F.cross_entropy(
                        outputs["cause_logits"][has_delayed], batch["cause_label"][has_delayed],
                        weight=cause_class_weights,
                    )
                else:
                    loss_c = torch.zeros((), device=device)

                loss = weighted_loss(loss_a, loss_b, loss_c)

            if training:
                optimizer.zero_grad()
                loss.backward()
                # Standard safety net against exploding gradients from any
                # source -- cheap insurance now that we know this run can
                # silently poison itself over hours with nobody watching.
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(flight_repr_layer.parameters()) + list(heads.parameters()) + list(weighted_loss.parameters()),
                    max_norm=5.0,
                )
                optimizer.step()

            batch_size = batch["true_delay"].shape[0]  # actual flights used, after dropping any outside the top-200 airports
            total_loss_a += loss_a.item() * batch_size
            total_loss_b += loss_b.item() * batch_size
            total_loss_c += loss_c.item() * batch_size
            n_windows += 1
            month_windows += 1
            n_flights += batch_size
            month_flights += batch_size

        elapsed = time.time() - t0
        print(f"  {month_key}: {month_windows} windows, {month_flights:,} flights, {elapsed:.1f}s", flush=True)

    return {
        "loss_a": total_loss_a / max(n_flights, 1),
        "loss_b": total_loss_b / max(n_flights, 1),
        "loss_c": total_loss_c / max(n_flights, 1),
        "n_windows": n_windows,
        "n_flights": n_flights,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", nargs="*", default=None, help="Restrict training to these months (verification mode). Default: full training range.")
    parser.add_argument("--val-months", nargs="*", default=None, help="Validation months. Default: full validation range.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resume", action="store_true", help="Resume from data_processed/checkpoints/latest.pt if it exists")
    parser.add_argument("--tag", default=None, help="Suffix for checkpoint filenames, to keep verification runs separate from the real training run, e.g. --tag verify")
    parser.add_argument("--encoder", choices=list(ENCODER_BUILDERS), default="full", help="Which encoder to train -- 'full' is the real model, 'gat_only'/'lstm_only' are the ablation baselines from Methodology_Refinements_PreImplementation.docx Section 3.")
    args = parser.parse_args()

    latest_path = CHECKPOINT_DIR / (f"latest_{args.tag}.pt" if args.tag else "latest.pt")
    best_path = CHECKPOINT_DIR / (f"best_{args.tag}.pt" if args.tag else "best.pt")
    history_path = CHECKPOINT_DIR / (f"training_history_{args.tag}.json" if args.tag else "training_history.json")

    train_months = args.months if args.months else FULL_TRAIN_MONTHS
    val_months = args.val_months if args.val_months else FULL_VAL_MONTHS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Train months: {train_months}", flush=True)
    print(f"Validation months: {val_months}", flush=True)

    airport_to_idx = load_airport_index()
    print(f"Loading carrier vocabulary...", flush=True)
    carrier_to_idx = load_carrier_index()
    print(f"{len(carrier_to_idx)} carriers, {len(airport_to_idx)} airports", flush=True)

    print(f"Encoder: {args.encoder}", flush=True)
    encoder = ENCODER_BUILDERS[args.encoder]().to(device)
    flight_repr_layer = FlightRepresentation(airport_emb_dim=64, n_carriers=len(carrier_to_idx), carrier_emb_dim=8, cause_emb_dim=4, flight_numeric_dim=4).to(device)
    heads = MultiTaskHeads(input_dim=flight_repr_layer.output_dim, hidden_dim=64).to(device)
    weighted_loss = UncertaintyWeightedLoss().to(device)

    params = list(encoder.parameters()) + list(flight_repr_layer.parameters()) + list(heads.parameters()) + list(weighted_loss.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    model_parts = (encoder, flight_repr_layer, heads, weighted_loss)

    start_epoch = 0
    history = []
    best_val_loss = float("inf")
    if args.resume and latest_path.exists():
        start_epoch, history = load_checkpoint(latest_path, model_parts, optimizer, device)
        best_val_loss = min((h["val_total"] for h in history), default=float("inf"))
        print(f"Resumed from {latest_path} at epoch {start_epoch}, best val loss so far: {best_val_loss:.3f}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===", flush=True)
        print("Training:", flush=True)
        train_metrics = run_epoch(train_months, model_parts, airport_to_idx, carrier_to_idx, device, optimizer=optimizer)
        print(f"  TRAIN loss_a={train_metrics['loss_a']:.3f} loss_b={train_metrics['loss_b']:.3f} loss_c={train_metrics['loss_c']:.3f} "
              f"({train_metrics['n_windows']} windows, {train_metrics['n_flights']:,} flights)", flush=True)

        print("Validating:", flush=True)
        val_metrics = run_epoch(val_months, model_parts, airport_to_idx, carrier_to_idx, device, optimizer=None)
        print(f"  VAL   loss_a={val_metrics['loss_a']:.3f} loss_b={val_metrics['loss_b']:.3f} loss_c={val_metrics['loss_c']:.3f} "
              f"({val_metrics['n_windows']} windows, {val_metrics['n_flights']:,} flights)", flush=True)

        val_total = val_metrics["loss_a"] + val_metrics["loss_b"] + val_metrics["loss_c"]
        history.append({
            "epoch": epoch + 1,
            "train_loss_a": train_metrics["loss_a"], "train_loss_b": train_metrics["loss_b"], "train_loss_c": train_metrics["loss_c"],
            "val_loss_a": val_metrics["loss_a"], "val_loss_b": val_metrics["loss_b"], "val_loss_c": val_metrics["loss_c"],
            "val_total": val_total,
        })

        save_checkpoint(latest_path, epoch + 1, model_parts, optimizer, history)
        if val_total < best_val_loss:
            best_val_loss = val_total
            save_checkpoint(best_path, epoch + 1, model_parts, optimizer, history)
            print(f"  New best validation loss ({val_total:.3f}) -- saved to {best_path.name}", flush=True)

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"  Checkpoint saved to {latest_path.name} (epoch {epoch + 1} complete -- safe to interrupt and resume with --resume)", flush=True)


if __name__ == "__main__":
    main()
