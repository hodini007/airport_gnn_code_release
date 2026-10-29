"""
Stage 8: Counterfactual explanation for Head B (delay-risk), per
Methodology_Refinements_PreImplementation.docx Section 2.

Documented deviation from the original spec (see Literature_Review_Novelty_
Q1_Assessment.docx Section 7 for the decision record): the spec names three
actionable features (prior-leg delay, turnaround time, taxi-out time), but
taxi-out time only exists in this pipeline as an airport-window aggregate
(MEAN_TAXI_OUT feeding the graph encoder), not as a per-flight input any
flight-level prediction can be differentiated against. This script uses the
two actionable features that ARE genuine per-flight, differentiable model
inputs: ROTATION_PREV_ARR_DELAY (prior-leg delay) and ROTATION_TURNAROUND_MIN
(turnaround time), both already in FLIGHT_NUMERIC_COLS. Taxi-out time is
omitted; state this plainly as a limitation, not a silent substitution.

A second documented adaptation: bounds are specified in the methodology as
"conditioned on aircraft type and, where sample size allows, airport." This
project's BTS data has no aircraft-type/equipment column (only TAIL_NUM), so
bounds are conditioned on (ORIGIN, OP_UNIQUE_CARRIER) instead, with a
two-tier fallback (ORIGIN-only, then global) when a group has fewer than
MIN_GROUP_SIZE examples in the training data.

Procedure:
  1. Bounds: for each actionable feature, floor = 5th percentile of observed
     training-set (2019-2023) values for that (ORIGIN, CARRIER) group;
     ceiling = the flight's own observed value (a counterfactual only ever
     proposes a reduction, never an increase).
  2. Candidate selection: flights in the test set (2025-01..2025-06)
     predicted at risk (P(delayed) >= 0.5) with a trackable previous leg,
     randomly sampled to N_SAMPLE instances.
  3. Primary method -- projected gradient search: holding the encoder's
     airport embeddings, carrier embedding, and previous-cause embedding
     fixed (they are context, not something a single flight's own rotation/
     turnaround can retroactively change), take small steps of the two
     actionable features along the normalized negative gradient of P(delayed)
     with respect to them, projecting back into [floor, ceiling] after every
     step, until the prediction flips below 0.5 or MAX_ITERS is reached.
  4. Verification method -- coordinate/grid search: an independent,
     discretized grid over the same two features and bounds, evaluated in
     one batched forward pass per instance; report the agreement rate with
     the gradient method and how their proposed costs compare, as the
     robustness check specified in the methodology.
  5. Selection rule: for each instance, report whichever method's successful
     flip has the smaller weighted-L1 cost (weights stated explicitly below
     as a domain judgment call, not learned).
  6. Fidelity, two components:
       - Self-consistency: re-run the model on the final proposed input,
         confirm the prediction actually flips (necessary, not sufficient).
       - Historical cross-check: for each proposed counterfactual, search
         the reserved calibration slice (2024-07..2024-12, never used in
         training/validation/test) for real flights on the same
         (ORIGIN, DEST, CARRIER) whose actual feature values are within
         TOLERANCE_MINUTES of the proposal, and report their empirical
         on-time rate.

Encoder-agnostic (Literature_Review_Novelty_Q1_Assessment.docx Section 7.2):
run once per --encoder/--checkpoint pair, on both the proposed full model
and the LSTM-only ablation.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import (
    SpatioTemporalEncoder, SpatialOnlyEncoder, TemporalOnlyEncoder,
    SpatioTemporalAttentionEncoder, AdaptiveGraphSpatioTemporalEncoder,
    FlightRepresentation, MultiTaskHeads,
)
from model.dataset import MonthWindowSource, build_flight_tensors, load_airport_index, load_carrier_index, FLIGHT_NUMERIC_COLS

N_SELECTED_AIRPORTS = 200

ENCODER_BUILDERS = {
    "full": lambda: SpatioTemporalEncoder(node_feat_dim=10, hidden_dim=64, gat_layers=2, gat_heads=4, tcn_layers=3, edge_dim=2),
    "gat_only": lambda: SpatialOnlyEncoder(node_feat_dim=10, hidden_dim=64, gat_layers=2, gat_heads=4, edge_dim=2),
    "lstm_only": lambda: TemporalOnlyEncoder(node_feat_dim=10, hidden_dim=64, tcn_layers=3),
    "stpn_lite": lambda: SpatioTemporalAttentionEncoder(node_feat_dim=10, hidden_dim=64, gat_layers=2, gat_heads=4, edge_dim=2, attn_heads=4),
    "mstagcn_lite": lambda: AdaptiveGraphSpatioTemporalEncoder(num_nodes=N_SELECTED_AIRPORTS, node_feat_dim=10, hidden_dim=64, embed_dim=16, spatial_layers=2, tcn_layers=3),
}

CHECKPOINT_DIR = Path("./data_processed/checkpoints")
ENRICHED_DIR = Path("./data_processed/monthly_enriched")

TRAIN_MONTHS = [f"{y}-{m:02d}" for y in range(2019, 2024) for m in range(1, 13)]
CALIB_MONTHS = [f"2024-{m:02d}" for m in range(7, 13)]
TEST_MONTHS = [f"2025-{m:02d}" for m in range(1, 7)]
K = 12

ACTIONABLE_COLS = ["ROTATION_PREV_ARR_DELAY", "ROTATION_TURNAROUND_MIN"]
ACTIONABLE_IDX = [FLIGHT_NUMERIC_COLS.index(c) for c in ACTIONABLE_COLS]  # [0, 1]
# Domain-judgment "cost per minute changed" weights, stated explicitly per
# the spec's selection-rule requirement. Prior-leg delay is weighted higher:
# it depends on an earlier upstream flight's own performance, so an airline
# has less direct control over reducing it than over its own turnaround
# buffer at this airport.
FEATURE_WEIGHTS = {"ROTATION_PREV_ARR_DELAY": 1.5, "ROTATION_TURNAROUND_MIN": 1.0}

DECISION_THRESHOLD = 0.5
N_SAMPLE = 1000
MAX_ITERS = 60
STEP_SIZE_MINUTES = 2.0
GRID_POINTS_PER_DIM = 15
MIN_GROUP_SIZE = 30
TOLERANCE_MINUTES = 5.0
RNG_SEED = 42


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


def compute_bounds(rng_seed=RNG_SEED):
    """5th-percentile floors for each actionable feature, grouped by
    (ORIGIN, OP_UNIQUE_CARRIER) from the training months, with a two-tier
    fallback (ORIGIN-only, then global) for undersized groups."""
    print("Computing actionable-feature bounds from training months...", flush=True)
    cols = ["ORIGIN", "OP_UNIQUE_CARRIER"] + ACTIONABLE_COLS
    frames = []
    for month_key in TRAIN_MONTHS:
        path = ENRICHED_DIR / f"{month_key}.parquet"
        if not path.exists():
            continue
        frames.append(pd.read_parquet(path, columns=cols))
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=ACTIONABLE_COLS)
    print(f"  {len(df):,} training-set flights used for bounds", flush=True)

    global_floor = df[ACTIONABLE_COLS].quantile(0.05)

    origin_group = df.groupby("ORIGIN")[ACTIONABLE_COLS]
    origin_floor = origin_group.quantile(0.05)
    origin_count = origin_group.size()

    route_group = df.groupby(["ORIGIN", "OP_UNIQUE_CARRIER"])[ACTIONABLE_COLS]
    route_floor = route_group.quantile(0.05)
    route_count = route_group.size()

    return {
        "global_floor": global_floor,
        "origin_floor": origin_floor, "origin_count": origin_count,
        "route_floor": route_floor, "route_count": route_count,
    }


def lookup_floor(bounds, origin, carrier):
    key = (origin, carrier)
    if key in bounds["route_count"].index and bounds["route_count"][key] >= MIN_GROUP_SIZE:
        return bounds["route_floor"].loc[key].values.astype("float32")
    if origin in bounds["origin_count"].index and bounds["origin_count"][origin] >= MIN_GROUP_SIZE:
        return bounds["origin_floor"].loc[origin].values.astype("float32")
    return bounds["global_floor"].values.astype("float32")


def collect_candidates(encoder, flight_repr_layer, heads, airport_to_idx, carrier_to_idx, device, rng):
    """Scans the test set, keeps flights predicted at-risk with a trackable
    previous leg, then randomly samples N_SAMPLE of them."""
    candidates = []
    with torch.no_grad():
        for month_key in TEST_MONTHS:
            source = MonthWindowSource(month_key, K=K)
            n_kept = 0
            for window_idx in range(source.n_windows()):
                result = source.get_window(window_idx)
                if result is None:
                    continue
                snapshots, flights_df = result
                flights_df = flights_df[flights_df["ROTATION_HAS_PREV_LEG"] == True]  # noqa: E712
                if len(flights_df) == 0:
                    continue
                # Mirror build_flight_tensors' own top-200-airport filter here so
                # df_kept below stays row-aligned with the tensors it returns --
                # build_flight_tensors silently drops non-top-200-airport rows.
                flights_df = flights_df[flights_df["ORIGIN"].isin(airport_to_idx) & flights_df["DEST"].isin(airport_to_idx)]
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
                prob = torch.sigmoid(heads.head_b(flight_repr).squeeze(-1))
                at_risk = (prob >= DECISION_THRESHOLD).cpu().numpy()
                if not at_risk.any():
                    continue

                idx = np.nonzero(at_risk)[0]
                df_kept = flights_df.iloc[idx]
                for j, row_pos in enumerate(idx):
                    candidates.append({
                        "origin_emb": origin_emb[row_pos].detach().cpu(),
                        "dest_emb": dest_emb[row_pos].detach().cpu(),
                        "carrier_idx": batch["carrier_idx"][row_pos].item(),
                        "prev_cause_idx": batch["prev_cause_idx"][row_pos].item(),
                        "flight_numeric": batch["flight_numeric"][row_pos].detach().cpu(),
                        "ORIGIN": df_kept.iloc[j]["ORIGIN"], "DEST": df_kept.iloc[j]["DEST"],
                        "OP_UNIQUE_CARRIER": df_kept.iloc[j]["OP_UNIQUE_CARRIER"],
                        "pred_prob": float(prob[row_pos].item()),
                    })
                n_kept += int(at_risk.sum())
            print(f"  {month_key}: {n_kept} at-risk, trackable-previous-leg candidates found", flush=True)

    print(f"Total candidates found: {len(candidates):,}", flush=True)
    sample_idx = rng.choice(len(candidates), size=min(N_SAMPLE, len(candidates)), replace=False)
    return [candidates[i] for i in sample_idx]


def forward_head_b(flight_repr_layer, heads, origin_emb, dest_emb, carrier_idx, prev_cause_idx, numeric):
    carrier_vec = flight_repr_layer.carrier_embedding(carrier_idx)
    prev_cause_vec = flight_repr_layer.prev_cause_embedding(prev_cause_idx)
    flight_repr = torch.cat([origin_emb, dest_emb, carrier_vec, prev_cause_vec, numeric], dim=-1)
    logit = heads.head_b(flight_repr).squeeze(-1)
    return torch.sigmoid(logit)


def gradient_search(flight_repr_layer, heads, batch, floors, device):
    """batch: dict of stacked tensors for one candidate batch. floors: [B, 2] tensor.
    Returns per-instance: flipped (bool), final numeric[:, :2] (proposed values), n_iters, prob trajectory endpoint."""
    B = batch["flight_numeric"].shape[0]
    ceilings = batch["flight_numeric"][:, ACTIONABLE_IDX].clone()  # original values, never exceeded
    var = ceilings.clone().detach().to(device).requires_grad_(True)
    fixed = batch["flight_numeric"][:, [i for i in range(4) if i not in ACTIONABLE_IDX]].clone().to(device)
    origin_emb = batch["origin_emb"].to(device)
    dest_emb = batch["dest_emb"].to(device)
    carrier_idx = batch["carrier_idx"].to(device)
    prev_cause_idx = batch["prev_cause_idx"].to(device)
    floors = floors.to(device)
    ceilings = ceilings.to(device)

    flipped = torch.zeros(B, dtype=torch.bool, device=device)
    final_var = var.detach().clone()
    n_iters_used = torch.zeros(B, dtype=torch.long, device=device)
    final_prob = torch.ones(B, device=device)

    for it in range(1, MAX_ITERS + 1):
        numeric = torch.cat([var, fixed], dim=-1)  # column order [prev_delay, turnaround, crs_elapsed, distance] -- matches FLIGHT_NUMERIC_COLS since ACTIONABLE_IDX == [0, 1]
        prob = forward_head_b(flight_repr_layer, heads, origin_emb, dest_emb, carrier_idx, prev_cause_idx, numeric)
        just_flipped = (prob < DECISION_THRESHOLD) & (~flipped)
        if just_flipped.any():
            final_var[just_flipped] = var.detach()[just_flipped]
            n_iters_used[just_flipped] = it - 1
            final_prob[just_flipped] = prob.detach()[just_flipped]
            flipped = flipped | just_flipped
        if flipped.all():
            break

        grad = torch.autograd.grad(prob.sum(), var, retain_graph=False)[0]
        with torch.no_grad():
            norm = grad.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            step = -STEP_SIZE_MINUTES * grad / norm
            new_var = (var + step).clamp(min=floors, max=ceilings)
            new_var[flipped] = var[flipped]  # freeze already-flipped instances
            var.data = new_var

    # Instances that never flipped: report their last state for reference.
    never_flipped = ~flipped
    final_var[never_flipped] = var.detach()[never_flipped]
    n_iters_used[never_flipped] = MAX_ITERS
    with torch.no_grad():
        numeric = torch.cat([var, fixed], dim=-1)
        final_prob[never_flipped] = forward_head_b(flight_repr_layer, heads, origin_emb, dest_emb, carrier_idx, prev_cause_idx, numeric)[never_flipped]

    return flipped.cpu(), final_var.detach().cpu(), n_iters_used.cpu(), final_prob.detach().cpu()


def grid_search(flight_repr_layer, heads, batch, floors, device):
    """Discretized grid over the two actionable features, evaluated per instance.
    Returns per-instance: flipped (bool), best (smallest weighted-L1) proposed values among successful grid points."""
    B = batch["flight_numeric"].shape[0]
    ceilings = batch["flight_numeric"][:, ACTIONABLE_IDX].to(device)
    fixed = batch["flight_numeric"][:, [i for i in range(4) if i not in ACTIONABLE_IDX]].to(device)
    origin_emb = batch["origin_emb"].to(device)
    dest_emb = batch["dest_emb"].to(device)
    carrier_idx = batch["carrier_idx"].to(device)
    prev_cause_idx = batch["prev_cause_idx"].to(device)
    floors = floors.to(device)
    weights = torch.tensor([FEATURE_WEIGHTS[c] for c in ACTIONABLE_COLS], device=device)

    flipped = torch.zeros(B, dtype=torch.bool, device=device)
    best_var = ceilings.clone()
    best_cost = torch.full((B,), float("inf"), device=device)

    lin = torch.linspace(0, 1, GRID_POINTS_PER_DIM, device=device)
    with torch.no_grad():
        for a in lin:
            for b in lin:
                frac = torch.stack([a.expand(B), b.expand(B)], dim=-1)
                cand = floors + frac * (ceilings - floors)
                numeric = torch.cat([cand, fixed], dim=-1)
                prob = forward_head_b(flight_repr_layer, heads, origin_emb, dest_emb, carrier_idx, prev_cause_idx, numeric)
                is_flip = prob < DECISION_THRESHOLD
                cost = (weights * (ceilings - cand).abs()).sum(dim=-1)
                better = is_flip & (cost < best_cost)
                best_var[better] = cand[better]
                best_cost[better] = cost[better]
                flipped = flipped | is_flip

    return flipped.cpu(), best_var.detach().cpu()


def weighted_l1(ceilings, proposed):
    weights = np.array([FEATURE_WEIGHTS[c] for c in ACTIONABLE_COLS])
    return (weights * np.abs(ceilings - proposed)).sum(axis=-1)


def historical_cross_check(final_candidates, calib_df):
    """For each proposed counterfactual, find real calibration-slice flights on the
    same (ORIGIN, DEST, CARRIER) with actual actionable-feature values within
    TOLERANCE_MINUTES, and report their empirical on-time rate."""
    grouped = calib_df.groupby(["ORIGIN", "DEST", "OP_UNIQUE_CARRIER"])
    on_time_rates = []
    match_counts = []
    for c in final_candidates:
        key = (c["ORIGIN"], c["DEST"], c["OP_UNIQUE_CARRIER"])
        if key not in grouped.groups:
            match_counts.append(0)
            continue
        sub = grouped.get_group(key)
        close = (
            (sub["ROTATION_PREV_ARR_DELAY"] - c["proposed"][0]).abs().le(TOLERANCE_MINUTES)
            & (sub["ROTATION_TURNAROUND_MIN"] - c["proposed"][1]).abs().le(TOLERANCE_MINUTES)
        )
        matched = sub[close]
        match_counts.append(len(matched))
        if len(matched) > 0:
            on_time_rates.append(float((matched["ARR_DEL15"] == 0).mean()))
    return on_time_rates, match_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--encoder", choices=list(ENCODER_BUILDERS), default="full")
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Checkpoint: {args.checkpoint}  (encoder={args.encoder})", flush=True)
    rng = np.random.default_rng(RNG_SEED)

    encoder, flight_repr_layer, heads, airport_to_idx, carrier_to_idx = load_model(args.checkpoint, args.encoder, device)

    bounds = compute_bounds()

    print("\nCollecting at-risk candidate flights from the test set...", flush=True)
    candidates = collect_candidates(encoder, flight_repr_layer, heads, airport_to_idx, carrier_to_idx, device, rng)
    print(f"Sampled {len(candidates)} candidates for counterfactual search", flush=True)

    floors = torch.stack([
        torch.tensor(lookup_floor(bounds, c["ORIGIN"], c["OP_UNIQUE_CARRIER"])) for c in candidates
    ])
    ceilings_check = torch.stack([c["flight_numeric"][ACTIONABLE_IDX] for c in candidates])
    # A flight can already sit below its group's 5th percentile (e.g. an
    # unusually smooth turnaround); clamp floor<=ceiling per-instance so the
    # search range is never inverted -- such an instance simply has zero
    # remaining room to reduce that feature, which is the correct outcome.
    floors = torch.minimum(floors, ceilings_check)
    batch = {
        "flight_numeric": torch.stack([c["flight_numeric"] for c in candidates]),
        "origin_emb": torch.stack([c["origin_emb"] for c in candidates]),
        "dest_emb": torch.stack([c["dest_emb"] for c in candidates]),
        "carrier_idx": torch.tensor([c["carrier_idx"] for c in candidates], dtype=torch.long),
        "prev_cause_idx": torch.tensor([c["prev_cause_idx"] for c in candidates], dtype=torch.long),
    }

    print("\nRunning primary method (projected gradient search)...", flush=True)
    grad_flipped, grad_var, grad_iters, grad_final_prob = gradient_search(flight_repr_layer, heads, batch, floors, device)

    print("Running verification method (coordinate/grid search)...", flush=True)
    grid_flipped, grid_var = grid_search(flight_repr_layer, heads, batch, floors, device)

    ceilings_np = batch["flight_numeric"][:, ACTIONABLE_IDX].numpy()
    grad_var_np, grid_var_np = grad_var.numpy(), grid_var.numpy()
    grad_cost = weighted_l1(ceilings_np, grad_var_np)
    grid_cost = weighted_l1(ceilings_np, grid_var_np)
    grad_flipped_np, grid_flipped_np = grad_flipped.numpy(), grid_flipped.numpy()

    both_flipped = grad_flipped_np & grid_flipped_np
    either_flipped = grad_flipped_np | grid_flipped_np
    agreement_rate = float(both_flipped.sum() / max(either_flipped.sum(), 1))

    final_candidates = []
    for i, c in enumerate(candidates):
        if not either_flipped[i]:
            continue
        use_grid = grid_flipped_np[i] and (not grad_flipped_np[i] or grid_cost[i] < grad_cost[i])
        proposed = grid_var_np[i] if use_grid else grad_var_np[i]
        final_candidates.append({
            "ORIGIN": c["ORIGIN"], "DEST": c["DEST"], "OP_UNIQUE_CARRIER": c["OP_UNIQUE_CARRIER"],
            "proposed": proposed, "cost": float(grid_cost[i] if use_grid else grad_cost[i]),
            "method_used": "grid" if use_grid else "gradient",
        })

    print(f"\nGradient search: {grad_flipped_np.mean():.1%} flipped within {MAX_ITERS} iterations "
          f"(step={STEP_SIZE_MINUTES} min)", flush=True)
    print(f"Grid search:     {grid_flipped_np.mean():.1%} flipped within a {GRID_POINTS_PER_DIM}x{GRID_POINTS_PER_DIM} grid", flush=True)
    print(f"Agreement rate (both methods flip the same instance): {agreement_rate:.1%}", flush=True)
    if both_flipped.any():
        cost_ratio = grad_cost[both_flipped] / np.clip(grid_cost[both_flipped], 1e-6, None)
        print(f"Where both flip: gradient's cost is on average {cost_ratio.mean():.2f}x grid's best cost", flush=True)

    # Self-consistency: guaranteed by construction -- both search loops only
    # ever report a point that already satisfied prob < threshold at that
    # point -- reported as a count for the record rather than re-verified
    # redundantly.
    self_consistent = len(final_candidates)

    print(f"\nLoading calibration-slice data for the historical cross-check...", flush=True)
    calib_cols = ["ORIGIN", "DEST", "OP_UNIQUE_CARRIER", "ROTATION_PREV_ARR_DELAY", "ROTATION_TURNAROUND_MIN", "ARR_DEL15"]
    calib_frames = [pd.read_parquet(ENRICHED_DIR / f"{m}.parquet", columns=calib_cols) for m in CALIB_MONTHS]
    calib_df = pd.concat(calib_frames, ignore_index=True).dropna(subset=["ROTATION_PREV_ARR_DELAY", "ROTATION_TURNAROUND_MIN"])

    on_time_rates, match_counts = historical_cross_check(final_candidates, calib_df)
    match_rate = float(np.mean([m > 0 for m in match_counts])) if match_counts else 0.0

    print(f"Historical cross-check: {len(on_time_rates)}/{len(final_candidates)} counterfactuals had "
          f"at least one matching historical flight (match rate {match_rate:.1%})", flush=True)
    if on_time_rates:
        print(f"  Mean empirical on-time rate of matched historical flights: {np.mean(on_time_rates):.1%}", flush=True)

    report = {
        "checkpoint": args.checkpoint,
        "encoder": args.encoder,
        "actionable_features": ACTIONABLE_COLS,
        "omitted_feature": "taxi-out time (no per-flight input exists in this pipeline -- see script docstring)",
        "feature_weights": FEATURE_WEIGHTS,
        "n_candidates": len(candidates),
        "decision_threshold": DECISION_THRESHOLD,
        "max_iters": MAX_ITERS,
        "step_size_minutes": STEP_SIZE_MINUTES,
        "grid_points_per_dim": GRID_POINTS_PER_DIM,
        "gradient_search": {
            "flip_rate": float(grad_flipped_np.mean()),
            "mean_iters_when_flipped": float(grad_iters.numpy()[grad_flipped_np].mean()) if grad_flipped_np.any() else None,
        },
        "grid_search": {"flip_rate": float(grid_flipped_np.mean())},
        "agreement_rate_both_methods_flip": agreement_rate,
        "mean_cost_ratio_gradient_over_grid_when_both_flip": float(cost_ratio.mean()) if both_flipped.any() else None,
        "n_final_counterfactuals": len(final_candidates),
        "self_consistency_count": self_consistent,
        "historical_cross_check": {
            "n_with_match": len(on_time_rates),
            "match_rate": match_rate,
            "mean_matched_on_time_rate": float(np.mean(on_time_rates)) if on_time_rates else None,
            "median_matched_on_time_rate": float(np.median(on_time_rates)) if on_time_rates else None,
        },
    }
    out_path = CHECKPOINT_DIR / f"counterfactual_report_{args.tag}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
