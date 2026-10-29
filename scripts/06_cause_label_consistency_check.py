"""
Stage 3b: Cause-label consistency check.

Implements the three checks specified in Methodology_Refinements_PreImplementation.docx
Section 1, to establish whether BTS's five self-reported delay-cause fields
are trustworthy enough to build RQ2's cause-shift claim on.

  1.1 Component-sum consistency: do CARRIER_DELAY + WEATHER_DELAY + NAS_DELAY
      + SECURITY_DELAY + LATE_AIRCRAFT_DELAY actually sum to ARR_DELAY for
      flights marked as delayed?
  1.2 Route-carrier cause-distribution stability: are dominant-cause labels
      for a given (ORIGIN, DEST, CARRIER) triple more concentrated than
      random chance would produce (measured via entropy vs. a shuffled null)?
  1.3 Rotation-plausibility cross-check: for flights blamed on
      LATE_AIRCRAFT_DELAY, was the same aircraft's previous leg (from
      scripts/05_build_rotation_feature.py) actually late?

Reads from monthly_enriched/ (must be run after 05_build_rotation_feature.py).
Writes a JSON report; prints a plain-language summary.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ENRICHED_DIR = Path("./data_processed/monthly_enriched")
OUT_PATH = Path("./data_processed/cause_label_consistency_report.json")

CAUSE_COLS = ["CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"]
NEEDED_COLS = [
    "ARR_DELAY", "ARR_DEL15", "ORIGIN", "DEST", "OP_UNIQUE_CARRIER",
    "ROTATION_PREV_ARR_DELAY", "ROTATION_HAS_PREV_LEG", "ROTATION_SAME_DAY",
] + CAUSE_COLS


def load_all_delayed() -> pd.DataFrame:
    files = sorted(ENRICHED_DIR.glob("*.parquet"))
    print(f"Loading delayed-flight rows from {len(files)} months...", flush=True)
    frames = []
    for f in files:
        df = pd.read_parquet(f, columns=NEEDED_COLS)
        df = df[df["ARR_DEL15"] == 1]
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    print(f"Total delayed flights: {len(combined):,}", flush=True)
    return combined


def check_component_sum(df: pd.DataFrame) -> dict:
    print("\n=== 1.1 Component-sum consistency ===", flush=True)
    cause_sum = df[CAUSE_COLS].fillna(0).sum(axis=1)
    residual = (df["ARR_DELAY"] - cause_sum).abs()
    all_causes_missing = df[CAUSE_COLS].isna().all(axis=1)

    checkable = ~all_causes_missing
    mismatch = checkable & (residual > 1.0)
    result = {
        "delayed_flights_total": int(len(df)),
        "flights_with_no_cause_breakdown_at_all": int(all_causes_missing.sum()),
        "checkable_flights": int(checkable.sum()),
        "mismatches_over_1min": int(mismatch.sum()),
        "mismatch_rate_pct": float(mismatch.sum() / checkable.sum() * 100) if checkable.sum() else None,
        "median_residual_minutes": float(residual[checkable].median()) if checkable.sum() else None,
    }
    print(f"  {result['checkable_flights']:,} of {result['delayed_flights_total']:,} delayed flights have a cause breakdown to check")
    print(f"  mismatches (>1 min off): {result['mismatches_over_1min']:,} ({result['mismatch_rate_pct']:.3f}%)")
    return result


def check_route_carrier_stability(df: pd.DataFrame, min_count: int = 30, n_shuffles: int = 20) -> dict:
    print("\n=== 1.2 Route-carrier cause-distribution stability ===", flush=True)
    sub = df.dropna(subset=CAUSE_COLS, how="all").copy()
    sub[CAUSE_COLS] = sub[CAUSE_COLS].fillna(0)
    sub["DOMINANT_CAUSE"] = sub[CAUSE_COLS].idxmax(axis=1)

    def entropy(labels: pd.Series) -> float:
        p = labels.value_counts(normalize=True).values
        return float(-(p * np.log2(p + 1e-12)).sum())

    group_key = ["ORIGIN", "DEST", "OP_UNIQUE_CARRIER"]
    counts = sub.groupby(group_key, observed=True).size()
    eligible_groups = counts[counts >= min_count].index

    print(f"  {len(eligible_groups):,} route-carrier triples have >= {min_count} delayed flights", flush=True)

    eligible_mask = sub.set_index(group_key).index.isin(eligible_groups)
    eligible = sub[eligible_mask]

    real_entropy = eligible.groupby(group_key, observed=True)["DOMINANT_CAUSE"].apply(entropy)
    real_mean_entropy = float(real_entropy.mean())

    shuffled_means = []
    rng = np.random.default_rng(42)
    for _ in range(n_shuffles):
        shuffled_labels = eligible["DOMINANT_CAUSE"].sample(frac=1.0, random_state=rng.integers(1e9)).values
        shuffled = eligible.assign(_SHUFFLED=shuffled_labels)
        shuffled_entropy = shuffled.groupby(group_key, observed=True)["_SHUFFLED"].apply(entropy)
        shuffled_means.append(float(shuffled_entropy.mean()))

    shuffled_mean = float(np.mean(shuffled_means))
    shuffled_std = float(np.std(shuffled_means))
    z_score = (real_mean_entropy - shuffled_mean) / shuffled_std if shuffled_std > 0 else None

    result = {
        "eligible_route_carrier_triples": int(len(eligible_groups)),
        "real_mean_entropy": real_mean_entropy,
        "shuffled_null_mean_entropy": shuffled_mean,
        "shuffled_null_std_entropy": shuffled_std,
        "z_score": z_score,
        "interpretation": (
            "Real cause labels are MORE concentrated (lower entropy) than random shuffling would produce -- "
            "evidence the labels carry real, route-specific signal."
            if z_score is not None and z_score < -2 else
            "Real cause labels are NOT reliably more concentrated than random shuffling -- weak or no evidence "
            "of route-specific signal; treat RQ2 claims cautiously."
        ),
    }
    print(f"  real mean entropy: {real_mean_entropy:.4f}  |  shuffled null: {shuffled_mean:.4f} +/- {shuffled_std:.4f}")
    print(f"  z-score: {z_score:.2f}" if z_score is not None else "  z-score: n/a")
    print(f"  -> {result['interpretation']}")
    return result


def check_rotation_plausibility(df: pd.DataFrame) -> dict:
    print("\n=== 1.3 Rotation-plausibility cross-check ===", flush=True)
    sub = df.dropna(subset=CAUSE_COLS, how="all").copy()
    sub[CAUSE_COLS] = sub[CAUSE_COLS].fillna(0)
    sub["DOMINANT_CAUSE"] = sub[CAUSE_COLS].idxmax(axis=1)

    blamed_on_prev = sub[sub["DOMINANT_CAUSE"] == "LATE_AIRCRAFT_DELAY"]
    has_prev_leg = blamed_on_prev["ROTATION_HAS_PREV_LEG"] == True  # noqa: E712

    checkable = blamed_on_prev[has_prev_leg]
    plausible = checkable["ROTATION_PREV_ARR_DELAY"] >= 10.0  # previous leg itself meaningfully late
    violation_rate = float((~plausible).mean()) if len(checkable) else None

    result = {
        "flights_blamed_on_late_aircraft": int(len(blamed_on_prev)),
        "of_those_with_a_trackable_previous_leg": int(len(checkable)),
        "of_those_missing_a_trackable_previous_leg": int(len(blamed_on_prev) - len(checkable)),
        "plausibility_violation_rate_pct": violation_rate * 100 if violation_rate is not None else None,
        "median_previous_leg_delay_minutes": float(checkable["ROTATION_PREV_ARR_DELAY"].median()) if len(checkable) else None,
    }
    print(f"  {result['flights_blamed_on_late_aircraft']:,} flights blamed on LATE_AIRCRAFT_DELAY")
    print(f"  {result['of_those_with_a_trackable_previous_leg']:,} have a trackable previous leg")
    print(f"  plausibility violations (previous leg was NOT meaningfully late): {result['plausibility_violation_rate_pct']:.2f}%")
    return result


def main():
    df = load_all_delayed()
    report = {
        "component_sum_check": check_component_sum(df),
        "route_carrier_stability_check": check_route_carrier_stability(df),
        "rotation_plausibility_check": check_rotation_plausibility(df),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
