"""
RQ2 follow-up: quantify how much of the observed cause-shift pattern
(scripts/20_cause_shift_analysis.py) is explained by BTS's own mechanical
per-leg labeling convention versus genuine cause-specific persistence.

Raised directly by an external review of the manuscript: the majority
transition pattern (most propagating chains ending up relabeled as
LATE_AIRCRAFT_DELAY) is a known, near-deterministic consequence of how BTS
attributes cause -- "this flight's own aircraft was late" is mechanically
true of any prior delay, regardless of root cause. A reader could
reasonably ask how much of the headline "cause changes in 58.4% of chains"
number is just a restatement of that convention. This script answers that
directly with two stylized null models, computed from the same population
used in script 20 (3,507,468 same-aircraft chains where both legs were
delayed with a real attributed cause):

  Null A -- independence baseline: if the current leg's cause were drawn
  independently of the previous leg's cause (according to the population's
  own marginal frequencies), how often would they coincidentally match?
  This is the "pure chance" floor for the same-cause persistence rate.

  Null B -- pure mechanical relabeling: if BTS's per-leg convention were
  fully deterministic (every propagating chain relabels to
  LATE_AIRCRAFT_DELAY, no exceptions), what same-cause rate and
  non-late-aircraft -> late-aircraft rate would that predict? This is the
  "pure mechanism, no genuine persistence" account.

Comparing the actually observed rates (script 20) against both nulls
isolates the residual, cause-specific persistence (e.g. weather-to-weather)
that neither null explains.
"""
import json
from pathlib import Path

import pandas as pd

ENRICHED_DIR = Path("./data_processed/monthly_enriched")
OUT_PATH = Path("./data_processed/cause_shift_null_model_report.json")
EXISTING_REPORT = Path("./data_processed/cause_shift_report.json")

REAL_CAUSES = ["CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"]
NEEDED_COLS = ["OWN_DOMINANT_CAUSE", "ROTATION_PREV_CAUSE"]


def load_propagating_chains() -> pd.DataFrame:
    files = sorted(ENRICHED_DIR.glob("*.parquet"))
    print(f"Loading cause columns from {len(files)} months...", flush=True)
    frames = []
    for f in files:
        df = pd.read_parquet(f, columns=NEEDED_COLS)
        df = df[df["OWN_DOMINANT_CAUSE"].isin(REAL_CAUSES) & df["ROTATION_PREV_CAUSE"].isin(REAL_CAUSES)]
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    print(f"Chains: {len(combined):,}", flush=True)
    return combined


def main():
    df = load_propagating_chains()
    prev = df["ROTATION_PREV_CAUSE"].astype(str)
    curr = df["OWN_DOMINANT_CAUSE"].astype(str)

    prev_marginal = prev.value_counts(normalize=True).reindex(REAL_CAUSES)
    curr_marginal = curr.value_counts(normalize=True).reindex(REAL_CAUSES)

    with open(EXISTING_REPORT) as f:
        existing = json.load(f)
    observed_same_cause_rate = existing["same_cause_persistence_rate"]
    observed_non_la_becomes_la = existing["non_late_aircraft_prev_becomes_late_aircraft_rate"]

    # --- Null A: independence baseline ---
    null_a_same_cause_rate = float((prev_marginal * curr_marginal).sum())

    # --- Null B: pure mechanical relabeling (every propagating chain -> LATE_AIRCRAFT_DELAY) ---
    null_b_same_cause_rate = float(prev_marginal["LATE_AIRCRAFT_DELAY"])
    null_b_non_la_becomes_la = 1.0  # deterministic under this null

    print("\n" + "=" * 70)
    print("Previous-cause marginal distribution (propagating chains):")
    print(prev_marginal.round(4).to_string())
    print("\nCurrent-cause marginal distribution:")
    print(curr_marginal.round(4).to_string())

    print(f"\nObserved same-cause persistence rate:      {observed_same_cause_rate:.4f}")
    print(f"Null A (independence/chance) predicts:      {null_a_same_cause_rate:.4f}")
    print(f"  -> observed is {observed_same_cause_rate / null_a_same_cause_rate:.2f}x the chance baseline")
    print(f"Null B (pure mechanical relabeling) predicts: {null_b_same_cause_rate:.4f}")
    print(f"  -> observed exceeds pure-mechanical-null by {(observed_same_cause_rate - null_b_same_cause_rate)*100:.1f} points")

    print(f"\nObserved non-late-aircraft -> late-aircraft rate: {observed_non_la_becomes_la:.4f}")
    print(f"Null B (pure mechanical) predicts:                 {null_b_non_la_becomes_la:.4f} (deterministic)")
    print(f"  -> {(null_b_non_la_becomes_la - observed_non_la_becomes_la)*100:.1f} points of propagating chains do NOT "
          f"follow the pure-mechanical rule (they keep their original cause or shift elsewhere)")
    print("=" * 70)

    report = {
        "previous_cause_marginal": prev_marginal.round(6).to_dict(),
        "current_cause_marginal": curr_marginal.round(6).to_dict(),
        "observed_same_cause_persistence_rate": observed_same_cause_rate,
        "observed_non_late_aircraft_becomes_late_aircraft_rate": observed_non_la_becomes_la,
        "null_a_independence": {
            "predicted_same_cause_rate": null_a_same_cause_rate,
            "observed_over_null_ratio": observed_same_cause_rate / null_a_same_cause_rate,
        },
        "null_b_pure_mechanical": {
            "predicted_same_cause_rate": null_b_same_cause_rate,
            "predicted_non_late_aircraft_becomes_late_aircraft_rate": null_b_non_la_becomes_la,
            "observed_minus_null_same_cause_pts": (observed_same_cause_rate - null_b_same_cause_rate) * 100,
            "null_minus_observed_non_la_becomes_la_pts": (null_b_non_la_becomes_la - observed_non_la_becomes_la) * 100,
        },
    }
    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
