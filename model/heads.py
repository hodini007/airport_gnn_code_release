"""
Stage 6: the flight representation layer (bridging the encoder's per-airport
embeddings to per-flight predictions) and the three prediction heads.

Implements Model_Architecture_Specification.docx Section 3 (the flight
representation layer -- not specified anywhere before this document) and
Section 4, Step 4 (the three heads).
"""
import torch
import torch.nn as nn

# The 6 categories OWN_DOMINANT_CAUSE / ROTATION_PREV_CAUSE can take, per
# scripts/10_add_previous_cause_feature.py, plus the "no usable previous
# leg" case that only ROTATION_PREV_CAUSE uses. Fixed vocabulary, indices
# assigned once here and reused consistently at data-loading time.
CAUSE_VOCAB = [
    "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY",
    "LATE_AIRCRAFT_DELAY", "NOT_DELAYED", "NO_PREV_LEG",
]
CAUSE_TO_IDX = {c: i for i, c in enumerate(CAUSE_VOCAB)}

# Class weights for Head C's loss, addressing the rare-cause blind spot
# found in the first full evaluation (macro-F1 0.412 vs. accuracy 0.653 --
# security delay, 17,128 of 5,595,002 training-set delayed flights, scored
# 0.000 F1). Computed as sqrt(total / (n_classes * count)) from the
# training set (2019-2023) counts below, in the order Head C's 5 output
# classes are indexed (CAUSE_VOCAB[:5]):
#   CARRIER_DELAY=1,887,270  WEATHER_DELAY=209,378  NAS_DELAY=1,438,867
#   SECURITY_DELAY=17,128    LATE_AIRCRAFT_DELAY=2,042,359
# Plain inverse-frequency weighting would give security a ~65x weight,
# risking the same kind of instability just fixed in the NaN bug -- sqrt
# moderates that to ~8x, still a real uplift for rare classes without
# letting one class dominate the gradient.
#
# WEATHER_DELAY's weight below is reduced from the original sqrt value
# (2.3118) to 2.3118 / WEATHER_OVERSAMPLE_FACTOR (model/dataset.py) = 0.7706,
# now that scripts/14_diagnose_head_c_probabilities.py showed real signal
# for weather worth amplifying via actual repeated training examples rather
# than a bigger loss multiplier on the same single example -- the two
# mechanisms would otherwise compound to ~7x the original intended boost.
CAUSE_CLASS_WEIGHTS = [0.7700, 0.7706, 0.8819, 8.0828, 0.7402]
N_CAUSE_VOCAB = len(CAUSE_VOCAB)          # vocabulary for the PREVIOUS-cause input feature
N_CAUSE_CLASSES = 5                        # Head C only ever predicts one of the 5 real causes


class FlightRepresentation(nn.Module):
    """
    Model_Architecture_Specification.docx Section 3.2. For one flight,
    combines:
      - the origin airport's space+time embedding
      - the destination airport's space+time embedding
      - a learned carrier embedding
      - a learned embedding of the previous leg's dominant cause (the new
        input this document added specifically so Head C has direct signal
        for cause-shift, not just delay size)
      - the flight's own numeric features (prior-leg delay, turnaround
        time, scheduled elapsed time, distance)
    into one vector, which is what all three heads actually see.
    """

    def __init__(self, airport_emb_dim=64, n_carriers=25, carrier_emb_dim=8,
                 cause_emb_dim=4, flight_numeric_dim=4):
        super().__init__()
        self.carrier_embedding = nn.Embedding(n_carriers, carrier_emb_dim)
        self.prev_cause_embedding = nn.Embedding(N_CAUSE_VOCAB, cause_emb_dim)
        self.output_dim = airport_emb_dim * 2 + carrier_emb_dim + cause_emb_dim + flight_numeric_dim

    def forward(self, origin_emb, dest_emb, carrier_idx, prev_cause_idx, flight_numeric):
        """
        origin_emb, dest_emb : [batch, airport_emb_dim]
        carrier_idx          : [batch] long
        prev_cause_idx       : [batch] long
        flight_numeric       : [batch, flight_numeric_dim] (raw minute/distance
                                units, not normalized -- see model/dataset.py
                                FLIGHT_NUMERIC_COLS/build_flight_tensors)
        Returns: [batch, output_dim]
        """
        carrier_vec = self.carrier_embedding(carrier_idx)
        prev_cause_vec = self.prev_cause_embedding(prev_cause_idx)
        return torch.cat([origin_emb, dest_emb, carrier_vec, prev_cause_vec, flight_numeric], dim=-1)


class MultiTaskHeads(nn.Module):
    """
    Heads A (regression), B (binary classification), C (5-class,
    conditioned on delay). All three are small feed-forward networks on
    top of the same flight representation vector -- Model_Architecture_
    Specification.docx Section 4, Step 4.
    """

    def __init__(self, input_dim, hidden_dim=64, n_cause_classes=N_CAUSE_CLASSES, dropout=0.1):
        super().__init__()

        def make_head(out_dim):
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )

        self.head_a = make_head(1)                # regression: predicted delay, minutes
        self.head_b = make_head(1)                 # binary logit: P(ARR_DEL15 = 1)
        self.head_c = make_head(n_cause_classes)   # 5-class logits; loss-masked to delayed flights only

    def forward(self, flight_repr):
        return {
            "delay_minutes": self.head_a(flight_repr).squeeze(-1),
            "delay_risk_logit": self.head_b(flight_repr).squeeze(-1),
            "cause_logits": self.head_c(flight_repr),
        }


class UncertaintyWeightedLoss(nn.Module):
    """
    Kendall, Gal & Cipolla (2018) homoscedastic uncertainty weighting.
    Learns a per-task weight during training instead of the loss weights
    between Heads A/B/C being guessed by hand -- Model_Architecture_
    Specification.docx Section 6.
    """

    def __init__(self):
        super().__init__()
        # log-variance per task; starts at 0 (equal weighting) and adapts.
        self.log_var_a = nn.Parameter(torch.zeros(()))
        self.log_var_b = nn.Parameter(torch.zeros(()))
        self.log_var_c = nn.Parameter(torch.zeros(()))

    def forward(self, loss_a, loss_b, loss_c):
        # Clamp before exp() so an unattended multi-hour run can't drift a
        # log-variance far enough negative to overflow to inf (which would
        # poison every parameter the same way the Head C NaN bug did).
        # Range is generous (weights from ~e^-10 to e^10) -- this only
        # guards against runaway drift, not normal adaptation.
        log_var_a = self.log_var_a.clamp(-10, 10)
        log_var_b = self.log_var_b.clamp(-10, 10)
        log_var_c = self.log_var_c.clamp(-10, 10)
        total = (
            torch.exp(-log_var_a) * loss_a + log_var_a
            + torch.exp(-log_var_b) * loss_b + log_var_b
            + torch.exp(-log_var_c) * loss_c + log_var_c
        )
        return total
