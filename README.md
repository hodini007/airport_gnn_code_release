# Code for "Beyond the Graph: Cause-Aware Flight Delay Propagation Prediction with Conformal Prediction and Counterfactual Explanation"

Anonymized code release accompanying a manuscript under review. It contains the data pipeline, model, baselines and analyses; it does not contain data, trained checkpoints or the manuscript.

## Data
All flight data come from the public U.S. Bureau of Transportation Statistics On-Time Performance dataset (https://www.transtats.bts.gov), January 2019 to June 2025. The raw files and intermediate outputs are not included; scripts 01-05 build them from the raw BTS downloads (see each script's docstring for the expected input folder).

## Setup
```
pip install -r requirements.txt   # install the PyTorch build for your CUDA version first
```
Run every script from the repository root, e.g. `python scripts/12_train_model.py --encoder full --epochs 15`. Paths are relative to the root (`data_processed/` is created and read there).

## Pipeline order
| Script | Purpose |
|---|---|
| 01-05 | Inventory, clean and merge BTS files; verify against the official download; build aircraft-rotation features |
| 06 | Cause-label reliability checks |
| 07-10 | Node/edge aggregation, airport selection and edge weights, graph sequence, previous-leg cause feature |
| 11 | Forward-pass smoke test |
| 12-13 | Train (`--encoder full / gat_only / lstm_only / ...`) and evaluate the multi-task model |
| 14-16 | Head C probability diagnostic, GAT attention diagnostic, cross-airport correlation analysis |
| 17, 23 | XGBoost baseline; constant reference baselines |
| 18, 22 | Split conformal prediction (overall and per test month) |
| 19 | Counterfactual explainer |
| 20-21 | Cause-shift transition analysis and null-model baselines |

Package `model/` holds the encoder, prediction heads and data loading.
