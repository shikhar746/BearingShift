#!/usr/bin/env bash
# End-to-end pipeline check on SYNTHETIC data. Outputs go to results/smoke/ (git-ignored)
# and are not results: they only show that every step runs.
set -euo pipefail
python scripts/make_synthetic_data.py --config configs/smoke.yaml
python scripts/build_index.py --config configs/smoke.yaml
python scripts/run_experiment.py --config configs/smoke.yaml
