#!/usr/bin/env bash
# One-command reproduction of the MVP results on the real Paderborn data.
# Prerequisites: pip install -r requirements.txt && pip install -e .
#                an unrar-capable tool (unrar, unar, 7z or bsdtar) for extraction.
set -euo pipefail
CONFIG=${1:-configs/default.yaml}

python scripts/download_paderborn.py --config "$CONFIG"           # list files + total size
read -r -p "Download these files? [y/N] " answer
[[ "$answer" == "y" ]] || exit 1
python scripts/download_paderborn.py --config "$CONFIG" --yes --extract
python scripts/build_index.py --config "$CONFIG"
python scripts/run_experiment.py --config "$CONFIG"
