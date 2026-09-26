#!/usr/bin/env bash
# LoCoMo is redistributed by its authors; fetch rather than vendor it.
set -euo pipefail
mkdir -p data
curl -fsSL -o data/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
echo "fetched data/locomo10.json ($(wc -c < data/locomo10.json) bytes)"
