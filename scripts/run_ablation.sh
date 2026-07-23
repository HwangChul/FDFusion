#!/usr/bin/env bash

# Run from any directory.  Each mode receives a separate experiment directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

for ablation in wo_teb wo_ceb wo_clma; do
    echo "Starting ablation: ${ablation}"
    python train_LLVIP.py --ablation "$ablation" --module "$ablation"
done

/usr/bin/shutdown -h now