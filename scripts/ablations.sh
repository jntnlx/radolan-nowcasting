#!/usr/bin/env bash
# Matched-budget experiments: each configuration trains for the same wall-clock
# time and is then evaluated on the full domain (runs/eval/val_120/<run>.json).
#
#   bash scripts/ablations.sh [hours]
#
# lag_v3_film           reference: optical-flow first guess, FiLM noise, multiscale CRPS
# lag_v3_eulerian       no advection: all change must come from the source term
# lag_v3_deterministic  one member: CRPS reduces to the mean absolute error

set -u
cd "$(dirname "$0")/.."
hours=${1:-3}

run() {
    local name=$1; shift
    python -m nowcast.train --run "$name" --hours "$hours" --lr 2e-4 --compile --workers 6 "$@" > "runs/$name.log" 2>&1 \
        && python -m nowcast.evaluate --method "runs/$name/best.pt" --cases 120 > "runs/eval_$name.log" 2>&1 \
        || echo "$name failed" >> runs/ablations.failed
}

run lag_v3_film
run lag_v3_eulerian --eulerian
run lag_v3_deterministic --members 1
