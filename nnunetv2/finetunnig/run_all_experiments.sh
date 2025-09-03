#!/bin/bash
set -euo pipefail

# Lista de nomes de experimentos
EXPERIMENTS=(
    "lr3e-4_warmup5"
    "lr3e-4_warmup10"
    "lr1e-4_nowarmup"
)

for EXP in "${EXPERIMENTS[@]}"; do
    echo ">>> Running experiment: $EXP"
    ./finetune.sh "$EXP"
done
