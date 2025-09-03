#!/bin/bash
set -euo pipefail

# Lista de nomes de experimentos
TRAINERS=(
    "MyTrainer_LRWarmup5"
    "MyTrainer_LRWarmup10"
    "MyTrainer_LR1e4_nowarmup"
)

for TRAINER in "${TRAINERS[@]}"; do
    echo ">>> Running trainer: $TRAINER"
    ./nnunetv2/finetunnig/finetune.sh "$TRAINER"
done
