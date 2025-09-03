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
    
    if [[ "$EXP" == "lr3e-4_warmup5" ]]; then
        export NNUNET_USE_NO_WARMUP=False
        export NNUNET_INITIAL_LR=0.0003
        export NNUNET_FREEZE_EPOCHS=5
    elif [[ "$EXP" == "lr3e-4_warmup10" ]]; then
        export NNUNET_USE_NO_WARMUP=False
        export NNUNET_INITIAL_LR=0.0003
        export NNUNET_FREEZE_EPOCHS=10
    elif [[ "$EXP" == "lr1e-4_nowarmup" ]]; then
        export NNUNET_USE_NO_WARMUP=True
        export NNUNET_INITIAL_LR=0.0001
        export NNUNET_FREEZE_EPOCHS=0
    else
        echo "Unknown experiment: $EXP"
        exit 1
    fi

    ./finetune.sh "$EXP"
done
