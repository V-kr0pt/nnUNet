#!/bin/bash
set -euo pipefail

FINETUNE_DS="Dataset996_BreastPectoralSegmentation_finetunning"
PREV_DS="Dataset995_BreastPectoralSegmentation"
CONFIG="3d_fullres"

# Experimento deve ser passado como argumento
if [ $# -lt 1 ]; then
    echo "Usage: $0 <TRAINER_NAME>"
    exit 1
fi

TRAINER_NAME="$1"

for FOLD in 0 1 2 3 4; do #0 1 2 3 4; do
    CKPT="$nnUNet_results/$PREV_DS/nnUNetTrainer__nnUNetPlans__${CONFIG}/fold_${FOLD}/checkpoint_best.pth"

    echo "============================================================"
    echo "Starting training for fold ${FOLD}"
    echo "Finetuning dataset : $FINETUNE_DS"
    echo "Configuration      : $CONFIG"
    echo "Trainer name    : $TRAINER_NAME"
    echo "Pretrained weights : $CKPT"
    echo "Results will go to : $nnUNet_results/$FINETUNE_DS/MyTrainer_LRWarmup__PlansFrom995__${CONFIG}__${TRAINER_NAME}/fold_${FOLD}"
    echo "============================================================"

    if [ ! -f "$CKPT" ]; then
        echo "WARNING: $CKPT not found. Trying checkpoint_final.pth ..."
        CKPT_ALT="$nnUNet_results/$PREV_DS/nnUNetTrainer__nnUNetPlans__${CONFIG}/fold_${FOLD}/checkpoint_final.pth"
        if [ -f "$CKPT_ALT" ]; then
            CKPT="$CKPT_ALT"
            echo "Using: $CKPT"
        else
            echo "ERROR: No pretrained checkpoint found for fold ${FOLD}."
            exit 1
        fi
    fi

    set -x
    nnUNetv2_train "$FINETUNE_DS" "$CONFIG" "$FOLD" \
        -p PlansFrom995 \
        -tr "$TRAINER_NAME" \
        -pretrained_weights "$CKPT" 
    set +x

    echo "Training finished for fold ${FOLD} (experiment $TRAINER_NAME)"
    echo
done
