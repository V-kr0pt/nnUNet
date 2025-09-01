#!/bin/bash
set -euo pipefail

FINETUNE_DS="Dataset996_BreastPectoralSegmentation_finetunning"
PREV_DS="Dataset995_BreastPectoralSegmentation"
CONFIG="3d_fullres"
LOG_DIR="logs_finetune_996"
mkdir -p "$LOG_DIR"

for FOLD in 0 1 2 3 4; do
    CKPT="$nnUNet_results/$PREV_DS/nnUNetTrainer__nnUNetPlans__${CONFIG}/fold_${FOLD}/checkpoint_best.pth"

    echo "============================================================"
    echo "Starting training for fold ${FOLD}"
    echo "Finetuning dataset : $FINETUNE_DS"
    echo "Configuration      : $CONFIG"
    echo "Pretrained weights : $CKPT"
    echo "Results will go to : $nnUNet_results/$FINETUNE_DS/nnUNetTrainer__PlansFrom995__${CONFIG}/fold_${FOLD}"
    echo "Log file           : $LOG_DIR/fold_${FOLD}.log"
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
    nnUNetv2_train "$FINETUNE_DS" "$CONFIG" "$FOLD" -p PlansFrom995\
        -pretrained_weights "$CKPT" 2>&1 | tee "$LOG_DIR/fold_${FOLD}.log"
    set +x

    echo "Training finished for fold ${FOLD}"
    echo
done