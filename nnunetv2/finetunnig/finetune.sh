#!/bin/bash
set -euo pipefail

FINETUNE_DS="Dataset996_BreastPectoralSegmentation_finetunning"
PREV_DS="Dataset995_BreastPectoralSegmentation"
CONFIG="3d_fullres"
LOG_DIR="logs_finetune_996"
mkdir -p "$LOG_DIR"

# Experimento deve ser passado como argumento
if [ $# -lt 1 ]; then
    echo "Usage: $0 <EXP_NAME>"
    exit 1
fi

EXP_NAME="$1"

for FOLD in 0; do
    CKPT="$nnUNet_results/$PREV_DS/nnUNetTrainer__nnUNetPlans__${CONFIG}/fold_${FOLD}/checkpoint_best.pth"

    echo "============================================================"
    echo "Starting training for fold ${FOLD}"
    echo "Finetuning dataset : $FINETUNE_DS"
    echo "Configuration      : $CONFIG"
    echo "Experiment name    : $EXP_NAME"
    echo "Pretrained weights : $CKPT"
    echo "Results will go to : $nnUNet_results/$FINETUNE_DS/MyTrainer_LRWarmup__PlansFrom995__${CONFIG}__${EXP_NAME}/fold_${FOLD}"
    echo "Log file           : $LOG_DIR/fold_${FOLD}_${EXP_NAME}.log"
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
        -tr MyTrainer_LRWarmup \
        -pretrained_weights "$CKPT" \
        --results_identifier "$EXP_NAME" \
        2>&1 | tee "$LOG_DIR/fold_${FOLD}_${EXP_NAME}.log"
    set +x

    echo "Training finished for fold ${FOLD} (experiment $EXP_NAME)"
    echo
done
