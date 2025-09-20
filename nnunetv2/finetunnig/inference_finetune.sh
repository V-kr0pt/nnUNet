#!/bin/bash
set -euo pipefail

#TRAINERS: "nnUNetTrainer", "MyTrainer_LR", "MyTrainer_LRWarmup5", "MyTrainer_LRWarmup10", "MyTrainer_LR1e4_nowarmup"
TRAINERS=(
    "nnUNetTrainer"
    "MyTrainer_LR"
    "MyTrainer_LRWarmup5"
    "MyTrainer_LRWarmup10"
    "MyTrainer_LR1e4_nowarmup"
)
DATASET_ID=996 # 995
#INPUT_NAME= "bad_performance_imgs", "birads4_val_inputs"
INPUT_NAME="bad_performance_imgs"
INPUT_FOLDER="../media/${INPUT_NAME}"

for TRAINER in "${TRAINERS[@]}"; do
    echo ">>> Running inferece from trainer: $TRAINER"

    OUTPUT_FOLDER="../media/finetune_outputs/output_${DATASET_ID}_${TRAINER}_${INPUT_NAME}"

    FOLD="all"  # Pode ser 0,1,2,3,4 ou all
    SKIP_PREPROCESSING=false
    SKIP_POSTPROCESSING=false
    RESTART_PREPROCESSING=false
    FORCE=true

    cmd="python nnunetv2/run/run_inference.py \
        -i $INPUT_FOLDER \
        -o $OUTPUT_FOLDER \
        -did $DATASET_ID \
        -tr $TRAINER \
        -f $FOLD \
        $( [ "$SKIP_PREPROCESSING" = true ] && echo --skip_pre ) \
        $( [ "$SKIP_POSTPROCESSING" = true ] && echo --skip_post ) \
        $( [ "$RESTART_PREPROCESSING" = true ] && echo --restart_preprocess ) \
        $( [ "$FORCE" = true ] && echo --force )"
    #echo "executing: $cmd"
    eval $cmd
done