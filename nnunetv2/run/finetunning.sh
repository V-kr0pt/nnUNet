#!/bin/bash

for FOLD in 0 1 2 3 4; do
    echo "==============================================="
    echo "Starting training for fold $FOLD"
    echo "Finetuning dataset: Dataset996_BreastPectoralSegmentation_finetunning"
    echo "Pretrained weights: nnUNet_results/Dataset995_BreastPectoralSegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_$FOLD/checkpoint_best.pth"
    echo "==============================================="

    nnUNetv2_train Dataset996_BreastPectoralSegmentation_finetunning 3d_fullres $FOLD \
        -pretrained_weights nnUNet_results/Dataset995_BreastPectoralSegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_$FOLD/checkpoint_best.pth

    echo "Training finished for fold $FOLD :)"
    echo ""
done
