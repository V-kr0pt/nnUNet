#!/bin/bash
set -euo pipefail

FINETUNE_DS_ID=996
PREV_DS_ID=995
FINETUNE_DS="Dataset996_BreastPectoralSegmentation_finetunning"
PREV_DS="Dataset995_BreastPectoralSegmentation"


echo ">>> Preparando dataset $FINETUNE_DS_ID para fine-tuning a partir de $PREV_DS_ID..."
nnUNetv2_plan_and_preprocess -d "$FINETUNE_DS_ID"
nnUNetv2_extract_fingerprint -d "$PREV_DS_ID"
nnUNetv2_move_plans_between_datasets -s "$PREV_DS" -t "$FINETUNE_DS" -sp nnUNetPlans -tp PlansFrom$PREV_DS_ID
nnUNetv2_preprocess -d "$FINETUNE_DS_ID" -plans_name PlansFrom$PREV_DS_ID
echo ">>> Preparo finalizado! Dataset $FINETUNE_DS_ID está pronto para treinamento."
