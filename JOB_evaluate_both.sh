#!/bin/bash

#$ -M oineza@nd.edu
#$ -m ae
#$ -q gpu@@crc_a10
#$ -l gpu_card=1
#$ -pe smp 8
#$ -N eval_both_epochs
#$ -cwd
#$ -o eval_both_epochs.o$JOB_ID
#$ -e eval_both_epochs.e$JOB_ID

module load pytorch/2.9.1
cd "/users/oineza/Summer Project" || exit 1

# matplotlib was missing during training — install for curve + scatter plots
pip install --user matplotlib -q

echo "=== Regenerating training curves from train_log_full.csv ==="
python3 - <<'PY'
import pandas as pd
from train_quality import plot_curves
rows = pd.read_csv("train_log_full.csv").to_dict("records")
plot_curves(rows, "train_curve_full.png")
PY

echo "=== Evaluating epoch-40 checkpoint (best_model_full.pt) ==="
python3 evaluate.py \
    --model best_model_full.pt \
    --plot eval_scatter_full_ep40.png \
    --train-plot eval_scatter_full_ep40_train.png

echo "=== Reproducing epoch-38 weights (same hyperparams, stop at 38) ==="
python3 train_quality.py \
    --csv ffhq_all_results.csv \
    --root . \
    --target UnifiedQualityScore.native \
    --arch resnet_small \
    --loss combined \
    --tail-weight 0.5 \
    --aux 10 \
    --aux-weight 0.3 \
    --img-size 256 \
    --grad-clip 1.0 \
    --lr 1e-3 \
    --epochs 38 \
    --batch-size 64 \
    --workers 8 \
    --out best_model_epoch38.pt \
    --log train_log_epoch38.csv \
    --curve ""

echo "=== Evaluating epoch-38 checkpoint (best_model_epoch38.pt) ==="
python3 evaluate.py \
    --model best_model_epoch38.pt \
    --plot eval_scatter_full_ep38.png \
    --train-plot eval_scatter_full_ep38_train.png

echo "All evaluation finished at: $(date)"
