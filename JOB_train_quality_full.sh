#!/bin/bash

#$ -M aogunley@nd.edu        # Email address for job notification
#$ -m ae                     # Send mail when job ends and aborts
#$ -q gpu@@crc_a10           # A10 (Ampere) nodes only — Pascal Titan Xp nodes fail with CUDA 13
#$ -l gpu_card=1             # Number of GPUs
#$ -pe smp 8                 # 8 CPU cores so the data loaders keep the GPU fed on 70k images
#$ -N train_quality_full     # Job name

# Use the CRC-provided PyTorch module (the conda env 'myenviroment' is empty).
module load pytorch/2.9.1

# Show which GPU we actually got, for the log
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Re-train on the COMPLETE OFIQ output (all images scored by the time this runs).
# The script auto-filters to rows whose image exists and drops any failed/NaN rows,
# so it transparently handles ffhq_MISSING_list.txt cases.
# Uses SmallResNet (resnet_small, ~0.5M params) with spatial dropout and
# UnifiedQualityScore.native (the raw linear measure) to reduce overfitting.
python train_quality.py \
    --csv ffhq_all_results.csv \
    --root . \
    --target UnifiedQualityScore.native \
    --arch resnet_small \
    --loss combined \
    --grad-clip 1.0 \
    --lr 1e-3 \
    --epochs 40 \
    --patience 8 \
    --batch-size 64 \
    --workers 8 \
    --out best_model_full.pt \
    --log train_log_full.csv \
    --curve train_curve_full.png

echo "FULL retrain finished at: $(date)"
