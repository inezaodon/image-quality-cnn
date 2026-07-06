#!/bin/bash

#$ -M aogunley@nd.edu        # Email address for job notification
#$ -m ae                     # Send mail when job ends and aborts
#$ -q gpu@@crc_a10           # A10 (Ampere) nodes only — Pascal Titan Xp nodes fail with CUDA 13
#$ -l gpu_card=1             # Number of GPUs
#$ -pe smp 8                 # 8 CPU cores so data loaders keep the GPU fed
#$ -N train_quality_resnet   # Job name

module load pytorch/2.9.1
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# SmallResNet trained from scratch on the current partial OFIQ output.
# resnet_small (~0.5M params, spatial dropout) instead of resnet18 (~11M)
# to reduce overfitting.  No pretrained weights -> lr 1e-3 is fine.
python train_quality.py \
    --csv ffhq_all_results.csv \
    --root . \
    --target UnifiedQualityScore.native \
    --arch resnet_small \
    --lr 1e-3 \
    --epochs 40 \
    --patience 8 \
    --batch-size 64 \
    --workers 8 \
    --out best_model_resnet.pt \
    --log train_log_resnet.csv \
    --curve train_curve_resnet.png

echo "ResNet partial run finished at: $(date)"
