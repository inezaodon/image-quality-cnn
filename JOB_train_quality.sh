#!/bin/bash

#$ -M aogunley@nd.edu        # Email address for job notification
#$ -m ae                     # Send mail when job ends and aborts
#$ -q gpu                    # General GPU queue: lands on any FREE node (not the OFIQ node)
#$ -l gpu_card=1             # Number of GPUs (1 is plenty for this model)
#$ -N train_quality          # Job name

# Use the CRC-provided PyTorch module (the conda env 'myenviroment' is empty).
module load pytorch/2.9.1

# Show which GPU(s) we actually got, for the log
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Train the quality-prediction CNN on whatever OFIQ has scored so far.
python train_quality.py \
    --csv ffhq_all_results.csv \
    --root . \
    --target UnifiedQualityScore.native \
    --arch simple \
    --epochs 40 \
    --patience 8 \
    --batch-size 64 \
    --workers 8 \
    --out best_model.pt \
    --log train_log.csv \
    --curve train_curve.png

echo "Training finished at: $(date)"
