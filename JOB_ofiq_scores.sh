#!/bin/bash

#$ -M oineza@nd.edu        # Email address for job notification
#$ -m ae                     # Send mail when job ends and aborts
#$ -q gpu@qa-rtx6k-017       # Specify queue (open GPU node from free_gpus.sh @cvrl)
#$ -l gpu_card=4             # Number of GPUs
#$ -N out_OFIQ_scores        # Specify job name

# conda needs to be initialized in a fresh batch shell before `conda activate`
source /software/c/conda/26.3.2/etc/profile.d/conda.sh
module load cuda/11.8
conda activate myenviroment

# Run OFIQ on all 70,000 FFHQ images
ofiq_project/OFIQ-Project/install_x86_64_linux/Release/bin/OFIQSampleApp \
    -c ofiq_project/OFIQ-Project/data/ofiq_config.jaxn \
    -i ffhq_all \
    -o ffhq_all_results.csv

echo "OFIQ run finished at: $(date)"
 