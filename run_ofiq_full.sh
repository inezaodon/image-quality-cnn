#!/bin/bash
#$ -q long
#$ -N ofiq_70k
#$ -l h_rt=24:00:00
#$ -o /users/aogunley/ofiq_full.out
#$ -e /users/aogunley/ofiq_full.err
#$ -M aogunley@nd.edu
#$ -m abe

# Re-setup environment (batch jobs start fresh)
export PATH=$HOME/.local/bin:$PATH
source /software/c/conda/26.3.2/etc/profile.d/conda.sh
conda activate myenviroment
module load cmake/3.26.4

# Run OFIQ on all 70k images
cd /users/aogunley/ofiq_project/OFIQ-Project/install_x86_64_linux/Release/bin/
./OFIQSampleApp \
    -c ../../../data/ofiq_config.jaxn \
    -i /users/aogunley/ffhq_all/ \
    -o /users/aogunley/ffhq_all_results.csv

echo "OFIQ run finished at: $(date)"
