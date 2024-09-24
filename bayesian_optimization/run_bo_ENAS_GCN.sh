#!/bin/bash

#SBATCH --job-name=GCN_bo           # Job name
#SBATCH --output=GCN_output_%j.txt # Standard output log, %j will be replaced with job ID
#SBATCH --error=GCN_error_%j.txt   # Standard error log, %j will be replaced with job ID
#SBATCH --ntasks=1                 # Run on a single CPU
#SBATCH --mem=10G                        # Memory limit
#SBATCH --time=3:00:00                 # Time limit hrs:min:sec
#SBATCH --gres=gpu:1            # Request one GPU
#SBATCH --partition=preempt    # Specify the partition

# export PYTHONPATH="$(pwd)"
echo "$(pwd)"
python /scratch/hajorlou/D-VAE/bayesian_optimization/bo.py \
  --data-name final_structures6 \
  --save-appendix GCN \
  --checkpoint 140 \
  --res-dir="ENAS_results/" \
  --BO-rounds 10 \
  --BO-batch-size 50 

  #--save-appendix SVAE \
  #--save-appendix GraphRNN \
  #--save-appendix GCN \
  #--save-appendix DeepGMG \
  #--save-appendix DVAE_fast \

