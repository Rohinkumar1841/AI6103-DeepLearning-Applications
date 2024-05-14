#!/bin/bash
#SBATCH --partition=SCSEGPU_M1
#SBATCH --qos=q_amsai
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=1G
#SBATCH --job-name=Myjob
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err


module load anaconda3/23.5.2
eval "$(conda shell.bash hook)"
conda activate DL


export CUBLAS_WORKSPACE_CONFIG=:16:8


python assignment1.py \
--dataset_dir ./datasets \
--batch_size 128 \
--epochs 300 \
--lr 0.05  \
--mixup \
--seed 0 \
--lr_scheduler \
--wd 0.00001 \
--fig_name "lr= Weight_2"
--test
