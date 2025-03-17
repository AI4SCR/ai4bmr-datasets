#!/bin/bash
#SBATCH --job-name=MAE-imagenet
#SBATCH --output=/users/amarti51/logs/MAE-imagenet-%A-%a.log
#SBATCH --error=/users/amarti51/logs/MAE-imagenet-%A-%a.err
#SBATCH --account=mrapsoma_prometex

#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# Load modules
module load python/3.12.1
source /work/FAC/FBM/DBC/mrapsoma/prometex/.venvs/steinbock/bin/activate

DATASET_DIR=/users/amarti51/data/datasets/NSCLC
cd "$DATASET_DIR" || exit

steinbock preprocess imc images
