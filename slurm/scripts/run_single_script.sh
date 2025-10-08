#!/bin/bash

#SBATCH -c 8
#SBATCH --mem 120G
#SBATCH --partition cpu
#SBATCH --time 12:50:00

#SBATCH --job-name=uni_umap
#SBATCH --output=/users/mensmeng/logs/%x-%A-%a.out
#SBATCH --error=/users/mensmeng/logs/%x-%A-%a.err

conda activate /work/PRTNR/CHUV/DIR/rgottar1/spatial/conda_envs/wsi_trident

PATH_TO_SCRIPT="/users/mensmeng/workspace/ai4bmr-datasets/scripts/BEAT/process_embeddings_descriptive.py"

python $PATH_TO_SCRIPT 
conda deactivate