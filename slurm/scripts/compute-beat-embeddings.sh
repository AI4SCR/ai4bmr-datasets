#!/bin/bash

#SBATCH -c 8
#SBATCH --mem 50G
#SBATCH --partition gpu
#SBATCH --time 00:30:00
#SBATCH --gres gpu:1

#SBATCH --job-name=rerun_beat_embeddings
#SBATCH --output=/users/mensmeng/logs/%x-%A-%a.out
#SBATCH --error=/users/mensmeng/logs/%x-%A-%a.err
#SBATCH --array=17-17


SAMPLE_LIST="/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/beat_rescanned/02_processed/sample_ids.txt"
SAMPLE_ID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $SAMPLE_LIST)
echo "Processing sample ID: $SAMPLE_ID"


conda activate wsi_trident

PATH_TO_SCRIPT="/users/mensmeng/workspace/ai4bmr-datasets/slurm/scripts/compute-beat-embeddings.py"

python $PATH_TO_SCRIPT --sample_id $SAMPLE_ID

echo "Finished processing sample ID: $SAMPLE_ID"
conda deactivate