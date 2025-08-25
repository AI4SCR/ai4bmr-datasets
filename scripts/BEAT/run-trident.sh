#!/usr/bin/env bash

export BATCH_SIZE=4
export NUM_WORKERS=16
export TOOL=/work/FAC/FBM/DBC/mrapsoma/prometex/projects/trident/run_batch_of_slides.py
export WSI_DIR=/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/beat/02_processed_v2/trident/datasets/beat/wsi_slides
export JOB_DIR=/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/beat/02_processed_v2/trident/datasets/beat/wsi_embed
python $TOOL --task seg --wsi_dir $WSI_DIR --job_dir $JOB_DIR --gpu 0 --segmenter hest --max_workers $NUM_WORKERS --batch_size $BATCH_SIZE
python $TOOL --task coords --wsi_dir $WSI_DIR --job_dir $JOB_DIR --mag 20 --patch_size 512 --overlap 0 --max_workers $NUM_WORKERS --batch_size $BATCH_SIZE
# python $TOOL --task feat --wsi_dir $WSI_DIR --job_dir $JOB_DIR --patch_encoder uni_v1 --mag 20 --patch_size 512 --max_workers $NUM_WORKERS --batch_size $BATCH_SIZE
python $TOOL --task feat --wsi_dir $WSI_DIR --job_dir $JOB_DIR --slide_encoder titan --mag 20 --patch_size 512 --max_workers $NUM_WORKERS --batch_size $BATCH_SIZE