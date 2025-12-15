#!/usr/bin/env bash

# NOTE: requires at least 128GB of RAM

export BATCH_SIZE=1
export NUM_WORKERS=1
export TOOL=/work/FAC/FBM/DBC/mrapsoma/prometex/projects/trident/run_batch_of_slides.py
export WSI_DIR=/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/beat/02_processed/datasets/beat-trident/wsi_slides
export JOB_DIR=/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/beat/02_processed/datasets/beat-trident/wsi_embed
python $TOOL --task seg --wsi_dir $WSI_DIR --job_dir $JOB_DIR --gpu 0 --segmenter hest --max_workers $NUM_WORKERS --batch_size $BATCH_SIZE --seg_batch_size $BATCH_SIZE --feat_batch_size $BATCH_SIZE
python $TOOL --task coords --wsi_dir $WSI_DIR --job_dir $JOB_DIR --mag 20 --patch_size 512 --overlap 0 --max_workers $NUM_WORKERS --batch_size $BATCH_SIZE --seg_batch_size $BATCH_SIZE --feat_batch_size $BATCH_SIZE
# python $TOOL --task feat --wsi_dir $WSI_DIR --job_dir $JOB_DIR --patch_encoder uni_v1 --mag 20 --patch_size 512 --max_workers $NUM_WORKERS --batch_size $BATCH_SIZE --seg_batch_size $BATCH_SIZE --feat_batch_size $BATCH_SIZE

# NOTE: this errors without of memory. Probably we need to limit the number of patches but the tool does not support this from the cmd line
python $TOOL --task feat --wsi_dir $WSI_DIR --job_dir $JOB_DIR --slide_encoder titan --mag 20 --patch_size 512 --max_workers $NUM_WORKERS --batch_size $BATCH_SIZE --seg_batch_size $BATCH_SIZE --feat_batch_size $BATCH_SIZE