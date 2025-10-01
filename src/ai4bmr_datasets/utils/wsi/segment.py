#%%
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd


# from dotenv import load_dotenv
# load_dotenv('/users/mensmeng/workspace/dermatology/derm-pipeline/.env')

from trident import OpenSlideWSI
from trident.segmentation_models import segmentation_model_factory
from trident.patch_encoder_models import encoder_factory

#%%

def segment_wsi(wsi_path, wsi_name, seg_model = None, seg_dir=None, force_rerun=False):



    if seg_dir.exists():
        if (Path(seg_dir) / 'contours_geojson' / f"{wsi_name}.geojson").exists():
            print(f"Segmentation has already been done for {wsi_path}.")
            if force_rerun:
                print("Force rerun is enabled. Rerunning segmentation...")
            else:
                return
       
    ######################### ALL THIS SHOULD BE RUN AS A MULTIARRAY SLURM JOB #######################
    import torch
    DEVICE = f"cuda:0" if torch.cuda.is_available() else "cpu"
        
    model_name = seg_model
    segmentation_model = segmentation_model_factory(model_name)

    wsi_path = Path(wsi_path)
    assert wsi_path.exists(), f"WSI path {wsi_path} does not exist."
    

    # Load WSI
    slide = OpenSlideWSI(slide_path=wsi_path, lazy_init=False)
    wsi_name = wsi_name or wsi_path.stem
    slide.name = wsi_name
    print(slide)
    gdf = None
    # Run segmentation
    print(f"Running segmentation for slide {wsi_name}...")
    
    print(segmentation_model)
    geojson_contours = slide.segment_tissue(segmentation_model=segmentation_model, target_mag=10, job_dir=str(seg_dir),
                                            device=DEVICE)
    print(geojson_contours)
    print(f'successfully segmented WSI {wsi_name}')

def filter_contours():
    pass