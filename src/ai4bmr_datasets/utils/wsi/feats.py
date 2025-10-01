#%%
import json
import subprocess
import textwrap
from pathlib import Path

# from dotenv import load_dotenv
# load_dotenv('/users/mensmeng/workspace/dermatology/derm-pipeline/.env')


import geopandas as gpd
import numpy as np
import openslide
import pandas as pd
#import tifffile
import torch
from tqdm import tqdm



from openslide import OpenSlide
import os

#%%
from trident import OpenSlideWSI
from trident.segmentation_models.load import segmentation_model_factory
from trident.IO import read_coords
from trident.patch_encoder_models.load import encoder_factory
from trident.slide_encoder_models.load import encoder_factory as slide_encoder_factory
import h5py
from dotenv import load_dotenv, dotenv_values
from huggingface_hub import login
from trident.slide_encoder_models.load import slide_to_patch_encoder_name
from trident.IO import create_lock, remove_lock, is_locked, update_log


#%%
def check_fm_compatibility(target_mag, patch_size, feature_extractor, slide_extractor):

    avail_feature_extractors = ['uni_v2', 'hoptimus0', 'virchow2', 'hibou_l', 'conch_v15', 'resnet50']
    avail_slide_encoders = ['titan']


    slide_to_patch_encoder_name = {
    'threads': 'conch_v15',
    'titan': 'conch_v15',
    'tcga': 'conch_v15',
    'prism': 'virchow',
    'chief': 'ctranspath',
    'gigapath': 'gigapath',
    'madeleine': 'conch_v1',
    'feather': 'conch_v15'
    }

    params_feature_extracor = {
        'uni_v2': {'mag': 20, 'patch_size': 256},
        'hoptimus0': {'mag': 20, 'patch_size': 224},
        'virchow2': {'mag': 20, 'patch_size': 224},
        'hibou_l': {'mag': 20, 'patch_size': 224},
        'conch_v15': {'mag': 20, 'patch_size': 512},
        'resnet50': {'mag': 20, 'patch_size': 256}
    }

    
    if slide_extractor is not None:
        if slide_extractor not in avail_slide_encoders:
            raise ValueError(f"Slide encoder '{slide_extractor}' is not available. Please choose from {avail_slide_encoders}.")
        
    
        if feature_extractor is None:
            feature_extractor = slide_to_patch_encoder_name.get(slide_extractor)
        else:
            expected_extractor = slide_to_patch_encoder_name.get(slide_extractor)
            if feature_extractor != expected_extractor:
                raise ValueError(f"Feature extractor '{feature_extractor}' is not compatible with slide encoder '{slide_extractor}'. Please choose '{expected_extractor}' instead.")

    if feature_extractor is not None:
        if feature_extractor not in avail_feature_extractors:
            raise ValueError(f"Feature extractor '{feature_extractor}' is not available. Please choose from {avail_feature_extractors}.")

        expected_mag = params_feature_extracor.get(feature_extractor, {}).get('mag')
        expected_patch_size = params_feature_extracor.get(feature_extractor, {}).get('patch_size')

        if expected_mag is not None and target_mag != expected_mag:
            raise ValueError(f"Target magnification '{target_mag}' is not compatible with feature extractor '{feature_extractor}'. Please choose '{expected_mag}' instead.")

        if expected_patch_size is not None and patch_size != expected_patch_size:
            raise ValueError(f"Patch size '{patch_size}' is not compatible with feature extractor '{feature_extractor}'. Please choose '{expected_patch_size}' instead.")


def extract_patch_features(wsi_path, wsi_name, patch_dir, feature_dir, feature_extractor, target_mag=20, patch_size=256, overlap=0):

        check_fm_compatibility(target_mag=target_mag, patch_size=patch_size, feature_extractor=feature_extractor, slide_extractor=None)
        
        ## check if exists and not empty
        if not (Path(patch_dir) / 'patches' / f"{wsi_name}_patches.h5").exists() :
            print(f"Patch file for {wsi_name} does not exist in {patch_dir}. Please run patch_slide() first.")
            return

        if (Path(feature_dir) / f"{wsi_name}.h5").exists():
            print(f"Feature file for {wsi_name} already exists in {feature_dir}. Skipping feature extraction.")
            return
        else:
            Path(feature_dir).mkdir(parents=True, exist_ok=True)

        #SET PARAMETERS
        DEVICE = f"cuda:0" if torch.cuda.is_available() else "cpu"
        PATCH_ENCODER = feature_extractor
        print(f"Using patch encoder: {PATCH_ENCODER}")

        
        coords_dir = str(patch_dir)
        print(f"Coordinates directory: {coords_dir}")


        ######## ALL FROM HERE SHOULD BE RUN AS A MULTIARRAY SLURM JOB #######################

        ## initalize encoder
        encoder = encoder_factory(PATCH_ENCODER)
        encoder.eval()
        encoder.to(DEVICE)
        
        #file_path = "/users/mensmeng/workspace/dermatology/TRIDENT_pipeline/processed_slides/20x_256px_0px_overlap/patches/D2101790_patches.h5"
        ## function that takes coords dir and slide name and returns the different section names and its file paths

     
        coords_path = os.path.join(coords_dir, 'patches', f"{wsi_name}_patches.h5")
        print(f"Processing coordinates file: {coords_path}")
        # make sure the file exists
        if not os.path.exists(coords_path):
            print(f"File {coords_path} does not exist.")
            return
                
        coords_attrs, coords = read_coords(coords_path)

        slide_id = coords_attrs["name"]
        print(f"Slide ID: {slide_id}")
        assert (slide_id == wsi_name), 'Slide ID does not match the WSI name'

        print('########################\n')
        print(f'Extracting features for {slide_id}\n')
        print('########################')
        slide = OpenSlideWSI(slide_path=wsi_path, lazy_init=False)
        wsi_to_process = slide
        wsi_to_process.name = wsi_name
        feats_path = wsi_to_process.extract_patch_features(
            patch_encoder=encoder,
            coords_path=coords_path,
            save_features=str(feature_dir),
            device=DEVICE, 
            batch_limit=512
        )

        print(feats_path)

def extract_slide_features(wsi_path, wsi_name, feature_dir, slide_feature_dir, slide_extractor, feature_extractor,
                           target_mag=20, patch_size=256, overlap=0):

    check_fm_compatibility(target_mag=target_mag, patch_size=patch_size, feature_extractor=feature_extractor, slide_extractor=slide_extractor)

    if not (Path(feature_dir) / f"{wsi_name}.h5").exists():
        print(f"Patch feature file for {wsi_name} does not exist in {feature_dir}. Please run extract_patch_features() first.")
        return
    if (Path(slide_feature_dir) / f"{wsi_name}.h5").exists():
        print(f"Slide features already extracted to: {slide_feature_dir}")
        return
    else:
        Path(slide_feature_dir).mkdir(parents=True, exist_ok=True)
        print(f"Slide features will be saved to: {slide_feature_dir}")


    slide_encoder = slide_encoder_factory(slide_extractor)
    patch_encoder = slide_to_patch_encoder_name[slide_encoder.enc_name]
    assert patch_encoder == feature_extractor, f"Patch encoder {feature_extractor} does not match the expected patch encoder {patch_encoder} for slide encoder {slide_encoder.enc_name}. Please extract patch features using {patch_encoder} first."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    slide_extractor.to(device)

    patch_features_path = os.path.join(feature_dir, f"{wsi_name}.h5")
    print(f"Reading patch features from: {patch_features_path}")

    

    ##check if the WSI file exists
    assert os.path.exists(wsi_path), f"WSI file {wsi_path} does not exist."

    wsi = OpenSlideWSI(slide_path=wsi_path, lazy_init=False)
    wsi.name = wsi_name

    
        
    # Call the extract_slide_features method
    wsi.extract_slide_features(
        patch_features_path=patch_features_path,
        slide_encoder=slide_encoder,
        device=device,
        save_features=os.path.join(slide_feature_dir)
    )

    print(f"Slide features saved to: {slide_feature_dir}")

# %%
