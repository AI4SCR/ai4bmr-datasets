import argparse
import os
from pathlib import Path
import h5py
import geopandas as gpd

from trident import OpenSlideWSI
from trident.segmentation_models import segmentation_model_factory
from trident.patch_encoder_models import encoder_factory

def read_contour_file(wsi=None, wsi_name=None, job_dir=None, add_section=False):
    """
    Read the contour file from the job directory.
    Args:
        wsi (WSI object, optional): WSI object containing the name of the slide.
        wsi_name (str, optional): Name of the WSI slide.
        job_dir (str): Directory where the contour files are stored.
        add_section (bool): Whether to add section_id and slide_id to the GeoDataFrame.
    Returns:
        gdf (GeoDataFrame): GeoDataFrame containing the contours.
    """
    if wsi_name is None and wsi is None:
        raise ValueError("Either wsi_name or wsi must be provided.")
    if wsi is not None:
        wsi_name = wsi.name
    #check if file exists

    
    contour_file = os.path.join(job_dir, 'contours_geojson', f"{wsi_name}.geojson")
    if not os.path.exists(contour_file):
        raise FileNotFoundError(f"Contour file {contour_file} not found.")
    gdf = gpd.read_file(contour_file)

    ### add column for slide name and section id to the contour data
    if add_section:
        gdf['slide_id'] = wsi_name
        gdf['section_id'] = gdf['slide_id'] + '_' + gdf['tissue_id'].astype(str)
    
    return gdf

def patch_slide(wsi_name, wsi_path, seg_dir, patch_dir, target_mag=20, patch_size=256, overlap=0, min_tissue_proportion=0):

        ## initialize patching parameters
        ## read json file (optional)
        target_magnification = target_mag
        patch_size = patch_size
        overlap = overlap


        ### create patch dir and check if already exists
        
        ## check if exists and not empty
        if (Path(patch_dir) / 'patches' / f"{wsi_name}_patches.h5").exists() :
            print(f"Patch file for {wsi_name} already exists in {patch_dir}. Skipping patching.")
            return
        else:
            os.makedirs(patch_dir, exist_ok=True)



        ######################### ALL THIS SHOULD BE RUN AS A MULTIARRAY SLURM JOB #######################
        slide = OpenSlideWSI(slide_path=wsi_path, lazy_init=False)
        gdf = read_contour_file(wsi_name=wsi_name, job_dir=seg_dir)
        print('successfully read geojson file')

        wsi_to_process = slide
        wsi_to_process.gdf_contours = gdf
        wsi_to_process.name = wsi_name
        print(f"WSI name: {wsi_to_process.name}")

        coords_path = wsi_to_process.extract_tissue_coords(
        target_mag=target_magnification,
        patch_size=patch_size,
        save_coords=str(patch_dir),
        overlap=overlap,
        min_tissue_proportion=min_tissue_proportion,
        )
        print("Coordinates saved to:", coords_path)
        

        viz_coords_path = wsi_to_process.visualize_coords(
            coords_path=coords_path,
            save_patch_viz=os.path.join(patch_dir, 'visualization', wsi_name),
        )
        print(f"Tissue coordinates extracted and saved to {viz_coords_path}.")


        

    