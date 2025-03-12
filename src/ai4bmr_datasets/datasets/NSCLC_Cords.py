from .BaseIMCDataset import BaseIMCDataset
import pandas as pd
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS

import tifffile
from tifffile import imwrite
from pathlib import Path
import numpy as np
import pandas as pd

from readimc import MCDFile, TXTFile

from tempfile import TemporaryDirectory
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union

import re
import os
from os import PathLike


class NSCLC(BaseIMCDataset):


    def __init__(self, base_dir: Path):


        super().__init__(base_dir)

        # TO DO get logger 
        #self.logger = get_logger('TNBCv2', verbose=1)
        self.version_name = 'published'

        self.raw_masks_dir = self.raw_dir / 'Cell_masks'
        self.raw_images_mcd = self.raw_dir / 'raw/mcd'
        self.raw_panel = self.raw_dir / 'raw/raw_panel.csv'
        self.tiff_imgs = self.processed_dir / 'tiff_imgs'


        self.supplementary_table_1_path = self.raw_dir / '1-s2.0-S0092867418311000-mmc1.xlsx'  # downloaded as Table S1 from paper
        self.raw_sca_path = self.raw_dir / 'TNBC_shareCellData' / 'cellData.csv'
        self.patient_class_path = self.raw_dir / 'TNBC_shareCellData' / 'patient_class.csv'



    def create_images(self):

        mcd_folders = os.listdir(self.raw_images_mcd)
        
        for mcd_folder in mcd_folders : 
            print('mcd_folder ', mcd_folder)
            folder_path = os.path.join(self.raw_images_mcd, mcd_folder)

            files_in_folder = os.listdir(folder_path)

            for file in files_in_folder:
                print('file ', file)

                mcd_file_path = self.raw_images_mcd / mcd_folder / file

                print('MCD FILE PATH ', mcd_file_path)

                self.extract_tiff_metadata(mcd_file_path, self.tiff_imgs, self.raw_panel)

        import pdb; pdb.set_trace()



    @staticmethod
    def get_tiff_metadata(path):
        import tifffile
        tag_name = 'PageName'
        metadata = pd.DataFrame()
        metadata.index.name = 'page'
        with tifffile.TiffFile(path) as tif:
            for page_num, page in enumerate(tif.pages):
                page_name_tag = page.tags.get(tag_name)
                import pdb; pdb.set_trace()
                metadata.loc[page_num, tag_name] = page_name_tag.value
        metadata = metadata.assign(PageName=metadata.PageName.str.strip())
        metadata = metadata.PageName.str.extract(r'(?P<target>\w+)\s+\((?P<mass_channel>\d+)')

        metadata.mass_channel = metadata.mass_channel.astype('int')
        metadata = metadata.convert_dtypes()

        return metadata
    

    def extract_tiff_metadata(self, mcd_file, output_dir, raw_panel_path):

        mcd_file = Path(mcd_file)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not mcd_file.is_file():
            raise FileNotFoundError(f"MCD file not found: {mcd_file}")
        
        tiff_files = []
        
        # Open the MCD file
        with MCDFile(mcd_file) as mcd:
            num_tiff_images = sum(len(slide.acquisitions) for slide in mcd.slides)
            
            if num_tiff_images == 0:
                raise ValueError("No TIFF images found in the MCD file.")
            
            # Extract number of channels from the first acquisition (assuming all are similar)
            first_acq = mcd.slides[0].acquisitions[0]
            channels_per_image = len(first_acq.channel_names)
            channel_names = first_acq.channel_names
            
            # Process each acquisition (similar to the try_preprocess_images_from_disk method)
            for slide_idx, slide in enumerate(mcd.slides):

                for acq_idx, acquisition in enumerate(slide.acquisitions):
                    print('acq_idx ', acq_idx)  

                    raw_panel = pd.read_csv(raw_panel_path)

                    # channels to keep 
                    panel_channels_to_keep = raw_panel[raw_panel['full'] == 1]

                    # Read the acquisition data, which includes all channels
                    img_all_channels = mcd.read_acquisition(acquisition, strict=False)

                    # mask acquisition layers and channels to keep , MASK RELATED TO IMAGE (mcd.read_acquisition(acquisition, strict=False)) ORDER 
                    mask = [1 if name in panel_channels_to_keep['Metal Tag'].values else 0 for name in acquisition.channel_names]

                    # verify the kept channels are 43, accordin to the proteins in the panel
                    assert np.sum(mask) == 43 and np.sum(mask) == len(panel_channels_to_keep), 'number of channels to keep is not 43'

                    # filtered image with the 43 relevant protein channels 
                    img_filtered_channels = img_all_channels[np.array(mask, dtype=bool), :, :]

                    # mapping channels to protein 
                    filtered_channel_protein_metal_tag = [element for element, m in zip(acquisition.channel_names, mask) if m == 1]

                    metal_to_target_mapping = dict(zip(raw_panel['Metal Tag'], raw_panel['Target']))
                    mapped_targets = {metal: metal_to_target_mapping.get(metal, "No Target Found") for metal in filtered_channel_protein_metal_tag}
                    mapped_df = pd.DataFrame(list(mapped_targets.items()), columns=['Metal Tag', 'Target'])

                    # metedata on protein name preparation
                    metadata = {f"Layer {i+1}": {"Target": mapped_df['Target'][i]} for i in range(len(mapped_df))}
                    # Convert metadata to a string format compatible with TIFF files
                    metadata_str = "\n".join([f"Layer {i+1}: {mapped_df['Target'][i]}" for i in range(len(mapped_df))])

                    pattern = r'(\d+[A-C]?)\b'
                    pattern = r'(\d+_[A-Za-z])'
                    pattern = r'_(\d+_[A-Z])\.mcd'

                    match = re.search(pattern, str(mcd_file)[-20:])

                    #import pdb; pdb.set_trace()
                    #match = re.search(pattern, str(mcd_file))
                    # TO DO use re 
                    TMA_cell_pattern = match.group(1)

                    # save ndarray as a multi-page TIFF file with metadata
                    imwrite(output_dir / f'{TMA_cell_pattern}_{acq_idx}.tiff', img_filtered_channels.astype(np.float32), metadata={'ImageDescription': metadata_str})
        




    


NSCLS_path = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/NSCLC/')
NSCLC_dataset = NSCLC(base_dir=NSCLS_path)

NSCLC_dataset.create_images()
    
img_path = '/work/FAC/FBM/DBC/mrapsoma/prometex/data/NSCLC/01_raw/raw/img/20210129_LC_NSCLC_TMA_88_A_073.tiff'
NSCLC_dataset.get_tiff_metadata(img_path)





