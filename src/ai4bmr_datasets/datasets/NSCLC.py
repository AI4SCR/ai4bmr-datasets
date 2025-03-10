import os 
from imageio import imread, imsave
from pathlib import Path
import re
import anndata
import pickle
import psutil
import wandb
import shutil
import numpy as np
import pandas as pd



class NSCLC():

    def __init__(self, 
                 NSCLS_path,
                 store_samples_adata=False):
        
        print('hello')
        self.TNBC_path = Path(NSCLS_path)
        self.store_samples_adata = store_samples_adata

        self.masks_path = self.TNBC_path / '02_processed/masks/'
        self.adatas_path = self.TNBC_path / '02_processed/adata_samples/'

        self.all_cells_adata_path = self.TNBC_path / '02_processed/sce_objects/sce.h5ad'
        self.save_single_adata_path = self.TNBC_path / '02_processed/adata_samples/'




    def load(self) : 

        masks, anndatas = self.return_paths()

        panel = pd.read_csv('/work/FAC/FBM/DBC/mrapsoma/prometex/data/NSCLC/01_raw/cp_csv/panel.csv')

        return dict(adata = anndatas,
                    masks = masks,
                    panel = panel)



    def return_paths(self):
        """
        return masks, anndatas paths
        """

        print(self.masks_path)

        masks_path_directories = os.listdir(self.masks_path)

        masks = {}
        anndatas = {}

        for mask_id in masks_path_directories : 
            print(mask_id)

            masks[mask_id[:-5]] = self.masks_path / mask_id
            anndatas[mask_id[:-5]] = self.adatas_path / f"{mask_id[:-5]}.h5ad"

            #import pdb; pdb.set_trace()

            #masks_folder_path = self.masks_path / mask_id

        return masks, anndatas




NSCLS_path = '/work/FAC/FBM/DBC/mrapsoma/prometex/data/NSCLC/'

NSCLC_data = NSCLC(NSCLS_path = NSCLS_path,
                   store_samples_adata=True)

NSCLC_preprocessed = NSCLC_data.load()

import pdb; pdb.set_trace()



