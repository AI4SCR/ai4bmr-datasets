import os 
import re
from imageio import imread, imsave
from pathlib import Path
import anndata
import pickle
import psutil
import wandb
import shutil
import numpy as np

import pandas as pd
import skimage
from pathlib import Path

import athena as ath
import networkx as nx

import psutil
import gc



class NSCLC():

    def __init__(self, 
                 NSCLS_path,
                 store_samples_adata=False,
                 store_graphs_adata=False):
        
        print('hello')
        self.TNBC_path = Path(NSCLS_path)
        self.store_samples_adata = store_samples_adata
        self.store_graphs_adata = store_graphs_adata

        self.masks_path = self.TNBC_path / '02_processed/Cell_masks_copy/'
        self.all_cells_adata_path = self.TNBC_path / '02_processed/sce_objects/sce.h5ad'
        self.renamed_masks_path = self.TNBC_path / '02_processed/masks_all_2/'
        self.save_single_adata_path = self.TNBC_path / '02_processed/adata_samples_all/'
        self.path_to_store_adata_graphs =  self.TNBC_path / '02_processed/export/anndata_graphs_86/'


        if self.store_samples_adata : 
            self.preprocess()

        # select IDs where there are images
        pattern = re.compile(r"TMA_(\d+_[A-Z]_\d+)")
        ids = [match.group(1) for filename in os.listdir('/work/FAC/FBM/DBC/mrapsoma/prometex/data/NSCLC/01_raw/raw/img') if (match := pattern.search(filename))]
        ids = [re.sub(r'(\d+_[A-Z]_)(0+)(\d+)', r'\1\3', item) for item in ids]

        # remove IDs that ahve errors
        ids = [item for item in ids if item != '86_B_24']




        #with open("ids.txt", "w") as f:
            #for id in ids:
                #f.write(id + "\n")

        #import pdb; pdb.set_trace()



        if self.store_graphs_adata : 

            #for adata_path in self.save_single_adata_path.iterdir():

                #adata_id = adata_path.stem
                #print(adata_id)

            for adata_id in ids : 
                print('ADATA ID ', adata_id)

                self.create_graph_adata(adata_id, self.path_to_store_adata_graphs)


            import pdb; pdb.set_trace()


        if self.store_graphs_adata : 
            for ids, adata_id in enumerate(self.NSCLC_preprocessed['adata'].keys()) :

                self.create_graph_adata(adata_id, self.path_to_store_adata_graphs)
                mem_after = psutil.Process().memory_info().rss / 1024 ** 2  
                print(f"Memory usage {mem_after:.2f} MB, ")

        #wandb.init(
            #project='NSCLC',
            #name = 'run',
            #config={
            #}
        #)


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



    def preprocess(self) : 

        all_masks_paths, all_sample_identifiers, all_masks_objects = self.return_masks_paths()

        import pdb; pdb.set_trace()

        if self.store_samples_adata :
            all_cells_adata_objects = self.store_all_adata_samples(all_sample_identifiers)


        # verify that mask and anndata contain the same objects IDs
        # 0 is discarded from elements in the mask since it is the background
        for i in range(len(all_masks_objects)):

            # maybe just need to set that it is not equal instead of equal 

            assert np.array_equal(
                all_masks_objects[i][all_masks_objects[i] != 0], 
                np.sort(all_cells_adata_objects[i])
            ), f"Mismatch found at index {all_sample_identifiers[i]}"


        import pdb; pdb.set_trace()


    def return_masks_paths(self):
        """
        return all the masks available in the raw data, identifiers for each sample, all unique objects in the masks
        """

        print(self.masks_path)

        masks_path_directories = os.listdir(self.masks_path)

        print(self.masks_path)

        all_masks_paths = []
        all_sample_identifiers = []
        all_masks_objects = []

        tot_nb_raw_masks = 0
        tot_nb_renamed_masks = 0

        for masks_path_directory in masks_path_directories : 
            print(masks_path_directory)

            masks_folder_path = self.masks_path / masks_path_directory

            for mask in os.listdir(masks_folder_path): 

                mask_path = self.masks_path / masks_path_directory / mask
                all_masks_paths.append(mask_path)

                print(mask)

                pattern = r"TMA_(\d+)_([A-Za-z])_s0_a(\d+)_ac"
                mask_identifiers = {}

                match = re.search(pattern, mask)
                if match:
                    TmaID = match.group(1)
                    TmaBlock = match.group(2)
                    acID = match.group(3)
                    print(f"TmaID: {TmaID}, TmaBlock: {TmaBlock}, acID: {acID}")

                    mask_identifiers['TmaID'] = TmaID
                    mask_identifiers['TmaBlock'] = TmaBlock
                    mask_identifiers['acID'] = acID
                    all_sample_identifiers.append(mask_identifiers)

                    assert TmaID, "TmaID is empty or None"
                    assert TmaBlock, "TmaBlock is empty or None"
                    assert acID, "acID is empty or None"

                # rename the mask and store it in the processed data folder 
                # no preprocessing on the amsk is actually performed compared to the raw data from the publication
                new_dir = Path(self.renamed_masks_path)
                new_filename = f"{mask_identifiers['TmaID']}_{mask_identifiers['TmaBlock']}_{mask_identifiers['acID']}.tiff"

                new_file_path = mask_path.parent / new_filename

                mask_path.rename(new_file_path)
                
                #import pdb; pdb.set_trace()

                # make a copy of the mask and rename it, copy2 copy the files with metadata
                #new_path = new_dir / new_filename
                #shutil.copy2(str(mask_path), str(new_path))

                # store all the objects IDs in the mask
                #all_masks_objects.append(np.unique(imread(mask_path)))

                # total number
                
            #tot_nb_raw_masks += len(list((self.masks_path / masks_path_directory).glob('*')))
            #tot_nb_renamed_masks += len(list((new_dir).glob('*')))

            #import pdb; pdb.set_trace()

        return all_masks_paths, all_sample_identifiers, all_masks_objects
    

    def store_all_adata_samples(self, all_sample_identifiers) :
        """
        starting from sample identifiers, store anndata specific to each sample

        
        Parameters
        ----------
        all_sample_identifiers : dict with keys ['TmaID'] ['TmaBlock'] ['acID']

        """

        all_cells_adata = anndata.read_h5ad(self.all_cells_adata_path)
        all_cells_adata_objects = []
        
        for id, sample_identifier in enumerate(all_sample_identifiers) :
            print(sample_identifier)
            adata_sample, adata_objects = self.return_single_adata_sample(all_cells_adata, sample_identifier['TmaID'], sample_identifier['TmaBlock'], sample_identifier['acID'])
            all_cells_adata_objects.append(adata_objects)

            save_path = self.save_single_adata_path / f"{sample_identifier['TmaID']}_{sample_identifier['TmaBlock']}_{sample_identifier['acID']}.h5ad"

            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()

            print(f"Memory Usage: {mem_info.rss / (1024 * 1024):.2f} MB") 

            # instead of saving the anndata as a pickle, save it using anndata write 
            adata_sample.write(save_path)

            #with open(save_path, 'wb') as f : 
                #pickle.dump(adata_sample, f)
            
            #wandb.log({'sample processed':  f"{sample_identifier['TmaID']}_{sample_identifier['TmaBlock']}_{sample_identifier['acID']}.pkl"})
            #wandb.log({'sample id': id})

        
        print('THE END ')
        #import pdb; pdb.set_trace()

        return all_cells_adata_objects
        

    

    def return_single_adata_sample(self, all_cells_adata, TmaID, TmaBlock, acID):
        """
        starting from single cell experiment anndata (containing all experiments and all cells)
        return anndata associated to specific TmaID, TmaBlock, acID 

        Parameters
        ----------
        all_cells_adata : str
        TmaID : str
        TmaBlock : str
        acID : int

        returns anndata 
        """

        adata_sample = all_cells_adata[(
            (all_cells_adata.obs['TmaID'] == str(TmaID)) & 
            (all_cells_adata.obs['TmaBlock'] == str(TmaBlock)) & 
            (all_cells_adata.obs['acID'] == int(acID))
        ), :]

        adata_objects = adata_sample.obs['CellNumber'].values

        return adata_sample, adata_objects
    



    def create_graph_adata(self, anndata_id, path_to_store) : 
        print('hei')

        ad = anndata.read_h5ad(self.save_single_adata_path / f'{anndata_id}.h5ad')
        mask_path = self.renamed_masks_path / f'{anndata_id}.tiff'

        # load anndata of the NSCLS dataset
        #with open(self.NSCLC_preprocessed['adata'][str(anndata_id)], "rb") as f: 
            #ad = pickle.load(f) 

        # generated cell centroids that needs to be sotred in the obsm of the anndata
        props = skimage.measure.regionprops_table(imread(mask_path), properties=['label', 'centroid'])
        cell_centroids = pd.DataFrame(props)
        cell_centroids.rename(columns={'label': 'object_id'}, inplace=True)
        cell_centroids.rename(columns={'centroid-0' : 'x', 'centroid-1' : 'y'}, inplace=True)
        cell_centroids.set_index('object_id', inplace=True)
        obsm = cell_centroids

        # check that obs and obsm have the same size
        assert len(ad.obs) == len(obsm), "length of anndata obs does not match length of annadata obsm"

        # need to reset the index of the anndata.obs to be aligned with the athena library (need to have as index of the obs the id of the cells as an int, no characters in it)
        ad.obs = ad.obs.set_index("CellNumber", drop=False) 
        ad.obs.index.name = "object_id" 

        # store obsm into the anndata object 
        ad.obsm = {'centroids': obsm.values}

        # BUILD GRAPH 
        # load the mask of the specific anndata
        mask = imread(mask_path)
        # check that elements in the mask matches the elements in the anndata 
        assert np.array_equal(np.unique(mask)[np.unique(mask) != 0], np.sort(ad.obs['CellNumber'].values)), "content of mask and anndata not equal !"

        ath.graph.build_graph(ad, topology='radius', graph_key='radius_32', radius=32, include_self=True, mask = mask) 

        # store anndata
        path_to_store = Path(path_to_store) / f"{anndata_id}.pkl"
        with open(path_to_store, "wb") as f:
            pickle.dump(ad, f)

        #import pdb; pdb.set_trace()

        del ad  # Remove Anndata object
        del mask  # Remove image mask if not needed
        gc.collect()  # Force garbage collection


                    



NSCLS_path = '/work/FAC/FBM/DBC/mrapsoma/prometex/data/NSCLC/'

NSCLC_data = NSCLC(NSCLS_path = NSCLS_path,
                   store_samples_adata=True,
                   store_graphs_adata=False)

#NSCLC_data.preprocess()







