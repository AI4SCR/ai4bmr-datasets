from pathlib import Path
#from ai4bmr_datasets.utils.download import download_file_map, unzip_recursive
from readimc import MCDFile, TXTFile

from ai4bmr_datasets.datasets.BaseIMCDataset import BaseIMCDataset
from ai4bmr_datasets.utils import io
from ai4bmr_datasets.utils.download import download_file_map, unzip_recursive

from pathlib import Path
import numpy as np

from loguru import logger


'''
def unzip_recursive(zip_path: Path, extract_dir: Path = None):
    """
    Recursively unzips a given zip file to a specified directory.

    If `extract_dir` is not provided, it defaults to a directory named after the zip file
    (without the .zip extension) in the same parent directory as the zip file.
    The function checks if files have already been extracted to avoid redundant operations.
    It also handles nested zip files by recursively calling itself on any .zip files found
    within the extracted content.

    Args:
        zip_path (Path): The path to the zip file to be unzipped.
        extract_dir (Path, optional): The directory where the contents should be extracted.
                                      Defaults to None.
    """
    import zipfile
    from loguru import logger

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        extracted_names = zip_ref.namelist()
        extract_dir = extract_dir or zip_path.parent / zip_path.stem
        is_extracted = all((extract_dir / name).exists() for name in extracted_names)
        if is_extracted:
            logger.info(f"Skipping extraction for {zip_path.name}, files already exist.")
        else:
            logger.info(f"Unzipping {zip_path.name}")

            # zip_ref.extractall(extract_dir)
            for zip_info in zip_ref.infolist():
                if zip_info.filename == '/':
                    continue
                elif zip_info.filename.strip():  # Skip empty or whitespace-only filenames
                    zip_ref.extract(zip_info, extract_dir)

        # Recursively unzip any .zip files in extracted content
        for name in extracted_names:
            extracted_path = extract_dir / name
            if extracted_path.suffix == ".zip":
                unzip_recursive(extracted_path)


def download_file_map(file_map: dict[str, Path], force: bool = False):
    """
    Downloads multiple files specified in a dictionary from URLs to target paths.

    This function iterates through the `file_map`, downloading each file.
    It includes a progress bar for each download and checks if a file already exists
    at the target path to skip re-downloading, unless `force` is set to True.

    Args:
        file_map (dict[str, Path]): A dictionary where keys are URLs (str) and values
                                    are the target file paths (Path) for the downloads.
        force (bool): If True, forces re-download of files even if they already exist.
    """
    from tqdm import tqdm
    import requests

    for url, target_path in file_map.items():
        import pdb; pdb.set_trace()
        if not target_path.exists() or force:
            try:
                logger.info(f"Downloading {url} → {target_path}")
                headers = {"User-Agent": "Mozilla/5.0"}
                with requests.get(url, headers=headers, stream=True) as r:
                    r.raise_for_status()
                    total_size = int(r.headers.get('Content-Length', 0))
                    chunk_size = 8192

                    with open(target_path, 'wb') as f, tqdm(
                            desc=target_path.name,
                            total=total_size,
                            unit='B',
                            unit_scale=True,
                            unit_divisor=1024,
                    ) as pbar:
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            if chunk:  # filter out keep-alive chunks
                                f.write(chunk)
                                pbar.update(len(chunk))
            except Exception as e:
                logger.error(f"Failed to download {url}. Network connection might have been interrupted. Error: {e}")
                if target_path.exists():
                    target_path.unlink()
        else:
            logger.info(f"Skipping download of {target_path.name}, already exists.")

'''

'''
import sys
import os

# Hardcoded repo root path
repo_root = '/users/tmaffei1/dataset_dev/ai4bmr-datasets/'

if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
    
from src.ai4bmr_datasets.utils.download import download_file_map
'''



class Meyer2025(BaseIMCDataset):

    def __init__(self,
                base_dir: Path | None = None,
                image_version: str | None = None,
                mask_version: str | None = None,
                feature_version: str | None = None,
                metadata_version: str | None = None,
                load_intensity: bool = False,
                load_spatial: bool = False,
                load_metadata: bool = False,
                align: bool = False,
                join: str = "outer",
                ):

        super().__init__(base_dir=base_dir)
                         #image_version=image_version,
                         #mask_version=mask_version,
                         #feature_version=feature_version,
                         #metadata_version=metadata_version,
                         #load_intensity=load_intensity,
                         #load_spatial=load_spatial,
                         #load_metadata=load_metadata,
                         #align=align,
                         #join=join)

        self.raw_acquisitions_dir = self.raw_dir / 'acquisitions/'
        

    def prepare_data(self):
        print('prepare data')

        self.download()

        self.process_acquisitions()

        import pdb; pdb.set_trace()


    def download(self, force: bool = False):

        download_dir = self.raw_dir / 'acquisitions/'
        download_dir.mkdir(parents=True, exist_ok=True)

        file_map = {
            'https://zenodo.org/records/10942609/files/ZTMA174_raw.zip?download=1': download_dir / 'ZTMA174_raw.zip',
            'https://zenodo.org/records/10942609/files/ZTMA249_raw.zip?download=1': download_dir / 'ZTMA249_raw.zip',
        }

        download_file_map(file_map, force=force)

        # Extract zip files
        for target_path in file_map.values():
            try:
                unzip_recursive(target_path)
            except Exception as e:
                logger.error(f'Failed to unzip {target_path}: {e}')

        import pdb; pdb.set_trace()

        print('download')


    def process_acquisitions(self):

        print('acq in process ')

        acquisitions_dir = self.raw_acquisitions_dir 

        channel_names_labels = {}

        for acquisitions_folder_i in acquisitions_dir.iterdir():
            print('acquisitions_folder_i ', acquisitions_folder_i)

            if acquisitions_folder_i.suffix == '.zip':
                continue

            for acquisitions_folder_i_j in acquisitions_folder_i.iterdir():
                print('-------- acquisitions_folder_i_j ', acquisitions_folder_i_j)

                from tqdm import tqdm

                for acquisition_file in tqdm(acquisitions_folder_i_j.iterdir(), desc="Processing acquisitions"):
                    print('---------------- acquisition_file ', acquisition_file)
                    if acquisition_file.suffix.lower() == '.txt':

                        with TXTFile(acquisition_file) as f:
                            #print(f.channel_names)  # metals
                            #print(f.channel_labels)  # targets

                            channel_names_labels[acquisition_file.stem] = {
                                'channel_names': f.channel_names,
                                'channel_labels': f.channel_labels
                            }

                            img_path = self.raw_dir / 'images/raw/'
                            img_name = acquisition_file.stem + '.tiff'


                            img = f.read_acquisition()
                            img = img.astype(np.float32)
                            io.imsave(save_path=img_path / img_name, img=img)


        # check if images have different channel_names and channel_labels
        first_key = next(iter(channel_names_labels))
        ref_names = channel_names_labels[first_key]['channel_names']
        ref_labels = channel_names_labels[first_key]['channel_labels']

        for key, values in channel_names_labels.items():
            assert values['channel_names'] == ref_names, f"❌ channel_names mismatch found in: {key}"
            assert values['channel_labels'] == ref_labels, f"❌ channel_labels mismatch found in: {key}"

            if values['channel_names'] != ref_names or values['channel_labels'] != ref_labels:
                print(f"❌ Mismatch found in: {key}")

        # generate panel , selecting a random sample (give all channels are the same)
        import random
        import pandas as pd
        random_key = random.choice(list(channel_names_labels.keys()))
        panel_df = pd.DataFrame(channel_names_labels[random_key])
        panel_df.to_parquet(img_path / 'panel.csv')

        import pdb; pdb.set_trace()



dataset = Meyer2025(base_dir=Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/ai4bmr-datasets/Meyer2025/'))

dataset.prepare_data()




    