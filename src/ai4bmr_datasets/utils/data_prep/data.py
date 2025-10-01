#%%
import os
from pathlib import Path
import pandas as pd
import numpy as np
import h5py
import json


#%%
def h5_to_pandas(file_path, key='coords'):
    with h5py.File(file_path, "r") as f:
        data = f[key][:]
        if key == 'coords':
            columns = ['x', 'y']
        else:
            columns = None
        df = pd.DataFrame(data, columns=columns)
        for attr_key, attr_value in f[key].attrs.items():
            df[attr_key] = attr_value
    return df

def coords_to_df(coords_dir, ext = ".h5"):


    coords_dir = Path(coords_dir)

    ### sort the filepaths
    file_paths = [f for f in coords_dir.glob("*" + ext) if f.is_file()]
    file_paths = sorted(file_paths)
    
    dataframes = []
    for file_path in file_paths:
        df = h5_to_pandas(file_path, key='coords')
        df['patch_id'] = df.index
            # Save or process the DataFrame as needed
        dataframes.append(df)


    df_coords = pd.concat(dataframes, ignore_index=True)
    df_coords.rename(columns={'name': 'block_id'}, inplace=True)
    df_coords.set_index(['block_id', 'patch_id'], inplace=True)
    assert df_coords.index.is_unique, "Coordinates index is not unique."

    return df_coords

def features_to_df(feats_dir, ext = ".h5"):

    feats_dir = Path(feats_dir)
    file_paths = [f for f in feats_dir.glob("*" + ext) if f.is_file()]
    file_paths = sorted(file_paths)

    dataframes = []
    for file_path in file_paths:
        df = h5_to_pandas(file_path, key='features')
        df['patch_id'] = df.index
            # Save or process the DataFrame as needed
        dataframes.append(df)

    df_features = pd.concat(dataframes, ignore_index=True)
    df_features.drop(columns=['savetodir'], inplace=True)
    df_features.rename(columns={'name': 'block_id'}, inplace=True)
    df_features.set_index(['block_id', 'patch_id'], inplace=True)
    encoder_col = df_features.columns[-1]
    encoder = df_features[encoder_col].iloc[0]
    df_features.drop(columns=['encoder'], inplace=True)
    ## rename all remaining columns to be feat_*
    df_features.columns = [f"{encoder}_{col}" for col in df_features.columns]

    assert df_features.index.is_unique, "Features index is not unique."
    return df_features


def _prepare_metadata(metadata_path, id_cols=['code', 'slide_id']):
    data = pd.read_csv(metadata_path)
    ### drop rows with missing values in id cols
    data.dropna(subset=id_cols, inplace=True)
    data.set_index(id_cols, inplace=True)

    assert data.index.is_unique, "Metadata index is not unique."

    return data


def construct_save_dir(base_dir, target_mag, patch_size, overlap, patch_extractor=None, slide_extractor=None):
    if slide_extractor is not None:
        path = os.path.join(base_dir,'data', f"{slide_extractor}_{target_mag}x_{patch_size}px_{overlap}ov")

    elif patch_extractor is not None:
       path = os.path.join(base_dir, 'data', f"{patch_extractor}_{target_mag}x_{patch_size}px_{overlap}ov")

    else:
        path = os.path.join(base_dir, 'data', f"{target_mag}x_{patch_size}px_{overlap}ov")

    return path
# %%