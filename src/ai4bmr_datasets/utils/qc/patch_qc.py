## file that contains the functions to extract features from patches (such as tissue proportion, intensity features, etc.)
## can be saved as patch information in the anndata (with prefix 'patch_')

import cv2
import numpy as np
import sys
#from skimage.feature import graycomatrix, graycoprops
from trident import OpenSlideWSI
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm  

# %%
def calculate_tissue_proportion(patch, visualize=False):
    """
    Calculate the proportion of tissue in a patch.
    Args:
        patch (np.ndarray): Image patch to analyze.
        visualize (bool): Whether to visualize the binary mask.
    Returns:
        float: Proportion of tissue in the patch.
    """
    # Convert patch to grayscale
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    
    # Threshold the image to binary
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Calculate tissue proportion
    tissue_proportion = np.sum(binary == 0) / (patch.shape[0] * patch.shape[1])
    
    if visualize:
        plt.imshow(binary, cmap='gray')
        plt.axis('off')
        plt.title(f"Tissue proportion: {tissue_proportion:.2f}")
        plt.show()
    
    return tissue_proportion
# %%
def calculate_basic_intensity_features(patch, visualize=False):
    """
    Calculate basic intensity features of a patch.
    Args:
        patch (np.ndarray): Image patch to analyze.
        visualize (bool): Whether to visualize the patch.
    Returns:
        tuple: Mean, standard deviation, and median intensity values for each channel.
    """
    mean_intensity = np.mean(patch, axis=(0,1))  # Mean per channel
    std_intensity = np.std(patch, axis=(0,1))
    median_intensity = np.median(patch, axis=(0,1))

    return mean_intensity, std_intensity, median_intensity

## Needs to be explained and checked -> not yet ready for use
# %%
# def calculate_glcm_features(patch):
#     # Convert patch to grayscale
#     gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    
#     # Calculate GLCM matrix
#     glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
    
#     # Calculate GLCM properties
#     contrast = graycoprops(glcm, prop='contrast')[0, 0]
#     dissimilarity = graycoprops(glcm, prop='dissimilarity')[0, 0]
#     homogeneity = graycoprops(glcm, prop='homogeneity')[0, 0]
#     energy = graycoprops(glcm, prop='energy')[0, 0]
#     correlation = graycoprops(glcm, prop='correlation')[0, 0]
    
#     return contrast, dissimilarity, homogeneity, energy, correlation

# %%
def calculate_edge_density(patch, visualize=False):
    """
    Calculate the edge density of a patch using Canny edge detection.
    Args:
        patch (np.ndarray): Image patch to analyze.
        visualize (bool): Whether to visualize the edges.
    Returns:
        float: Edge density of the patch.
    """
    # Convert patch to grayscale
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    
    # Calculate edges using Canny edge detector
    edges = cv2.Canny(gray, 100, 200)
    
    # Calculate edge density
    edge_density = np.sum(edges == 255) / (patch.shape[0] * patch.shape[1])
    
    if visualize:
        plt.imshow(edges, cmap='gray')
        plt.axis('off')
        plt.title(f"Edge density: {edge_density:.2f}")
        plt.show()
    
    return edge_density

# %%
def calculate_hsv_features(patch, visualize=False):
    """
    Calculate HSV features of a patch.
    Args:
        patch (np.ndarray): Image patch to analyze.
        visualize (bool): Whether to visualize the HSV channels.
    Returns:
        tuple: Mean and standard deviation of hue, saturation, and value channels.
        Hue = the color type, Saturation = the intensity of the color, Value = the brightness of the color.
    """
    # Convert patch to HSV
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    
    # Calculate mean and std of hue
    mean_hue = np.mean(hsv[:, :, 0])
    std_hue = np.std(hsv[:, :, 0])

    # calculate mean and std of saturation
    mean_saturation = np.mean(hsv[:, :, 1])
    std_saturation = np.std(hsv[:, :, 1])

    # calculate mean and std of value
    mean_value = np.mean(hsv[:, :, 2])
    std_value = np.std(hsv[:, :, 2])
    
    
    return mean_hue, std_hue, mean_saturation, std_saturation, mean_value, std_value

# %%
def calculate_features(patch, visualize=False):
    """
    Calculate various features from a patch.
    Args:
        patch (np.ndarray): Image patch to analyze.
        visualize (bool): Whether to visualize the results.
    Returns:
        dict: Dictionary containing calculated features.
    """
    # Calculate tissue proportion
    tissue_proportion = calculate_tissue_proportion(patch, visualize)
    
    # Calculate basic intensity features
    mean_intensity, std_intensity, median_intensity = calculate_basic_intensity_features(patch, visualize)
    
    # Calculate edge density
    edge_density = calculate_edge_density(patch, visualize)
    
    # Calculate HSV features
    mean_hue, std_hue, mean_saturation, std_saturation, mean_value, std_value = calculate_hsv_features(patch, visualize)
    
    # Calculate GLCM features
    #contrast, dissimilarity, homogeneity, energy, correlation = calculate_glcm_features(patch)
    dict_features = {
        'tissue_proportion': tissue_proportion,
        "mean_intensity_R": mean_intensity[0],
        "mean_intensity_G": mean_intensity[1],
        "mean_intensity_B": mean_intensity[2],
        "std_intensity_R": std_intensity[0],
        "std_intensity_G": std_intensity[1],
        "std_intensity_B": std_intensity[2],
        "median_intensity_R": median_intensity[0],
        "median_intensity_G": median_intensity[1],
        "median_intensity_B": median_intensity[2],
        'edge_density': edge_density,
        'mean_hue': mean_hue,
        'std_hue': std_hue,
        'mean_saturation': mean_saturation,
        'std_saturation': std_saturation,
        'mean_value': mean_value,
        'std_value': std_value
    }

    return dict_features


# %%
def construct_wsi_path(wsi_dir: Path, wsi_name:str, ext:str='.mrxs'):
    """
    Constructs the full path to a WSI file based on the directory, name, and extension.
    Args:
        wsi_dir (Path): Directory where the WSI files are stored.
        wsi_name (str): Name of the WSI slide.
        ext (str): File extension of the WSI file, default is '.mrxs'.
    Returns:
        Path: Full path to the WSI file.
    """
    return wsi_dir / (wsi_name + ext)
# %%
def extract_single_patch(wsi_path, location: tuple, level: int, patch_size: int):
    """
    Extracts a single patch from a WSI at a specified location and level.
    Args:
        wsi_path (str or Path): Path to the WSI file.
        location (tuple): Coordinates (x, y) of the patch center.
        level (int): Level of the WSI to extract the patch from.
        patch_size (int): Size of the patch to extract.
    Returns:
        np.ndarray: Extracted patch as a NumPy array.
    """
    wsi = OpenSlideWSI(wsi_path, lazy_init=False)
    area = (patch_size, patch_size)
    patch = np.array(wsi.read_region(location, level, area))
    assert patch.shape == (patch_size, patch_size, 3), f"Patch shape is {patch.shape}, expected {(patch_size, patch_size, 4)}"
    return patch
# %%
def extract_patch_features(df_coords: pd.DataFrame, base_dir: Path, name_col:str='slide_id'):
    """
    Extract features from patches defined by coordinates in a DataFrame.
    Args:
        df_coords (pd.DataFrame): DataFrame containing coordinates and slide identifiers.
        base_dir (Path): Base directory where WSI files are stored.
        level (int): Level of the WSI to extract patches from.
        patch_size (int): Size of the patches to extract.
        ext (str): File extension of the WSI files.
        name_col (str): Column name in df_coords that contains slide identifiers.
    Returns:
        pd.DataFrame: DataFrame containing extracted features for each patch.
    """
    features_list = []
    index_cols  = df_coords.index.names
    df_coords = df_coords.reset_index()
    for i, row in tqdm(df_coords.iterrows(), total=df_coords.shape[0]):
        coords = (row['x'], row['y'])
        factor = int(np.sqrt(row['level0_magnification'] // row['level0_magnification']))
        import math
        level = int(math.log2(factor))
        patch_size = int(row['patch_size'])
        wsi_path = row['sample_path']
        patch = extract_single_patch(wsi_path, coords, level, patch_size)
        features = calculate_features(patch)
        features.update(row[index_cols].to_dict())
        features_list.append(features)


    patch_df = pd.DataFrame(features_list)
    patch_df.set_index(index_cols, inplace=True)
    return patch_df

# %%
def compute_ranks(df):
    """
    Compute ranks for x and y coordinates in the DataFrame and normalize them.
    Args:
        df (pd.DataFrame): DataFrame containing 'x' and 'y' columns.
    Returns:
        pd.DataFrame: DataFrame with additional columns for ranks and normalized ranks.
    """
    df['x_rank'] = df['x'].rank(method="min")
    df['y_rank'] = df['y'].rank(method="min")
    scaler = MinMaxScaler()
    df[['x_rank_norm', 'y_rank_norm']] = scaler.fit_transform(df[['x_rank', 'y_rank']])
    return round(df, 4)