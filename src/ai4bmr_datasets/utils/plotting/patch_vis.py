# %%
from pathlib import Path
from trident import OpenSlideWSI
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from tqdm import tqdm
import cv2
from PIL import Image
import matplotlib.colors as mcolors
#%%

def hex_to_rgb_dict(hex_colors, categories):
    """
    Convert a list of hex colors to a dictionary with normalized RGB tuples.
    
    Parameters:
        hex_colors (list): List of hex color strings.
        categories (list): List of category names.
        
    Returns:
        dict: Dictionary mapping category names to normalized RGB values.
    """
    if len(hex_colors) < len(categories):
        raise ValueError("Not enough colors for all categories!")
    
    # Convert hex to normalized RGB
    rgb_colors = [mcolors.to_rgb(hex_color) for hex_color in hex_colors[:len(categories)]]
    
    return dict(zip(categories, rgb_colors))

def get_colors(n):
    """Generate n distinct colors using seaborn or a fallback method."""
    if n <= 20:
        cmap = plt.cm.get_cmap('tab20', n)
    elif n <= 40:
        cmap = plt.cm.get_cmap('tab20b', n)
    elif n <= 60:
        cmap = plt.cm.get_cmap('tab20c', n)
    else:
        cmap = sns.color_palette("husl", n)  # HUSL generates distinct colors even for large n
    return cmap



def get_palette(adata, col):
    palette_name = col + '_palette'
    if palette_name in adata.uns.keys():
        print('palette already exists')
    else:
        ## check if columns as assigned colors
        default_name = col + '_colors'
        if default_name in adata.uns.keys():
            palette = hex_to_rgb_dict(adata.uns[default_name], adata.obs[col].cat.categories)
            adata.uns[palette_name] = palette
            print(f'Palette created for {col}')


    return adata.uns[palette_name]


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
def plot_multiple_patches(patches: list, border_colors=None, border_thickness=5,
                          title=None, subtitles=None):
    """
    Plots multiple patches in a single figure with optional borders and titles.
    Args:
        patches (list): List of patches to plot, each patch should be a NumPy array.
        border_colors (list, optional): List of colors for the borders of each patch.
        border_thickness (int, optional): Thickness of the border around each patch.
        title (str, optional): Title for the entire figure.
        subtitles (list, optional): List of subtitles for each patch.
    Returns:
        fig (Figure): Matplotlib figure containing the patches.
        axes (list): List of axes corresponding to each patch.
    """

    # Create Matplotlib figure
    num_patches = len(patches)
    fig, axes = plt.subplots(1, num_patches, figsize=(num_patches * 3, 3))

    # Ensure axes is iterable even if there's only 1 patch
    if num_patches == 1:
        axes = [axes]

    # Plot each patch with a border
    for i, ax in enumerate(axes):
        patch = patches[i]
        patch_size = patch.shape[0]
        ax.imshow(patch)
        ax.axis("off")


        # Draw border if border_colors is provided
        if border_colors:
            color = border_colors[i] if i < len(border_colors) else "black"  # Default to black if missing
            rect = Rectangle((0, 0), patch_size, patch_size, linewidth=border_thickness, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
        
        if subtitles:
            ax.set_title(subtitles[i])
        
        if title:
            fig.suptitle(title, fontsize=24)

    return fig, axes
# %%
def extract_multiple_patches(df_coords: pd.DataFrame, level: int, patch_size: int):
    """
    Extracts multiple patches from a WSI based on coordinates provided in a DataFrame.
    Args:
        df_coords (pd.DataFrame): DataFrame containing coordinates and slide identifiers.
        base_dir (Path): Base directory where WSI files are stored.
        level (int): Level of the WSI to extract the patches from.
        patch_size (int): Size of the patches to extract.
        ext (str): File extension of the WSI files, default is '.mrxs'.
        name_col (str): Column name in df_coords that contains slide identifiers.
    Returns:
        list: List of extracted patches as NumPy arrays.
    """
    patches = []
    for i, row in df_coords.iterrows():
        coords = (row['x'], row['y'])
        wsi_path = row['sample_path'] 
        patch = extract_single_patch(wsi_path, coords, level, patch_size)
        patches.append(patch)
    return patches

# %%
def get_border_colors(df: pd.DataFrame, label_col: str, color_dict: dict=None):
    """
    Returns a list of border colors for each row in the DataFrame based on the specified label column.
    Args:
        df (pd.DataFrame): DataFrame containing the label column.
        label_col (str): Column name in df that contains the labels.
        color_dict (dict, optional): Dictionary mapping labels to colors. If None, uses a default colormap.
    Returns:
        list: List of colors corresponding to each row in the DataFrame.
    """
    if color_dict is None:
        ## take categories from the label column (error if not categorical)
        assert df[label_col].dtype.name == 'category', 'Label column must be categorical'
        color_keys = df[label_col].cat.categories
        color_dict = {key: plt.cm.tab20(i) for i, key in enumerate(color_keys)}
    
    border_colors = []
    for i, row in df.iterrows():
        label = row[label_col]
        border_colors.append(color_dict[label])
    return border_colors



# %%
## sample 10 random numbers 
def sample_random_coords(df_coords, num_samples=10):
    """
    Samples a specified number of random coordinates from a DataFrame.
    Args:
        df_coords (pd.DataFrame): DataFrame containing spatial coordinates.
        num_samples (int): Number of random samples to extract.
    Returns:
        pd.DataFrame: DataFrame containing the sampled coordinates.
    """
    return df_coords.sample(num_samples)



def extract_custom_thumbnail(wsi_path, max_size=10000, level0_magnification=80, target_magnification=20, patch_size=256):
    """
    Extracts a custom thumbnail from a WSI at a specified size and downsample factor.
    Args:
        wsi_path (str or Path): Path to the WSI file.
        max_size (int): Maximum size of the thumbnail image.
        level0_magnification (int): Magnification level of the source image.
        target_magnification (int): Target magnification level for the thumbnail.
        patch_size (int): Size of the patches to extract.
    Returns:
        np.ndarray: Thumbnail image as a NumPy array.
        float: Downsample factor for the thumbnail.
        int: Size of the patches in the thumbnail.
    """
    wsi = OpenSlideWSI(wsi_path, lazy_init=False)

    max_dimension = 10000
    if wsi.width > wsi.height:
        thumbnail_width = max_dimension
        thumbnail_height = int(thumbnail_width * wsi.height / wsi.width)
    else:
        thumbnail_height = max_dimension
        thumbnail_width = int(thumbnail_height * wsi.width / wsi.height)

    downsample_factor = wsi.width / thumbnail_width

    patch_size_src = round(patch_size * level0_magnification / target_magnification)
    thumbnail_patch_size = max(1, int(patch_size_src / downsample_factor))

    
    return wsi.get_thumbnail((thumbnail_width, thumbnail_height)), downsample_factor, thumbnail_patch_size

#%%

def visualize_clusters(df_coords, wsi_path, cluster_col='leiden', name_col = 'slide_id',
                       mag_scr=80, mag_target=20, patch_size=256,
                       plot_prefix=None, color_dict=None, result_dir=None, size_thumbnail=10000, save_as_pdf=False):
    """
    Visualizes clusters by overlaying colored rectangles on a thumbnail of the WSI.
    Args:
        df_coords (pd.DataFrame): DataFrame containing coordinates and cluster labels.
        wsi_path (str or Path): Path to the WSI file.
        cluster_col (str): Column name in df_coords that contains cluster labels.
        name_col (str): Column name in df_coords that contains slide identifiers.
        mag_scr (int): Magnification level of the source image.
        mag_target (int): Target magnification level for visualization.
        patch_size (int): Size of the patches to visualize.
        plot_prefix (str, optional): Prefix for the plot title.
        color_dict (dict, optional): Dictionary mapping cluster labels to colors. If None, uses a default colormap.
        result_dir (Path, optional): Directory where the visualization will be saved. If None, uses the current working directory.
        size_thumbnail (int): Maximum size of the thumbnail image.
    Returns:
        None: Saves the visualization as an image file.
    """
    
    assert df_coords[name_col].nunique() == 1, "All coordinates must belong to the same slide"
    wsi_name = df_coords[name_col].iloc[0]
    #wsi_path = construct_wsi_path(base_dir, wsi_name)


    ### to do: either pass the patch size, magnifications etc or pass directly thumbnail, patch size and downsample factor
    thumbnail, downsample_factor, patch_size_thumbnail = extract_custom_thumbnail(wsi_path, max_size=size_thumbnail, level0_magnification=mag_scr, target_magnification=mag_target, patch_size=patch_size)

    ## create colormap
    if color_dict is None:
        clusters = df_coords[cluster_col].unique()
        color_dict = {cluster: plt.cm.tab20(i) for i, cluster in enumerate(clusters)}
    
    custom_colors_bgr = {
            cluster: tuple(int(c * 255) for c in (color))  # Reverse RGB to BGR and scale to [0, 255]
            for cluster, color in color_dict.items()
    }

    coords = df_coords[['x', 'y']].values
    cluster_labels = df_coords[cluster_col].values

    plot_name = f"{plot_prefix}_{cluster_col}_patches.png" if plot_prefix else f"{wsi_name}_{cluster_col}_patches.png"

    canvas = np.array(thumbnail).astype(np.uint8)

    alpha = 0.5  # Adjust alpha for transparency (0 is fully transparent, 1 is fully opaque)

    for (x, y), cluster in zip(coords, cluster_labels):
        # Convert coordinates to appropriate scale
        x, y = int(x / downsample_factor), int(y / downsample_factor)
        thickness = max(1, patch_size_thumbnail // 10)
        #print(cluster)
        # Get the BGR color for the cluster
        color = custom_colors_bgr[cluster]
        if len(color) == 4:
            color = color[:3]
        #print(color)
        
        # Define the rectangle area
        rect = (x, y, x + patch_size_thumbnail, y + patch_size_thumbnail)
        
        # Extract the region of the canvas where the rectangle will be drawn
        canvas_patch = canvas[rect[1]:rect[3], rect[0]:rect[2]]
        
        # Create a transparent rectangle (BGR) of the same size
        transparent_rect = np.zeros_like(canvas_patch, dtype=np.uint8)
        transparent_rect[:] = color  # Set the color of the rectangle
        
        # Blend the rectangle with the canvas using the alpha value
        cv2.addWeighted(transparent_rect, alpha, canvas_patch, 1 - alpha, 0, canvas_patch)


    min_x = int(coords[:, 0].min() / downsample_factor)
    min_y = int(coords[:, 1].min() / downsample_factor)
    max_x = int(coords[:, 0].max() / downsample_factor)
    max_y = int(coords[:, 1].max() / downsample_factor)

    cropped_canvas = canvas[min_y:max_y, min_x:max_x]


    # Add title
    plot_title = False
    if plot_title:
        title = plot_prefix or "Cluster Visualization"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        font_thickness = 3
        text_size = cv2.getTextSize(title, font, font_scale, font_thickness)[0]
        text_x = (cropped_canvas.shape[1] - text_size[0]) // 2
        text_y = text_size[1] + 20

        cv2.putText(cropped_canvas, title, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    if save_as_pdf:
        viz_coords_path = result_dir / plot_name
        viz_coords_path.parent.mkdir(parents=True, exist_ok=True)

        im = Image.fromarray(cropped_canvas)

        # Save as PNG (or whatever extension you gave `plot_name`)
        im.save(viz_coords_path)

        # Save also as PDF with 300 dpi
        pdf_path = viz_coords_path.with_suffix(".pdf")
        im.save(pdf_path)
        viz_coords_path = pdf_path
    else:
        viz_coords_path = result_dir / plot_name
        viz_coords_path.parent.mkdir(parents=True, exist_ok=True)
        # Save the cropped canvas as an image
        Image.fromarray(cropped_canvas).save(viz_coords_path)

    print(f"Visualized patches saved at: {viz_coords_path}")
# %%
