import cv2
import itertools
import json
import os
import sys
import warnings
from glob import glob
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
import rtree
import scipy.ndimage as ndi
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from tqdm import trange, tqdm

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as mpl_Polygon

from PIL import Image
from skimage import measure
from skimage.feature import peak_local_max
from skimage.morphology import binary_dilation, binary_erosion
from skimage.segmentation import watershed
from skimage.measure import find_contours, label, regionprops, regionprops_table

import tensorflow as tf
from keras import Model
from keras.layers import BatchNormalization, Concatenate, Conv2D, Conv2DTranspose, Input, MaxPooling2D
from keras.optimizers import Adam
from keras.saving import load_model
from keras.utils import load_img

import segmenteverygrain as seg
from segment_anything import SamPredictor
from shapely import wkt
from shapely.affinity import translate
from shapely.geometry import MultiPolygon, Point, Polygon, mapping, shape



def plot_image_w_colorful_grains(image, all_grains, ax, cmap='viridis', plot_image=True, im_alpha=1.0):

    cmap = plt.cm.get_cmap(cmap)
    num_colors = len(all_grains)
    color_indices = np.random.randint(0, cmap.N, num_colors)
    colors = [cmap(i) for i in color_indices]

    if plot_image:
        ax.imshow(image, alpha=im_alpha)

    for i in trange(len(all_grains)):
        color = colors[i]
        poly = all_grains[i]
        ax.fill(poly.exterior.xy[0], poly.exterior.xy[1], facecolor=color, edgecolor='none', alpha=0.5)
        ax.plot(poly.exterior.xy[0], poly.exterior.xy[1], color='k', linewidth=1)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)


def plot_grains(fname, all_grains, step, cmap='Paired', figsize=(15, 10),
                save=True, show=True, dpi=300):
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    from keras.utils import load_img

    fname = Path(fname)
    image = np.array(load_img(fname))

    fig, ax = plt.subplots(figsize=figsize)

    plot_image_w_colorful_grains(image, all_grains, ax=ax, cmap=cmap)

    ax.axis('equal')

    step_lower = step.lower()
    if step_lower == 'initial':
        save_path = fname.with_name(fname.stem + '_Initialoutput.png')
    elif step_lower in ['deletions', 'initial deletions', 'after deleting']:
        save_path = fname.with_name(fname.stem + '_after_initial_deletions.png')
    elif step_lower in ['additions', 'after additions']:
        save_path = fname.with_name(fname.stem + '_after_initial_additions.png')
    elif step_lower in ['final', 'completed']:
        save_path = fname.with_name(fname.stem + '_final.png')
    else:
        print('Error: please specify a valid step.')
        plt.close(fig)
        return

    fig.tight_layout()

    if save:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        print(f"✅ Figure saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)



def load_polygons(csv_path, crs=None):
    """
    Load a CSV file containing WKT geometries and convert it to a GeoDataFrame.
    
    Parameters:
        csv_path (str): Path to the CSV file.
        crs (str or dict, optional): Coordinate reference system to assign to the GeoDataFrame.
        
    Returns:
        gpd.GeoDataFrame: GeoDataFrame with geometries loaded from the CSV.
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Convert WKT geometry strings to shapely geometries
    df['geometry'] = df['geometry'].apply(wkt.loads)
    
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=crs)
    
    print(gdf.head())
    print(f"Number of polygons loaded: {len(gdf)}")
    
    return gdf


def create_metadata_table(image_fname, final_n, model_fname, save_csv=False):
    """
    Creates a pandas dataframe and optionally a CSV file containing relevant metadata for segmentation results.

    Parameters:
        image_fname (str): Path to original image
        final_n (int): The final number of grains, after segmentation is completed 
        model_fname (str): Path to SAM model
        save_csv (bool, optional): Whether to save the metadata as a CSV. Default is False.

    Returns:
        pd.DataFrame: Pandas DataFrame with metadata information
    """
    # Create dataframe
    df_summary = pd.DataFrame([{
        'Sample Image': image_fname,
        'Final Number of Grains': final_n,
        'Model Used': model_fname
    }])
    
    # Save CSV if requested
    if save_csv:
        # Extract the sample name without extension
        sample_name = os.path.splitext(os.path.basename(image_fname))[0]
        csv_fname = f"{sample_name}_metadata.csv"
        df_summary.to_csv(csv_fname, index=False)
        print(f"Metadata saved to {csv_fname}")
    
    return df_summary

def create_train_val_test_data(image_dir, mask_dir, augmentation=True):
    """
    Splits image and mask data into training, validation, and test datasets,
    automatically matching images and masks even if extensions differ.
    """
    import os
    # Allow multiple formats
    valid_ext = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff')

    # Gather images and masks
    image_files = []
    for ext in valid_ext:
        image_files.extend(glob(os.path.join(image_dir, ext)))

    mask_files = []
    for ext in valid_ext:
        mask_files.extend(glob(os.path.join(mask_dir, ext)))

    image_files = sorted(image_files)
    mask_files = sorted(mask_files)

    # --- Match images and masks by base filename ---

    image_map = {os.path.splitext(os.path.basename(f))[0]: f for f in image_files}
    mask_map  = {os.path.splitext(os.path.basename(f))[0]: f for f in mask_files}

    # Only keep files that have BOTH image + mask
    common_keys = sorted(set(image_map.keys()) & set(mask_map.keys()))

    if len(common_keys) == 0:
        raise ValueError("No matching image/mask pairs found. Check file naming.")

    image_files = [image_map[k] for k in common_keys]
    mask_files  = [mask_map[k] for k in common_keys]

    print(f"Matched {len(image_files)} image/mask pairs.")

    # ----------------------
    #   Train/Val/Test split
    # ----------------------
    batch_size = 32
    shuffle_buffer_size = 1000

    train_val_images, test_images, train_val_masks, test_masks = train_test_split(
        image_files,
        mask_files,
        test_size=0.15,
        random_state=42
    )

    train_images, val_images, train_masks, val_masks = train_test_split(
        train_val_images,
        train_val_masks,
        test_size=0.25,
        random_state=42
    )

    # ----------------------
    #   Build TF datasets
    # ----------------------
    if not augmentation:
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (train_images, train_masks, tf.Variable([True] * len(train_images), dtype=tf.bool))
        )

    train_dataset = (
        train_dataset
        .map(seg.load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(shuffle_buffer_size)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_dataset = (
        tf.data.Dataset.from_tensor_slices((val_images, val_masks))
        .map(seg.load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(shuffle_buffer_size)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_dataset = (
        tf.data.Dataset.from_tensor_slices((test_images, test_masks))
        .map(seg.load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(shuffle_buffer_size)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_dataset, val_dataset, test_dataset

def grains_to_geodataframe(image_fname, all_grains):
    """
    Projects grain polygons from image coordinates to spatial coordinates
    based on the raster's transform, returning a GeoDataFrame.

    Parameters:
    - image_fname: str, path to the raster image
    - all_grains: list of shapely Polygon objects in image coordinates

    Returns:
    - gdf: GeoDataFrame with projected polygons
    """
    dataset = rasterio.open(image_fname)
    projected_polys = []

    for grain in all_grains:
        x, y = rasterio.transform.xy(
            dataset.transform, grain.exterior.xy[1], grain.exterior.xy[0]
        )
        poly = Polygon(np.vstack((x, y)).T)
        projected_polys.append(poly)

    gdf = gpd.GeoDataFrame(projected_polys, columns=["geometry"])
    return gdf

def convert_grain_units(grain_data, units_per_pixel):
    """
    Convert grain measurement columns from pixels to microns (or nanometers or millimeters).

    Parameters
    ----------
    grain_data : pandas.DataFrame
        DataFrame containing grain properties.
    units_per_pixel : float
        Conversion factor from pixels to microns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with additional columns converted to microns.
    """
    grain_data = grain_data.copy()

    # Area and length measurements
    grain_data['area_micron2'] = grain_data['area'] * (units_per_pixel ** 2)
    grain_data['perimeter_micron'] = grain_data['perimeter'] * units_per_pixel
    grain_data['major_axis_length_micron'] = grain_data['major_axis_length'] * units_per_pixel
    grain_data['minor_axis_length_micron'] = grain_data['minor_axis_length'] * units_per_pixel

    # Centroid coordinates
    grain_data['centroid_row_micron'] = grain_data['centroid-0'] * units_per_pixel
    grain_data['centroid_col_micron'] = grain_data['centroid-1'] * units_per_pixel

    # Bounding box coordinates
    grain_data['bbox_min_row_micron'] = grain_data['bbox-0'] * units_per_pixel
    grain_data['bbox_min_col_micron'] = grain_data['bbox-1'] * units_per_pixel
    grain_data['bbox_max_row_micron'] = grain_data['bbox-2'] * units_per_pixel
    grain_data['bbox_max_col_micron'] = grain_data['bbox-3'] * units_per_pixel

    return grain_data

def show_grain_overlay(grain_ID, grain_data, label_image, original_image, pad=50):
    """
    Display a zoomed-in overlay of a single grain on the original image.

    Parameters
    ----------
    grain_ID : int
        The label ID of the grain to display.
    grain_data : pandas.DataFrame
        DataFrame containing grain properties, including bounding boxes.
    label_image : np.ndarray
        Labeled image where each pixel has a grain ID.
    original_image : np.ndarray
        Original RGB or grayscale image.
    pad : int, optional
        Number of pixels to pad around the grain bounding box.
    """
    row = grain_data[grain_data['label'] == grain_ID].iloc[0]
    min_row, min_col = int(row['bbox-0']), int(row['bbox-1'])
    max_row, max_col = int(row['bbox-2']), int(row['bbox-3'])

    # Expand bounding box
    min_row = max(min_row - pad, 0)
    min_col = max(min_col - pad, 0)
    max_row = min(max_row + pad, original_image.shape[0])
    max_col = min(max_col + pad, original_image.shape[1])

    # Create binary mask for this grain
    label_crop = label_image[min_row:max_row, min_col:max_col]
    mask = (label_crop == grain_ID).astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    offset_contours = [c + [min_col, min_row] for c in contours]  # shift to full image coords

    # Draw on zoomed crop
    zoomed_view = original_image[min_row:max_row, min_col:max_col].copy()
    contour_in_zoomed_coords = [c - [min_col, min_row] for c in offset_contours]
    cv2.drawContours(zoomed_view, contour_in_zoomed_coords, -1, (0, 255, 0), 2)

    # Display
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(zoomed_view, cv2.COLOR_BGR2RGB))
    plt.title(f"Grain {grain_ID} Overlay (Zoomed In)")
    plt.axis('off')
    plt.show()