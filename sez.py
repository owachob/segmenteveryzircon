import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img  # or your preferred image loader
from pathlib import Path
import segmenteverygrain as seg  # <-- your segmentation module
import pandas as pd
import geopandas as gpd
import os
from shapely import wkt


def plot_grains(fname, all_grains, step, cmap='Paired', figsize=(15, 10),
                save=True, show=True, dpi=300):
    """
    Loads an image and plots it with colorful grains using seg.plot_image_w_colorful_grains().
    Automatically saves the figure as '<fname>_Initialoutput.png' if save=True.

    Parameters
    ----------
    fname : str or Path
        Path to the image file.
    all_grains : object
        Grain data structure expected by seg.plot_image_w_colorful_grains().
    step : str
        The step that you are trying to plot. 'Initial', 'Deletions', 'Additions', or 'Final'.
    cmap : str, optional
        Colormap for coloring grains (default: 'Paired').
    figsize : tuple, optional
        Figure size (default: (15, 10)).
    save : bool, optional
        Whether to automatically save the plot as '<fname>_Initialoutput.png'.
    show : bool, optional
        Whether to display the plot (default: True).
    dpi : int, optional
        Resolution when saving (default: 300).
    """

    fname = Path(fname)
    image = np.array(load_img(fname))

    fig, ax = plt.subplots(figsize=figsize)
    plt.xticks([])
    plt.yticks([])

    # Use your fixed seg module
    seg.plot_image_w_colorful_grains(image, all_grains, ax, cmap=cmap)

    plt.axis('equal')
    plt.xlim([0, np.shape(image)[1]])
    plt.ylim([np.shape(image)[0], 0])

    if step == 'Initial' or 'initial':
        save_path = fname.with_name(fname.stem + '_Initialoutput.png')
    elif step == 'Deletions' or 'deletions' or 'initial deletions' or 'after deleting':
        save_path = fname.with_name(fname.stem + '_after_initial_deletions.png')
    elif step == 'Additions' or 'additions' or 'after additions':
        save_path = fname.with_name(fname.stem + '_after_initial_additions.png')
    elif step == 'Final' or 'final' or 'completed':
        save_path = fname.with_name(fname.stem + '_final.png')
    else:
        print('Error: please specifiy step. Choices are "Initial", "Deletions", "Additions", or "Final")')

    # Save if requested
    if save:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        print(f"✅ Figure saved to: {save_path}")

    # Show or close the figure
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

