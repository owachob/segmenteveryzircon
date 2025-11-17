# segmenteveryzircon

<p align="center">
<img src="https://github.com/owachob/segmenteveryzircon/blob/main/example_images/example_crowded_zircons.png">
</p>

## Description

segmenteveryzircon is a python-based module aimed to extract zircon geometries from 2-D images to enhance geo-thermochronologic datasets and interpretations. Ideally, it provides the ability to measure the size and shape of grains reasonably well to capture their morphometric characteristics for either pre- or post- ablation analyses. 'segmenteveryzircon' is an extentsion of ['segmenteverygrain'](https://github.com/zsylvester/segmenteverygrain), a python package aimed to segment and measure sedimentary clasts more broadly. The underlying architecture for both of these packages is a U-Net convolutional neural network used on top of [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything), a semantic segmentation foundational model developed by Meta. 

Due to the nature of these machine learning models, segmenteveryzircon will likely not produce perfect results automatically. That is why the package includes a human-in-the-loop workflow to allow users to add and delete grains after the initial model output.

## Requirements

- numpy
- scipy
- pandas
- matplotlib
- scikit-image
- scikit-learn
- tqdm
- networkx
- shapely
- geopandas
- rtree
- rasterio
- tifffile
- seaborn
- pyqt5
- tensorflow
- opencv-python
- pillow
- torch
- torchvision
- segment-anything
- segmenteverygrain

## Getting Started

To begin creating grain polygons from images, see the [1_create_polygons_and_masks.ipynb](https://github.com/owachob/segmenteveryzircon/blob/main/1_create_polygons_and_masks.ipynb) notebook to see how to use the model and interact with the initial model output.

Segmentation run time will be dependent upon image size and computer resources. Please consider [downsampling](https://visionbook.mit.edu/upsamplig_downsampling_2.html) images. 

## Acknowledgements

This work is in collaboration with Daniel Stockli, Zoltan Sylvester, and Matthew Malkowski. Special thanks to Daniel Ruiz-Arriaga, Sandra Juarez-Zuniga, Edguardo Pujols, and Rachel Kramer for training images. Thank you to Rowan Martindale and Lisa Stockli for the imaging resources and helpful discussions. This work was made possible by the [UTChron Lab](https://www.jsg.utexas.edu/utchron-lab/) at the Jackson School of Geosciences, the University of Texas at Austin.

## License

segmenteveryzircon is licensed under the [Apache License 2.0](https://github.com/owachob/segmenteveryzircon/blob/main/LICENSE).