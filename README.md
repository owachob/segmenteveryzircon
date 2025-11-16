# segmenteveryzircon

<p align="center">
<img src="https://github.com/owachob/segmenteveryzircon/blob/main/example_crowded_zircons.png">
</p>

## Description

segmenteveryzircon is a python-based module aimed to extract zircon geometries from 2-D images to enhance geo-thermochronologic datasets and interpretations. Ideally, it provides the ability to measure the size and shape of grains reasonably well to capture their morphometric characteristics for either pre- or post- ablation analyses. 'segmenteveryzircon' is an extentsion of ['segmenteverygrain'](https://github.com/zsylvester/segmenteverygrain), a python package aimed to segment and measure sedimentary clasts more broadly. The underlying architecture for both of these packages is a U-Net convolutional neural network used on top of [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything), a semantic segmentation foundational model developed by Meta. 

Due to the nature of these machine learning models, segmenteveryzircon will likely not produce perfect results automatically. That is why the package includes a human-in-the-loop workflow to allow users to add and delete grains after the initial model output.