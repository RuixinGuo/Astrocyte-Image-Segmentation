# Astrocyte Image Segmentation

Pytorch Segmentation Model for 2D Astrocyte Image Segmentation.

## Dataset

The dataset is packed in the dataset.zip file, including two sets **data1** and **data2**.

+ **data1**: The original 85 astrocyte images and their masks. The masks are manually labeled using [ImageJ](https://imagej.nih.gov/ij/).

+ **data2**: A larger dataset with 1020 images, augmented from data1.

## Program

The program utilizes the framework of [Pytorch-Unet](https://github.com/milesial/Pytorch-UNet) but replaces the Unet model with various pretrained segmentation model from [SMP](https://smp.readthedocs.io/en/latest/index.html).

+ **train**: Train a single model. Transfer learning is available.

+ **ensemble**: Ensemble learning by combining multiple models, each of them is trained independently using **train**. The output is combined using voting method.

## Accuracy

Using default dataset and parameters:

For a single model, the dice score can reach 95% on training set and 77% on validation set.

For an ensemble model, the dice score can reach 95% on training set and 80% on validation set.

