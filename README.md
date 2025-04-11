# A Deep Learning approach to Distance Map generation applied to automatic fiber diameter computation from digital micrographs

This repository contains the source code and datasets to replicate the experimental results of "A Deep Learning approach to Distance Map generation applied to automatic fiber diameter computation from digital micrographs".

The project uses Python with OpenCV and PyTorch. 

[//]: <> (In this repository, we've implemented the code resulting in the paper "Deep regression model of the distance transform for the alpaca fiber diameter measurement". the code in this repository is made with *PyTorch*.)

## Datasets

The samples of real images are located at folder ```datasets/ofda```, while the synthetic dataset should be generated with <https://github.com/alainno/fibergen.git> and should be located at folder: ```datasets/synthetic```.

## Training

* Execute this command: ```python train_unet.py``` for train the model with the U-Net architecture, you can use these parameters:
    * Loss function: ```-l [mae,mse,smooth]```
    * UNet 1st convolution features: ```-nf [16,32,64]```
    * Start learning rate ($10^{lri}$): ```-lri [2,3,4,5,6]```
    * Weight decay: ```-wdi [3,4,5,6]```
    * Maximum of epochs without improve: ```-m [11,12,...,19]```

* Execute this command: ```python train_skeleton.py``` for train the model with the SkeletonNet architecture, you can use these parameters:
    * Emsemble type: ```-e [inner,outer]```
    * Loss function: ```-l [mae,mse,smooth]```
    * Maximum of epochs without improve: ```-m [11,12,...,19]```

## Tests

You can get the test results (MAE and MSE) executing ```python test.py``` for both architectures, you can use the next parameters:

* Model: ```-a [unet,skeleton]```
* Loss function: ```-l [mae,mse,smooth]```
* Testing subset: ```-ts [synthetic,ofda]```
* U-Net 1st convolution features: ```-nf [16,32,64]```
* U-Net start learning rate ($10^{lri}$): ```-lri [2,3,4,5,6]```
* U-Net weight decay: ```-wdi [3,4,5,6]```
* SkeletonNet emsemble type: ```-e [inner,outer]```

Finally, you can generate test images outputs with ```python output-losses.py``` using the same parameters above, except loss function because it show all losses results.

