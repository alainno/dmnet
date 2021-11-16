# Deep regression model of the distance transform for the alpaca fiber diameter measurement

[comment]: <> (Unofficial implementation of the model described at https://arxiv.org/abs/1907.01683)

In this repository, we've implemented the code resulting in the paper "Deep regression model of the distance transform for the alpaca fiber diameter measurement". the code in this repository is made with *PyTorch*.

## Dataset

The data was generated with <https://github.com/alainno/fibergen.git> and is located at folder: ```./data_dm_overlapping```.

## Running

Execute the following command: ```python train_all.py```. This command will train *U-Net regression* and *SkeletonNet regression*.

## Tests

Test are run on notebooks:

* Fiber diameter Mean and Std: [test_mean_std.ipynb](test_mean_std.ipynb)
* Measurement error: [test_measurement_error.ipyb](test_measurement_error.ipyb)