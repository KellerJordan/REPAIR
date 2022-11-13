# REPAIR: REnormalizing Permuted Activations for Interpolation Repair

The code for this work is organized as a series of standalone Jupyter Notebooks.

* [ResNet20](notebooks/Train-and-Permute-CIFAR10-ResNet20.ipynb)
* [VGG11](notebooks/Train-and-Permute-CIFAR10-VGG11.ipynb)
* [VGG11 with BatchNorm](notebooks/Train-and-Permute-CIFAR10-VGG11-BatchNorm.ipynb)
* [MLP](notebooks/Train-and-Permute-MNIST-MLP-NoBatchNorm.ipynb)
* [MLP with BatchNorm](notebooks/Train-and-Permute-MNIST-MLP-BatchNorm.ipynb)
* [ResNet50 trained on ImageNet](notebooks/Permute-ImageNet-ResNet50.ipynb)


Some notebooks use the [FFCV](https://github.com/libffcv/ffcv) dataloader to speed things up.
The script `write.py` should be run first in order to generate `.beton` files.

