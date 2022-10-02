# Linear mode connectivity of deep neural networks via permutation invariance and renormalization

The code for this work is organized as a series of standalone Jupyter Notebooks.

* [ResNet20](notebooks/Train-and-Permute-CIFAR10-ResNet20.ipynb)
* [VGG11](notebooks/Train-and-Permute-CIFAR10-VGG11.ipynb)
* [VGG11 with BatchNorm](notebooks/Train-and-Permute-CIFAR10-VGG11-BatchNorm.ipynb)
* [MLP](notebooks/Train-and-Permute-MNIST-MLP-NoBatchNorm.ipynb)
* [MLP with BatchNorm](notebooks/Train-and-Permute-MNIST-MLP-BatchNorm.ipynb)


Abstract:
> In this paper we empirically investigate the conjecture from Entezari et al. (2021) which states that if permutation invariance is taken into account, then there should be no loss barrier to the linear interpolation between SGD solutions. We conduct our investigation using standard computer vision architectures trained on CIFAR-10 and ImageNet.  First, we observe a general phenomenon in which interpolated deep networks suffer a collapse in the variance of their activations. We demonstrate that an appropriate rescaling of the pre-activations of the interpolated networks ameliorates this problem and significantly reduces the barrier. Second, by combining this with an algorithm for finding permutations based on maximizing correlations between the activations of matched neurons, we are able to reduce the interpolation barrier for a standard ResNet18 trained on CIFAR-10 to 1.5\% absolute test error. We explore the interaction between our method and the choice of normalization layer, and demonstrate its robustness across a variety of architectures and training sets.


Some notebooks use the [FFCV](https://github.com/libffcv/ffcv) dataloader to speed things up.
The script `write.py` should be run first in order to generate `.beton` files.

