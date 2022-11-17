# REPAIR: REnormalizing Permuted Activations for Interpolation Repair

The code for this work is organized as a series of standalone Jupyter Notebooks.

* The notebook [Train-Merge-REPAIR-VGG11](notebooks/Train-Merge-REPAIR-VGG11.ipynb) is a self-contained, PyTorch-only script which trains two standard VGG11 networks on CIFAR-10. It then merges them in weight-space using neuron alignment. Finally, it applies REPAIR to reduce the merging penalty from ~20% to ~5%. The width of the VGG network can be scaled up to obtain zero-barrier connectivity.
* The notebook [Merge-ResNet50-ImageNet](notebooks/Merge-ResNet50-ImageNet.ipynb) applies REPAIR to merging ResNet50s trained on ImageNet. It is less self-contained, depending on [FFCV](https://github.com/libffcv/ffcv) ImageNet files, and two pretrained ResNet50 checkpoints.

These two notebooks contain complete implementations of REPAIR from which further experiments can be derived. The remainder of the notebooks in `./notebooks` contain assorted such experiments.

* arxiv: [https://arxiv.org/abs/2211.08403](https://arxiv.org/abs/2211.08403)

