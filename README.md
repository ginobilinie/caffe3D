# caffe3D

This copy of caffe3D is based on the offical caffe platform, but we have developped the following (not limited to) new functions to work on 3D environment:

1. 3D operations (conv/deconv/pooling...)

2. Pixel-wise (voxel-vise) Weighted Euclidean Loss

3. Pixel-wise (voxel-vise) Weighted Softmax Loss

4. Category-wise Weighted Softmax Loss

5. Smooth L1 Loss

6. CRF as RNN (combine CRF into the FCN-like segmnetaiton network and train end-to-end)

7. Video Type Data Reader

8. Multilinear interpolation (official is bilinear, as medical image is usually 3D)
...
It will be very kind of you if you cite our paper (for which, we develop the caffe3D) when you feel this repository is helpful for you:

"Dong Nie, Li Wang, Ehsan Adeli, Cuijing Lao, Weili Lin, Dinggang Shen. 3D Fully Convolutional Networks for Multi-Modal Isointense Infant Brain Image Segmentation, IEEE Transactions on Cybernetics, 2018." (for details, you can refer to https://github.com/ginobilinie/infantSeg)

The comiplation have the same requirement with the official caffe. And we suggest to use the following environment:
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;unbuntu 14.04 (or 16.04)
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;cuda 8.0
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;cudnn 5.1
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;After compilation, please refer to https://github.com/ginobilinie/infantSeg for more example about how to use it.


The following part is directly from the offical caffe readme:

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }

