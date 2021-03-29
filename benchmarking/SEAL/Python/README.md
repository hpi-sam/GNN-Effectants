Benchmarking - SEAL
===============================================================================
Adjusted from the orginal implementaion found at (https://github.com/muhanzhang/SEAL)


Installation
------------

Install [PyTorch](https://pytorch.org/).

Type

    bash ./install.sh

to install the required software and libraries. It will download and install the default graph neural network software [\[pytorch_DGCNN\]](https://github.com/muhanzhang/pytorch_DGCNN) to the same level as the root SEAL folder (not this Python folder).

However we had the experience that the make step in the install script tends to not achieve its purpose. We had to manually go to pytorch_DGCNN/lib and excute ```make clean``` and then ```make -j4``` manually.

Furthermore we adjusted the main.py of pytorch_DGCNN to evalute PR-AUC. The file in the clone repository has to be replaced by the one located in SEAL/modified_DGCNN

Usage
------
Type

    bash ./loop.sh

This will execute all SEAL runs with and without Node2Vec.


Requirements
------------

We ran our benchmarks with the combination: Python 3.8.1 + Pytorch 1.8.1

Required python libraries: gensim and scipy; all python libraries required by pytorch_DGCNN such as networkx, tqdm, sklearn etc.