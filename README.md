# A Comprehensive Analysis of Deep Regression
This repository contains the code that was used in the experiments of [this paper](https://arxiv.org/abs/1803.08450).

Tested with keras 1.1.0 with theano backend and python 2.7.12.

Requieres the installation of scikit-learn.

------------------
## How to run:

We recommend you to use our exemple_script.sh. In this file you can specify the dataset and the options you want to use.

### Data
trainingAnnotations.txt must contain the list of the training images followed by the targets:
```
path_img_name_1.jpg y1 y2 y3
path_img_name_2.jpg y1 y2 y3 
...
```
validationAnnotations.txt and testAnnotations.txt must contain the list of the validation and test images with the same format.

Importantly the images and annotation files must be located at /pathToData/. In other words, 'path_img_name_1.jpg" is the path of the first traininng image relatively to /pathToData/. 

### Pretrained weights

 * Download the [VGG16 weights](https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view)
 * Download the [resNet weights](https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels.h5)

You need to change the weight file paths in VGG16_sequential.py and resnet50.py.

### Others
We provide 4 main files:
 * resNetStandard.py: resnet in the case where the validation set is automatically extracted from the training set.
 * resNetStandardWithVal.py: resnet in the case where the validation set is given.
 * VGGStandard.py: VGG16 in the case where the vqlidation set is automatically extracted from the training set.
 * VGGStandardWithVal.py: VGG16 in the case where the validation set is given.


The JOB_ID is a job id used to save the network weights. You can give any number. $rootpathData is the path to your dataset folder. 

------------------
## Options:

* BatchNormalization:
-bn: with BN
-bnba: with BN before the last activation
-nbn: no BN

* Finetunning:
'-ft x' with x in {0,1,2,3}, finetune x blocks
        nbBlock=int(sys.argv[idarg+1])

* Batch size:
      '-bs x': use batches of size x

* Optimizer:
'-opt x' with x in {adam, adadelta, rmsprop,adagrad}

* Regression layer:
'-rf x' with x in {cov,fc1}. otherwise the default value is fc2

* Dropout: '-do x': with x in {-1,0,1,2}
    * -1: refered to as 00 in the paper
    * 0: 10
    * 1: 01
    * 2: 11


## Support

For any question, please contact [Stéphane Lathuilière](https://team.inria.fr/perception/team-members/stephane-lathuiliere/).
