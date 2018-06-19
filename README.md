# A Comprehensive Analysis of Deep Regression
This repository contains the code that was used in the experiments of [this paper](https://arxiv.org/abs/1803.08450).

# Under construction, Coming soon!

Tested with keras 1.1.0 with theano backend and python 2.7.12
Requieres the installation of scikit-learn.

------------------
## How to run:

trainingAnnotations.txt must contain the list of the training images followed by the targets:
```
img_name_1.jpg y1 y2 y3
img_name_2.jpg y1 y2 y3 
...
```

validationAnnotations.txt and testAnnotations.txt must contain the list of the validation and test images with the same format.

Download the [VGG16 weights](https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view)

We provide 4 main files:
resNetStandard.py: resnet in the case where the validation set is automatically extracted from the training set.
resNetStandardWithVal.py: resnet in the case where the validation set is given.
VGGStandard.py: VGG16 in the case where the vqlidation set is automatically extracted from the training set.
VGGStandardWithVal.py: VGG16 in the case where the validation set is given.

To run a model (training and test), you can use ex_scrpt.sh as exemple.

The JOB_ID is a job id used to save the network weights. You can give any number. $rootpathData is the path to your dataset folder. The file vgg16_weights.h5 must be moved in the $rootpathData folder.

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

      -1: refered to as 00 in the paper
       0: 10
       1: 01
       2: 11


## Support

For any question, please contact [Stéphane Lathuilière](https://team.inria.fr/perception/team-members/stephane-lathuiliere/).
