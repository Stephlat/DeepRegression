# A Comprehensive Analysis of Deep Regression
This repository contains the code that was used in the experiments of [this paper](https://arxiv.org/abs/1803.08450).


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

Run the following command:
```shell
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,exception_verbosity='high' python vanillaDeepReg.py $rootpathData trainingAnnotations.txt validationAnnotations.txt testAnnotations.txt $JOB_ID
```
where JOB_ID is a job id used to save the network weights. You can give any number. $rootpathData is the path to your dataset folder. The file vgg16_weights.h5 must be moved in the $rootpathData folder.

------------------


## Support

For any question, please contact [Stéphane Lathuilière](https://team.inria.fr/perception/team-members/stephane-lathuiliere/).
