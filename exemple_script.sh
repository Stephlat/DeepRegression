#!/bin/bash


echo "Regression experiments"
JOB_ID=57868 # random job id

#For biwi
pgr="VGGStandard.py"
#pgr="resNetStandard.py"
pathData="/pathToData/"
TRset="trainingAnnotations.txt" # must be located at /pathTODATA/trainingAnnotations.txt
Testset="testAnnotationsNomirror.txt" # idem

Low_dim=3
PbFlag="biwi"
option="-bn"

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,exception_verbosity='high' python $pgr $pathData $TRset $Testset $Low_dim $PbFlag $OAR_JOB_ID $option

