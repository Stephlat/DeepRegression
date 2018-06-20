# DeepRegression
A Comprehensive Analysis of Deep Regression

pgr="VGGStandard.py"
pathData="/services/scratch/perception/dataBiwi/"
TRset="trainingAnnotations.txt"
Testset="testAnnotations.txt"

Low_dim=3
PbFlag="biwi"
OAR_JOB_ID=1278
option="-bn"

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,exception_verbosity='high' python $pathData $TRset $Testset $Low_dim $PbFlag $OAR_JOB_ID $option

Options:
BatchNormalization:
-bn: with BN
-bnba: with BN before the last activation
-nbn: no BN
Finetunning
'-ft x' with x in {0,1,2,3}, finetune x blocks
        nbBlock=int(sys.argv[idarg+1])
Batch size
      '-bs x': use batches of size x
Optimizer:
'-opt x' with x in {adam, adadelta, rmsprop,adagrad}

Regression layer:
'-rf x' with x in {cov,fc1}. otherwise the default value is fc2

Dropout
'-do x': with x in {-1,0,1,2}
-1: 00 in the paper
0: 10
1: 01
2: 11