'''Import modules'''
import time
import sys
import numpy as np
import math
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping,Callback,CSVLogger
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout
from keras.layers.pooling import GlobalMaxPooling2D,GlobalAveragePooling2D
from VGG16_sequential import VGG16
from data_generator import load_data_generator

from test import run_eval

WIDTH = 224
BATCH_SIZE = 128
NB_EPOCH = 50
LEARNING_RATE = 1e-04
PATIENCE=4
BN=True
layer_nb=24
optim='adadelta'

ROOTPATH=sys.argv[1]
train_txt = sys.argv[2]
test_txt = sys.argv[3]
LOW_DIM = int(sys.argv[4])
ssRatio = 1.0  # float(sys.argv[3])/100.0
PB_FLAG = sys.argv[5]  # to modify according to the task
idOar=sys.argv[6]
nbPop=0
dropoutConf=0
pool=None
BNBA=False
epochLength=-1

print sys.argv

for idarg,arg in enumerate(sys.argv):
    if arg=='-bn':
        BN=True
    if arg=='-bnba':
        BN=True
        BNBA=True
    if arg=='-nbn':
        BN=False

    elif arg=='-ft':
        nbBlock=int(sys.argv[idarg+1])
        if nbBlock==0:
            layer_nb=30
        elif nbBlock==1:
            layer_nb=24
        elif nbBlock==2:
            layer_nb=16
        elif nbBlock==3:
            layer_nb=8

    elif arg=='-bs':
        batch_size= int(sys.argv[idarg+1])
    elif arg=='-opt':
        optim=sys.argv[idarg+1]
        if optim=="sgd":
            LEARNING_RATE=float(sys.argv[idarg+2])
            optim = SGD(lr=LEARNING_RATE)
            print "LR " + str(LEARNING_RATE)

    elif arg=='-lr':
        LEARNING_RATE=float(sys.argv[idarg+1])
    elif arg=='-rf':
        if sys.argv[idarg+1]=="conv":
            nbPop=3
        elif sys.argv[idarg+1]=="fc1":
            nbPop=2
    elif arg=='-do':
        dropoutConf=float(sys.argv[idarg+1])
    elif arg=='-pool':
        pool=sys.argv[idarg+1]
        
    if arg=='-el':
        epochLength=int(sys.argv[idarg+1])
    if arg=='-p':
        PATIENCE=int(sys.argv[idarg+1])

        
print optim
            
class L2Model:
    ''' Class of forward model'''

    def __init__(self):


        self.network = VGG16(weights='imagenet')


    def fit(self, (generator_training, n_train), (generator_val, n_val)):
        '''Trains the model for a fixed number of epochs and iterations.
           # Arguments
                X_train: input data, as a Numpy array or list of Numpy arrays
                    (if the model has multiple inputs).
                Y_train : labels, as a Numpy array.
                batch_size: integer. Number of samples per gradient update.
                learning_rate: float, learning rate
                nb_epoch: integer, the number of epochs to train the model.
                validation_split: float (0. < x < 1).
                    Fraction of the data to use as held-out validation data.
                validation_data: tuple (x_val, y_val) or tuple
                    (x_val, y_val, val_sample_weights) to be used as held-out
                    validation data. Will override validation_split.
                it: integer, number of iterations of the algorithm

                

            # Returns
                A `History` object. Its `History.history` attribute is
                a record of training loss values and metrics values
                at successive epochs, as well as validation loss values
                and validation metrics values (if applicable).
            '''

        if pool is None:
            for pop in range(nbPop):
                self.network.pop()
                

            if dropoutConf==-1:
                self.network.layers[-2].rate=0.0
            elif dropoutConf==1:
                self.network.add(Dropout(0.5))
            elif dropoutConf==2:
                self.network.layers[-2].rate=0.0
                self.network.add(Dropout(0.5))

                
        else:
            for pop in range(4):
                self.network.pop()
            if pool=="max":
                self.network.add(GlobalMaxPooling2D())
            elif pool=="avg":
                self.network.add(GlobalAveragePooling2D())
            else:
                print "ERROR: pooling not valide"
                exit(-1)


                
        if BNBA:
            self.network.layers[-1].activation=Activation('linear')
            self.network.add(BatchNormalization())
            self.network.add(Activation('relu'))
        elif BN:

            self.network.add(BatchNormalization())
        self.network.add(Dense(LOW_DIM, activation='linear', trainable=True))

        self.network.summary()
        
        
        # train only some layers
        for layer in self.network.layers[:layer_nb]:
            layer.trainable = False
        for layer in self.network.layers[layer_nb:]:
            layer.trainable = True
        self.network.layers[-1].trainable = True

        # compile the model


        self.network.compile(optimizer=optim,
                             loss='mse',
                             metrics=['mae'])

        self.network.summary()
        csv_logger = CSVLogger(ROOTPATH+"VGG16_"+PB_FLAG+"_"+idOar+'_training.log')


        checkpointer = ModelCheckpoint(filepath=ROOTPATH+"VGG16_"+PB_FLAG+"_"+idOar+"_weights.hdf5",
                                       monitor='val_loss',
                                       verbose=1,
                                       save_weights_only=True,
                                       save_best_only=True,
                                       mode='min')

        early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE)

        
        class CheckNan(Callback):

            def on_batch_end(self, batch, logs={}):
                if math.isnan(logs.get('loss')):
                    print "\nReach a NAN\n"
                    sys.exit()

        # train the model on the new data for a few epochs
        if epochLength<0:
            spe=n_train
        else:
            spe=epochLength
            
        self.network.fit_generator(generator_training,
                                   samples_per_epoch=spe,
                                   nb_epoch=NB_EPOCH*int(n_train/(1.0*spe)),
                                   verbose=1,
                                   callbacks=[checkpointer,csv_logger,
                                              early_stopping,CheckNan()],
                                   validation_data=generator_val,
                                   nb_val_samples=n_val)

        
        self.network.load_weights(ROOTPATH+"VGG16_"+PB_FLAG+"_"+idOar+"_weights.hdf5")
        # self.network.save(ROOTPATH+"VGG16_"+PB_FLAG+"_"+idOar+"_network.hdf5")

            
        

    def predict(self, generator, n_predict):
        '''Generates output predictions for the input samples,
           processing the samples in a batched way.
        # Arguments
            generator: input a generator object.
            batch_size: integer.
        # Returns
            A Numpy array of predictions and GT.
        '''
        '''Extract VGG features and data targets from a generator'''
    
        i=0
        Ypred=[]
        Y=[]
        for x,y in generator:
            if i>=n_predict:
                break
            Ypred.extend(self.network.predict_on_batch(x))
            Y.extend(y)
            i+=len(y)

        return np.asarray(Ypred), np.asarray(Y)

   
    def evaluate(self, (generator, n_eval),flagFile, l=WIDTH, pbFlag=PB_FLAG):
        '''Computes the loss on some input data, batch by batch.

        # Arguments
            generator: input a generator object.
            batch_size: integer. Number of samples per gradient update.

        # Returns
            Scalar test loss (if the model has no metrics)
            or list of scalars (if the model computes other metrics).
            The attribute `model.metrics_names` will give you
            the display labels for the scalar outputs.
        '''
        
        Ypred, Y = self.predict(generator, n_eval)

        run_eval(Ypred, Y, l, pbFlag)
        file = open(ROOTPATH+"VGG_output"+pbFlag+ "_"+str(idOar)+"_"+flagFile+".txt", "w")
        file.write(" ".join(sys.argv)+"\n")
        for y in Ypred-Y:
            file.write(np.array_str(y, max_line_width=1000000)+"\n")



        
if __name__ == '__main__':

    l2_Model = L2Model()

    # t=[lambda x:random_rotation(x,2.0,row_index=2,col_index=3,channel_index=1),
    #    lambda x:random_shift(x,0.03,0.03,row_index=2,col_index=3,channel_index=1),
    #    lambda x:random_zoom(x,0.05,row_index=2,col_index=3,channel_index=1)]
    # t=[lambda x:random_rotation(x,2.0,row_index=1,col_index=2,channel_index=0),
    #    lambda x:random_shift(x,0.03,0.03,row_index=1,col_index=2,channel_index=0),
    #    lambda x:random_zoom(x,[0.95,1.05],row_index=1,col_index=2,channel_index=0)]

    (gen_training, N_train), (gen_val, N_val), (gen_test, N_test) = load_data_generator(ROOTPATH, train_txt, test_txt,validation=0.8,subsampling=ssRatio,batch_size=BATCH_SIZE)

    l2_Model.fit((gen_training, N_train),(gen_val, N_val))

    l2_Model.evaluate((gen_training, N_train),"training", 224)
    l2_Model.evaluate((gen_val, N_val),"validation", 224)
    l2_Model.evaluate((gen_test, N_test),"test", 224)
