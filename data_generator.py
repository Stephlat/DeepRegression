''' Create generators from dataset '''

import numpy as np
import cv2
import random

HIGH_DIM = 512
GLLIM_K = 1


BATCH_SIZE = 128

# Mode for the validation set for our mixture model

def load_data_generator_List(rootpath, imIn, file_test, validation=1.0,subsampling=1.0,processingTarget=None,transform=[],outSize=(224,224),batch_size=BATCH_SIZE,shuffle=False):
    ''' create generators from data'''

    
    def generator(rootpath, images):
        
        N=len(images)
        nbatches=N/batch_size+1
        if N%batch_size==0:
            nbatches-=1
        if shuffle:
            random.shuffle(images)

        i=0
        while 1:
            X, Y = get_xy_from_file(rootpath, images[i*batch_size:(i+1)*batch_size],processingTarget=processingTarget,transform=transform,outSize=outSize)
            yield(X, Y)
            i=i+1
            if i>=nbatches:  # we shuffle the data when the end of the dataset is reached
                i=0
                random.shuffle(images)

    imTest = open(rootpath+file_test, 'r').readlines()
    gen_test = generator(rootpath, imTest)
    test_size=len(imTest)





    # we subsample the data if needed
    if subsampling!=1.0:
        im=imIn[0:int(subsampling*len(imIn))][:]
    else:
        im=imIn[:]
        
    if validation!=1.0:  # if we use a validation set
        Ntot=len(im)
        training_size = int(validation*len(im))
        val_size = Ntot-training_size
        
        gen_train = generator(rootpath, im[:training_size])
        gen_val = generator(rootpath, im[training_size:])

        return (gen_train,training_size),(gen_val,val_size), (gen_test,test_size)
    else:  # without validation set
        gen_train = generator(rootpath, im)
        training_size = len(im)
     
        return (gen_train,training_size), (gen_test,test_size)

def load_data_generator(rootpath, file_train, file_test, validation=1.0,subsampling=1.0,processingTarget=None,transform=[],outSize=(224,224),batch_size=BATCH_SIZE,shuffle=False):
    im = open(rootpath+file_train, 'r').readlines()
    return load_data_generator_List(rootpath, im[:], file_test, validation,subsampling,processingTarget=processingTarget,transform=transform,outSize=outSize,batch_size=batch_size,shuffle=shuffle)

def load_data_generator_List_simple(rootpath, imIn,transform=[],outSize=(224,224),batch_size=BATCH_SIZE,processingTarget=None,sample_weights=None):
    ''' create generators from data'''

    
    def generator(rootpath, images):
        
        N=len(images)
        nbatches=N/batch_size+1
        if N%batch_size==0:
            nbatches-=1
        i=0
        if sample_weights is not None:
            rn= sample_weights[:]
        while 1:

            X, Y = get_xy_from_file(rootpath, images[i*batch_size:(i+1)*batch_size],processingTarget=processingTarget,transform=transform,outSize=outSize)
            if sample_weights is None:
                yield(X, Y)
            else:
                yield(X, Y,rn[i*batch_size:(i+1)*batch_size])
            i=i+1
            if i>=nbatches:  # we shuffle the data when the end of the dataset is reached
                i=0
                if sample_weights is None:
                    random.shuffle(images)
                else:
                    c = zip(images,rn)
                    np.random.shuffle(c)
                    images = np.asarray([e[0] for e in c])
                    rn = np.asarray([e[1] for e in c])

                    

    gen = generator(rootpath, imIn[:])
    size=len(imIn)

    return (gen,size)

def load_data_generator_simple(rootpath, fileName, transform=[],outSize=(224,224),batch_size=BATCH_SIZE,processingTarget=None):
    im = open(rootpath+fileName, 'r').readlines()
    return load_data_generator_List_simple(rootpath, im[:],transform=transform,outSize=outSize,batch_size=batch_size,processingTarget=processingTarget)



def applyTransform(x,transform):
    for t in transform:
        x=t(x)
    return x

    
def get_xy_from_file(rootpath, images, processingTarget=None,transform=[],outSize=(224,224),batch_size=BATCH_SIZE):
    '''Extract data arrays from text file'''
    
    X = np.zeros((len(images),3, outSize[0], outSize[1]), dtype=np.float32)
    Y=[]

    
    for i,image in enumerate(images):
        currentline=image.strip().split(" ")
        
        imFile=currentline[0]
        
        X[i]=get_image_for_vgg(rootpath+imFile,transform,outSize)
            
        Y.append(np.asarray(map(lambda x: float(x),currentline[1:])))


    if processingTarget:
        Y=processingTarget(Y)

    Y=np.squeeze(np.asarray(Y)).reshape((X.shape[0],len(Y[0])))
    return (X,Y)

def get_image_for_vgg(imName,transform=[],outSize=(224,224),batch_size=BATCH_SIZE):
    '''Preprocess images as VGG inputs'''
    im = (cv2.resize(cv2.imread(imName), (outSize[1],outSize[0]))).astype(np.float32)


    # we substract the mean value of imagenet
    if outSize==(224,224):
        im[:,:,0] -= 103.939
        im[:,:,1] -= 116.779
        im[:,:,2] -= 123.68
    im = im.transpose(2,0,1)
    
   
    if transform:
        im=applyTransform(im,transform)

    im = np.expand_dims(im, axis=0)
    
    return im
