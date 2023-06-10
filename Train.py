import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from keras import backend as K
   
from keras.applications.vgg16 import preprocess_input
#from keras.preprocessing.image import load_img
from keras.models import load_model

from keras import preprocessing
from keras import backend as K
from keras import models

from config1 import *
from model import ProposedModel,getAssembledModel

#Try turning this off for intel gpu env
tf.config.run_functions_eagerly(True)

config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))

config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


#do def get HEaat maps

def getPatches(folder, isTraining, p):
    patches = []
    i_i =[]
    i_j = []
    mean = 0
    var =10
    sigma = var ** 0.5
    act_size = 1000
    gaussian = np.random.normal(mean, sigma,(act_size,act_size))

    doChunking = True

    index = 0
    i2 = 1
    for filename in os.listdir(folder):
        if isTraining == True:
            print(str(i2) + "chunking training image" + filename)
        else:
            print(str(i2) + "chunking testing image" + filename)
        
        image = Image.open(folder+filename)
        image = image.resize((1000,1000))
        data = np.array(image)

        if isTraining == True:
            #add gaussian noise
            if len(data.shape) == 2:
                data = data + gaussian
            else:
                data[:,:,0] = data[:,:,0]+ gaussian
                data[:,:,1] = data[:,:,1]+ gaussian
                data[:,:,2] = data[:,:,2]+ gaussian
        
        data = data.astype('float32')/255
        row, col, ch = data.shape

        for i in range(row):
            for j in range(col):
                if (i+1)*p <= row and (j+1)*p <= col:
                    patch = data[(i)*p:(i+1)*p,(j)*p:(j+1)*p,:]
                    patches.append(patch)
                    i_i.append(i)
                    i_j.append(j)
          
        if doChunking == True:
            if index >= 10:
                break
            else:
                index = index + 1
        
    patches = np.array(patches)

    return i_i,i_j, patches

#Do def transfer weights

autoencoder = getAssembledModel(p)

#ONE TIME TRAINING
if doTraining == True:
    _,_,x_train = getPatches(tr_folder,True,p)
    _,_,x_valid = getPatches(te_folder,False,p)

    print(x_train.shape)
    print(x_valid.shape)

    autoencoder.fit(x_train, x_train,
                    epochs=20,
                    batch_size=16,
                    shuffle=True,
                    validation_data=(x_valid, x_valid))
    autoencoder.save("model.tf")