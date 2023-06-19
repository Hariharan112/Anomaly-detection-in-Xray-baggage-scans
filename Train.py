import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from keras import backend as K
   
#from keras.applications.vgg16 import preprocess_input
#from keras.preprocessing.image import load_img
#from keras.models import load_model

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
    act_size = 1040
    gaussian = np.random.normal(mean, sigma,(act_size,act_size))

    doChunking = False

    index = 0
    i2 = 1
    for filename in os.listdir(folder):
        if isTraining == True:
            print(str(i2) + "chunking training image" + filename)
        else:
            print(str(i2) + "chunking testing image" + filename)
        
        image = Image.open(folder+filename)
        image = image.resize((1040,1040))
        data = np.array(image)

        #add gaussian noise
        if isTraining == True:
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

autoencoder = getAssembledModel(p)

def transferWeights(model1, model2):    
    for i in range(1,len(model1.layers)):
        model2.layers[i].set_weights(model1.layers[i].get_weights())
    
    return model2


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
else:
    model1 = tf.keras.models.load_model("model.tf", compile=False)
    transferWeights(model1, autoencoder)

#Testing our images
i_i,i_j, x_test = getPatches(te_folder,False,p)
print(x_test.shape)
print("**********************Reconstructing Patches*******************")

decoded_imgs = []
import cv2 as cv

l1,r1,c1,ch1 = x_test.shape
img = np.zeros((act_size,act_size,3), dtype='float32')
for i in range(l1):
    decoded_imgs.append(autoencoder.predict(x_test[i].reshape(1,p,p,3)))

decoded_imgs = np.array(decoded_imgs)

print("**********************Stitching Images*******************")

for k in range(len(i_i)):
    patch = decoded_imgs[k].reshape(p, p, 3)
    cv.imwrite("datasets/"+ str(i_i) + str(i_j)+"patch.jpg",patch)
    i = i_i[k]
    j = i_j[k]
    img[(i)*p:(i+1)*p,(j)*p:(j+1)*p,:] = patch

def comp_disparity(input,decoded):
    diff = input - decoded
    return diff
inp1 = Image.open("datasets/sixray/abnormal/P00001.jpg")
inp2 = Image.open("datasets/sixray/abnormal/P00048.jpg")
inp1 = inp1.resize((1040,1040))
inp2 = inp2.resize((1040,1040))

tet = comp_disparity(inp1,img)
tet1 = comp_disparity(inp2,img)

