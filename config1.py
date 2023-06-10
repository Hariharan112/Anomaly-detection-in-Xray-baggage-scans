import tensorflow as tf
import os
import matplotlib.pyplot as plt
import tensorflow.keras
from PIL import Image
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img

from tensorflow.keras import preprocessing
from tensorflow.keras import models
from model import ProposedModel, getAssembledModel

numDatasets = 5

tr_folder = "datasets/sixray/normal/"

te_folder = "datasets/sixray/abnormal/"

res_fake = "datasets/sixray/results/fake"

res_real = "datasets/sixray/results/real/"

res_disp = "datasets/sixray/results/disp/"

x_train =[]
x_test =[]
x_valid =[]

i_i = []
i_j =[]

doTraining =True
act_size = 1000
p = 100
