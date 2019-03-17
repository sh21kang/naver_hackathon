from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Dense, TimeDistributed, Input
from keras.models import Model
from keras.layers.core import Dense, Activation
from keras.preprocessing import image
import keras.backend as K
from densenet121 import DenseNet
#from vgg16 import VGG16
from RoiPooling import RoiPooling
from get_regions import rmac_regions, get_size_vgg_feat_map
import nsml
import scipy.io
import numpy as np
import utils
import tensorflow as tf

def addition(x):
    sum = K.sum(x, axis=1)
    return sum


def weighting(input):
    x = input[0]
    w = input[1]
    w = K.repeat_elements(w, 512, axis=-1)
    out = x * w
    return out


def rmac(input_shape,num_rois,Densenet,in_roi):
      # Load sample image


    pooling_regions = [1]
    # Load VGG16
    

    # Regions as input
    #in_roi = Input(shape=(num_rois, 4), name='input_roi')(in_roi_ext)
    #in_roi=tf.constant(in_roi_ext)
    print("IN RMAC type(in_roi): ",type(in_roi))
    # ROI pooling
    x = RoiPooling(pooling_regions, num_rois, in_roi)([Densenet.layers[-4].output])

    # Normalization
    x = Lambda(lambda x: K.l2_normalize(x, axis=2), name='norm1')(x)

    # PCA
    x = TimeDistributed(Dense(512, name='pca',
                              kernel_initializer='identity',
                              bias_initializer='zeros'))(x)

    # Normalization
    x = Lambda(lambda x: K.l2_normalize(x, axis=2), name='pca_norm')(x)

    # Addition
    rmac = Lambda(addition, output_shape=(512,), name='rmac')(x)

    # # Normalization
    rmac_norm = Lambda(lambda x: K.l2_normalize(x, axis=1),output_shape=(512,), name='rmac_norm')(rmac)

    x = Dense(1383, name='fc6')(rmac_norm)
    x = Activation('softmax', name='prob')(x)

    # Define model
    model = Model(Densenet.input, x)

    # Load PCA weights
    mat = scipy.io.loadmat(utils.DATA_DIR + utils.PCA_FILE)
    b = np.squeeze(mat['bias'], axis=1)
    w = np.transpose(mat['weights'])
    #model.layers[-4].set_weights([w, b])
    
    return model
