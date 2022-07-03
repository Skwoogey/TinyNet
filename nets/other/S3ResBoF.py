from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from tensorflow.keras import backend as K
from tensorflow.keras.layers  import Layer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Reshape

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    Dropout
)
from tensorflow.keras.layers import (
    Convolution3D,
    MaxPooling3D,
    AveragePooling3D,
    Conv3D
)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling3D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import GlobalMaxPooling3D

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import add
from tensorflow.keras.layers import multiply
from tensorflow.keras.regularizers import l2
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD, Adadelta, RMSprop, Nadam

from contextlib import redirect_stdout
import tensorflow as tf


class BoF_Pooling(Layer):
    """
    Implements the CBoF pooling
    """

    def __init__(self, n_codewords, spatial_level=0 ,**kwargs):
        """
        Initializes a BoF Pooling layer
        :param n_codewords: the number of the codewords to be used
        :param spatial_level: 0 -> no spatial pooling, 1 -> spatial pooling at level 1 (4 regions). Note that the
         codebook is shared between the different spatial regions
        :param kwargs:
        """
        self.N_k = n_codewords
        self.spatial_level = spatial_level
        self.V, self.sigmas = None, None
        super(BoF_Pooling, self).__init__(**kwargs)

    def build(self,input_shape):
        k = []
        for i in input_shape:
            if( i == None):
                k.append(i)
            else:
                k.append(int(i))
        input_shape = tuple(k)
        print(input_shape)
        self.V = self.add_weight(name='codebook', shape=(1, 1, input_shape[3], self.N_k), initializer='uniform',trainable=True)
        self.sigmas = self.add_weight(name='sigmas', shape=(1, 1, 1, self.N_k), initializer=Constant(0.1),
                                      trainable=True)
        super(BoF_Pooling, self).build(input_shape)

    def call(self, x):
        print("call")

        # Calculate the pairwise distances between the codewords and the feature vectors
        x_square = K.sum(x ** 2, axis=3, keepdims=True)
        y_square = K.sum(self.V ** 2, axis=2, keepdims=True)
        #print(K.conv2d(self.V,x, strides=(1, 1), padding='valid').shape)
        dists = x_square + y_square - 2 * K.conv2d(x,self.V , strides=(1, 1), padding='valid')
        dists = K.maximum(dists, 0)
        # Quantize the feature vectors
        quantized_features = K.softmax(- dists / (self.sigmas ** 2))

        # Compile the histogram
        if self.spatial_level == 0:
            histogram = K.mean(quantized_features, [1, 2])
        elif self.spatial_level == 1:
            shape = K.shape(quantized_features)
            mid_1 = K.cast(shape[1] / 2, 'int32')
            mid_2 = K.cast(shape[2] / 2, 'int32')
            histogram1 = K.mean(quantized_features[:, :mid_1, :mid_2, :], [1, 2])
            histogram2 = K.mean(quantized_features[:, mid_1:, :mid_2, :], [1, 2])
            histogram3 = K.mean(quantized_features[:, :mid_1, mid_2:, :], [1, 2])
            histogram4 = K.mean(quantized_features[:, mid_1:, mid_2:, :], [1, 2])
            histogram = K.stack([histogram1, histogram2, histogram3, histogram4], 1)
            histogram = K.reshape(histogram, (-1, 4 * self.N_k))
        else:
            # No other spatial level is currently supported (it is trivial to extend the code)
            assert False

        # Simple trick to avoid rescaling issues
        return histogram * self.N_k

    def compute_output_shape(self, input_shape):
        print("output")
        k = []
        for i in input_shape:
            if( i == None):
                k.append(i)
            else:
                k.append(int(i))
        input_shape = tuple(k)
        print(input_shape)
        if self.spatial_level == 0:
            return (input_shape[0], self.N_k)
        elif self.spatial_level == 1:
            return (input_shape[0], 4 * self.N_k)
            

def _handle_dim_ordering():
    global CONV_DIM1
    global CONV_DIM2
    global CONV_DIM3
    global CHANNEL_AXIS
    if K.image_data_format() == 'channels_last':
        CONV_DIM1 = 1
        CONV_DIM2 = 2
        CONV_DIM3 = 3
        CHANNEL_AXIS = 4
    else:
        CHANNEL_AXIS = 1
        CONV_DIM1 = 2
        CONV_DIM2 = 3
        CONV_DIM3 = 4

def _bn_relu(input):
	norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
	return Activation("relu")(norm)

def _conv_bn_relu(**params):
    nb_filter = params["nb_filter"]
    kernel_dim1 = params["kernel_dim1"]
    kernel_dim2 = params["kernel_dim2"]
    kernel_dim3 = params["kernel_dim3"]
    subsample = params.setdefault("subsample", (1, 1, 1))
    init = params.setdefault("init", "he_normal")
    border_mode = params.setdefault("border_mode", "same")
    W_regularizer = params.setdefault("W_regularizer", regularizers.l2(1.e-4))
    def f(input):
        conv = Conv3D(kernel_initializer=init,strides=subsample,kernel_regularizer= W_regularizer, filters=nb_filter, kernel_size=(kernel_dim1,kernel_dim2,kernel_dim3))(input)
        return _bn_relu(conv)

    return f

def _bn_relu_conv(**params):
    nb_filter = params["nb_filter"]
    kernel_dim1 = params["kernel_dim1"]
    kernel_dim2 = params["kernel_dim2"]
    kernel_dim3 = params["kernel_dim3"]
    subsample = params.setdefault("subsample", (1,1,1))
    init = params.setdefault("init", "he_normal")
    border_mode = params.setdefault("border_mode", "same")
    W_regularizer = params.setdefault("W_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv3D(kernel_initializer=init, strides=subsample, kernel_regularizer=W_regularizer,
                          filters=nb_filter, kernel_size=(kernel_dim1, kernel_dim2, kernel_dim3), padding=border_mode)(activation)

    return f

def _shortcut(input, residual):
    stride_dim1 = (input.shape[CONV_DIM1]+1) // residual.shape[CONV_DIM1]
    stride_dim2 = (input.shape[CONV_DIM2]+1) // residual.shape[CONV_DIM2]
    stride_dim3 = (input.shape[CONV_DIM3]+1) // residual.shape[CONV_DIM3]
    equal_channels = residual.shape[CHANNEL_AXIS] == input.shape[CHANNEL_AXIS]

    shortcut = input
    print("input shape:", input.shape)

    shortcut = Conv3D(kernel_initializer="he_normal", strides=(stride_dim1, stride_dim2, stride_dim3), kernel_regularizer=regularizers.l2(0.0001),filters=residual.shape[CHANNEL_AXIS], kernel_size=(1, 1, 1), padding='valid')(input)
    shortcut = squeeze_excite_block(shortcut)
    return add([shortcut, residual])
	
def _shortcut_spc(input, residual):
    stride_dim1 = (input.shape[CONV_DIM1]+1) // residual.shape[CONV_DIM1]
    stride_dim2 = (input.shape[CONV_DIM2]+1) // residual.shape[CONV_DIM2]
    stride_dim3 = (input.shape[CONV_DIM3]+1) // residual.shape[CONV_DIM3]
    equal_channels = residual.shape[CHANNEL_AXIS] == input.shape[CHANNEL_AXIS]

    shortcut = input
    print("input shape:", input.shape)
    shortcut = squeeze_excite_block(residual)
    return add([shortcut, residual])
	
	
def basic_block(nb_filter, init_subsample=(1, 1, 1), is_first_block_of_first_layer=False):
    def f(input):

        if is_first_block_of_first_layer:
            conv1 = Conv3D(kernel_initializer="he_normal", strides=init_subsample, kernel_regularizer=regularizers.l2(0.0001),
                          filters=nb_filter, kernel_size=(3, 3, 1), padding='same')(input)
        else:
            conv1 = _bn_relu_conv(nb_filter=nb_filter, kernel_dim1=3, kernel_dim2=3, kernel_dim3=1, subsample=init_subsample)(input)

        residual = _bn_relu_conv(nb_filter=nb_filter, kernel_dim1=3, kernel_dim2=3, kernel_dim3=1)(conv1)
        return _shortcut(input, residual)

    return f

def basic_block_spc(nb_filter, init_subsample=(1, 1, 1), is_first_block_of_first_layer=False):
    def f(input):

        if is_first_block_of_first_layer:
            conv1 = Conv3D(kernel_initializer="he_normal", strides=init_subsample, kernel_regularizer=regularizers.l2(0.0001),
                          filters=nb_filter, kernel_size=(3, 3, 1), padding='same')(input)
        else:
            conv1 = _bn_relu_conv(nb_filter=nb_filter, kernel_dim1=3, kernel_dim2=3, kernel_dim3=1, subsample=init_subsample)(input)

        residual = _bn_relu_conv(nb_filter=nb_filter, kernel_dim1=3, kernel_dim2=3, kernel_dim3=1)(conv1)
        return _shortcut_spc(input, residual)

    return f

def squeeze_excite_block(input, ratio=16):
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    #se = GlobalAveragePooling3D()(init)
    se = GlobalMaxPooling3D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x
	
def SE_ResNet8(input_shape, num_outputs, codebooks):
	_handle_dim_ordering()
	if len(input_shape) != 4:
		raise Exception("Input shape should be a tuple (nb_channels, kernel_dim1, kernel_dim2, kernel_dim3)")
	
	if K.image_data_format() == 'channels_last':
		input_shape = (input_shape[1], input_shape[2],input_shape[3], input_shape[0])
	input = Input(shape = input_shape)
	
	conv1 = _conv_bn_relu(nb_filter=64, kernel_dim1=1, kernel_dim2=1, kernel_dim3=7, subsample=(1, 1, 2))(input)
	
	conv2 = basic_block_spc(64, is_first_block_of_first_layer = True)(conv1)
	bn3 = _bn_relu(conv2)
	#bn3 = _bn_relu(bn3_1)
	bn4 = _conv_bn_relu(nb_filter=64, kernel_dim1=1, kernel_dim2=1, kernel_dim3=bn3.shape[CONV_DIM3], subsample=(1, 1, 2) , border_mode='valid')(bn3)
	resh = Reshape((bn4.shape[CONV_DIM1],bn4.shape[CONV_DIM2],bn4.shape[CHANNEL_AXIS],1))(bn4)
	conv4 = _conv_bn_relu(nb_filter=64, kernel_dim1=3, kernel_dim2=3, kernel_dim3=64,
                              subsample=(1, 1, 1))(resh)
	conv5 = basic_block(64, is_first_block_of_first_layer = True)(conv4)
	bn5 = _bn_relu(conv5)
	bn6 = _bn_relu(bn5)

	#pool2 = AveragePooling3D(pool_size=(bn6.shape[CONV_DIM1],
                                            #bn6.shape[CONV_DIM2],
                                            #bn6.shape[CONV_DIM3],),strides=(1, 1, 1))(bn6)
	
	#flatten1 = Flatten()(bn6)
	print(bn6.shape)
	bn6 = Reshape((bn6.shape[1],bn6.shape[2],bn6.shape[4]))(bn6)
	print(bn6.shape)
	
	#bn6 = K.squeeze(bn6,axis = 3)
	flatten1 = BoF_Pooling(codebooks, spatial_level=0)(bn6)
	drop1 = Dropout(0.5)(flatten1)
	dense = Dense(units=num_outputs, activation="softmax", kernel_initializer="he_normal")(drop1)	
	model = Model( inputs = input , outputs = dense)
	model.summary()	
	return model

Name = 'base_model'
window_size = 15
window_depth = 15

learning_rate=0.0003
def getModel(input_shape, num_classes):
    input_shape_channels_first = (input_shape[3], input_shape[0], input_shape[1], input_shape[2])
    model  = SE_ResNet8(input_shape_channels_first, num_classes,64)

    RMS = RMSprop(learning_rate=learning_rate)
    # Let's train the model using RMSprop
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), 
            optimizer=RMS, 
            metrics = [tf.keras.metrics.CategoricalAccuracy()]
    )

    return model, model

def prepareData(data, labels, num_classes):
    #data_one_hot = np.eye(num_classes)[labels]
    inputs = [data]
    outputs = [labels]
    #print(outputs)
    #print(np.unique(labels))
    return data, labels

total_epochs = 200
def scheduler(epoch):

    return learning_rate