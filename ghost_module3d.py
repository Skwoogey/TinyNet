import tensorflow as tf
import numpy as np
from depthwise3DConv import DepthwiseConv3D

default_initializer = tf.keras.initializers.VarianceScaling(2.0)

def GhostModule3D(input, filters, kernel_size, dw_kernel_size, ratio, stride=1, padding='same', 
	data_format='channels_last',kernel_init=None, activation=None, use_bias=False, use_ReLU=True, use_BN=True,BN_momentum=0.99, BN_eps=0.001):
	if kernel_init == None:
		kernel_init = default_initializer

	conv_filters = int(np.ceil(filters / ratio))
	print("conv_filters: ", conv_filters)
	conv = tf.keras.layers.Conv3D(
		filters=conv_filters,
		kernel_size=kernel_size,
		strides=stride,
		padding=padding,
		data_format=data_format,
		activation=activation,
		use_bias=use_bias,
		kernel_initializer=kernel_init
		)(input)
	if use_BN:
		conv = tf.keras.layers.BatchNormalization(momentum=BN_momentum, epsilon=BN_eps)(conv)
	if use_ReLU:
		conv = tf.keras.layers.Activation('relu')(conv)
	print("ratio-1: ", ratio-1)
	ghost_filters = DepthwiseConv3D(
		kernel_size=dw_kernel_size,
		depth_multiplier=ratio-1,
		strides=(1, 1, 1),
		padding='same',
		data_format=data_format,
		use_bias=use_bias,
		)(conv)
	if use_BN:
		ghost_filters = tf.keras.layers.BatchNormalization(momentum=BN_momentum, epsilon=BN_eps)(ghost_filters)
	if use_ReLU:
		ghost_filters = tf.keras.layers.Activation('relu')(ghost_filters)

	return tf.keras.layers.Concatenate(axis = 4)([conv, ghost_filters])