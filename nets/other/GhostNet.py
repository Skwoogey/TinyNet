import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
from depthwise3DConv import DepthwiseConv3D
import numpy as np

Name = 'base_model'
window_size = 15
window_depth = -1

def getModel(input_shape, num_classes):
	def GhostModule2d(input_block, channels, ghost_ratio, use_relu=True):
		conved = keras.layers.Conv2D(
			kernel_size = (1, 1),
			filters = channels // (ghost_ratio),
			padding='valid'
		)(input_block)
		conved = keras.layers.BatchNormalization()(conved)
		if use_relu:
			conved = keras.layers.Activation('relu')(conved)
		
		dw_conved = keras.layers.DepthwiseConv2D(
			kernel_size=3,
			padding='same',
			depth_multiplier = ghost_ratio - 1
		)(conved)
		dw_conved = keras.layers.BatchNormalization()(dw_conved)
		if use_relu:
			dw_conved = keras.layers.Activation('relu')(dw_conved)
		
		res = keras.layers.Concatenate()([conved, dw_conved])
		
		return res
	
	def SE(input_block, reduction_channels, out_channels):
		res = keras.layers.GlobalAveragePooling2D(keepdims=True)(input_block)
		
		res = keras.layers.Conv2D(kernel_size=1, filters=reduction_channels)(res)
		res = keras.layers.Activation('relu')(res)
		res = keras.layers.Conv2D(kernel_size=1, filters=out_channels)(res)
		res = keras.layers.Activation('sigmoid')(res)
		
		return res
		
	def shortcut(input_block):
		conved = keras.layers.Conv2D(
			kernel_size = 3,
			filters = 16,
			padding='same'
		)(input_block)
		conved = keras.layers.BatchNormalization()(conved)
		conved = keras.layers.Activation('relu')(conved)
		
		conved = keras.layers.Conv2D(
			kernel_size=1,
			padding='valid',
			filters = 24
		)(conved)
		conved = keras.layers.BatchNormalization()(conved)
		#conved = keras.layers.Activation('relu')(conved)
		
		return conved

	# PART 1
	input_layer = keras.Input(shape = input_shape)
	net = input_layer
	
	# STEM
	net = keras.layers.Conv2D(kernel_size=3, filters=16, padding='valid')(net)
	net = keras.layers.BatchNormalization()(net)
	net = keras.layers.Activation('relu')(net)
	#net = keras.layers.Dropout(0.2)(net)
	after_stem = net
	
	# BLOCK 1
	net = GhostModule2d(net, 16, 2)
	
	net_se = SE(net, 4, 16)
	net = keras.layers.multiply([net, net_se])
	
	net = GhostModule2d(net, 16, 2, use_relu=True)
	
	net = keras.layers.Add()([net, after_stem])
	after_b1 = net

	# BLOCK 2
	net = GhostModule2d(net, 48, 2)
	
	net_se = SE(net, 12, 48)
	net = keras.layers.multiply([net, net_se])
	
	net = GhostModule2d(net, 24, 2, use_relu=True)
	
	net = keras.layers.Add()([net, shortcut(after_b1)])
	after_b2 = net
	
	# BLOCK 3
	net = GhostModule2d(net, 72, 2)
	
	net_se = SE(net, 20, 72)
	net = keras.layers.multiply([net, net_se])
	
	net = GhostModule2d(net, 24, 2, use_relu=True)
	
	net = keras.layers.Add()([net, after_b2])
	after_b3 = net
	
	# LAST CONV
	net = keras.layers.Conv2D(kernel_size=1, filters=72, padding='valid')(net)
	net = keras.layers.BatchNormalization()(net)
	net = keras.layers.Activation('relu')(net)
	net = keras.layers.GlobalAveragePooling2D()(net)
	
	net = keras.layers.Dense(216)(net)
	net = keras.layers.BatchNormalization()(net)
	net = keras.layers.Activation('relu')(net)
	#net = keras.layers.Dropout(0.2)(net)
	
	net = keras.layers.Dense(num_classes)(net)
	net = keras.layers.Activation('softmax')(net)
	
	model = keras.Model(inputs=input_layer, outputs=net)
	#optimizer = tfa.optimizers.SGD(weight_decay = 1e-5, momentum=0.9)
	optimizer = tf.keras.optimizers.SGD(momentum=0.9, nesterov=False)
	#optimizer = tf.keras.optimizers.Adam()
	loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
	model.compile(
		optimizer = optimizer,
		loss = loss,
		metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
	)

	return model, model

def prepareData(data, labels):
	return data, labels

total_epochs = 500
def scheduler(epoch):
	lr = 0.1
	print(lr)
	return lr