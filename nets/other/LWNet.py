import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
from depthwise3DConv import DepthwiseConv3D
import numpy as np

Name = 'base_model'
window_size = 27
window_depth = -1

def getModel(input_shape, num_classes):
	def LWModuleStride2(input_block, channels = 16):
		print('I', input_block.shape.dims)
		left = keras.layers.AveragePooling3D(pool_size = 2, strides = 2, padding='same')(input_block)
		left = keras.layers.Conv3D(kernel_size = (1, 1, 1), padding='same', filters = channels)(left)
		left = keras.layers.BatchNormalization()(left)
		print('L', left.shape.dims)
		right = keras.layers.Conv3D(kernel_size = (1, 1, 1), filters = channels)(input_block)
		right = keras.layers.BatchNormalization()(right)
		right = keras.layers.Activation('relu')(right)
		right = DepthwiseConv3D(kernel_size = (3, 3, 3), strides=(2, 2, 2), padding='same', depth_multiplier=4)(right)
		right = keras.layers.BatchNormalization()(right)
		right = keras.layers.Activation('relu')(right)
		right = keras.layers.Conv3D(kernel_size = (1, 1, 1), filters = channels)(right)
		right = keras.layers.BatchNormalization()(right)
		print('R', right.shape.dims)

		res = keras.layers.Add()([left, right])
		res = keras.layers.Activation('relu')(res)

		return res

	def LWModule(input_block, channels = 16):
		left = input_block

		right = keras.layers.Conv3D(kernel_size = (1, 1, 1), filters = channels)(input_block)
		right = keras.layers.BatchNormalization()(right)
		right = keras.layers.Activation('relu')(right)
		right = DepthwiseConv3D(kernel_size = (3, 3, 3), padding='same', depth_multiplier=4)(right)
		#right = keras.layers.Conv3D(kernel_size = (3, 3, 3), padding='same', filters = channels*4, use_bias=False)(right)
		right = keras.layers.BatchNormalization()(right)
		right = keras.layers.Activation('relu')(right)
		right = keras.layers.Conv3D(kernel_size = (1, 1, 1), filters = channels)(right)
		right = keras.layers.BatchNormalization()(right)

		res = keras.layers.Add()([left, right])
		res = keras.layers.Activation('relu')(res)

		return res

	# PART 1
	input_layer = keras.Input(shape = input_shape)
	net = input_layer
	net = keras.layers.Conv3D(kernel_size = (3, 3, 8), padding='valid', filters = 32)(net)
	net = keras.layers.BatchNormalization()(net)
	net = keras.layers.Activation('relu')(net)
	net = keras.layers.MaxPool3D(pool_size=(3, 3, 3), strides=2, padding='same')(net)

	# PART 2
	net = LWModule(net, 32)

	net = LWModuleStride2(net, 64)
	net = LWModule(net, 64)

	net = LWModuleStride2(net, 128)
	net = LWModule(net, 128)

	net = LWModuleStride2(net, 256)

	# PART 3
	avg_pool_size = (net.shape.dims[1], net.shape.dims[2], net.shape.dims[3])
	net = keras.layers.AveragePooling3D(pool_size = avg_pool_size, strides = 1, padding='valid')(net)
	net = keras.layers.Flatten()(net)
	net = keras.layers.Dense(256)(net)
	net = keras.layers.Dense(num_classes)(net)
	#net = keras.layers.Activation('softmax')(net)
	model = keras.Model(inputs=input_layer, outputs=net)
	optimizer = tfa.optimizers.SGDW(weight_decay = 1e-4, momentum=0.9)
	#optimizer = tf.keras.optimizers.SGD(momentum=0.9, nesterov=True)
	#optimizer = tf.keras.optimizers.Adam()
	loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
	model.compile(
		optimizer = optimizer,
		loss = loss,
		metrics = [tf.keras.metrics.CategoricalAccuracy()]
	)

	return model, model

def prepareData(data, labels, num_classes):
	#data_one_hot = np.eye(num_classes)[labels]
	inputs = [data]
	outputs = [labels]
	#print(outputs)
	#print(np.unique(labels))
	#return inputs, outputs
	return data, labels

total_epochs = 60
def scheduler(epoch):
	lr = 0.01
	if epoch >= 49:
		lr = 0.001
	print(lr)
	return lr