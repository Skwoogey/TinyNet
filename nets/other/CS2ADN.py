import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
from CenterLossLayer import CenterLossLayer
import numpy as np

class MultiplySpectralSpatialAttention(tf.keras.layers.Layer):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def build(self, input_shape):
		self.spat_input = input_shape[0]
		self.spec_input = input_shape[1]
		if len(self.spat_input) != 5 or len(self.spec_input) != 5:
			raise ValueError('wrong shapes: only shapes of 5 are accepted')

		if self.spat_input[1] != 1 and self.spat_input[2] != 1:
			raise ValueError('incorrect spatial input', self.spat_input)	

		if self.spec_input[3] != 1:
			raise ValueError('incorrect spectral input', self.spec_input)	

		if self.spat_input[4] != self.spec_input[4]:
			raise ValueError('missmatch in channels count')	
 
	def call(self, x):
		spat = x[0]
		spec = x[1]

		spat = K.repeat_elements(spat, self.spec_input[1], axis=1)
		spat = K.repeat_elements(spat, self.spec_input[2], axis=2)

		spec = K.repeat_elements(spec, self.spat_input[3], axis=3)

		mul = keras.layers.Multiply()([spat, spec])
		'''
		spat = K.concatenate([spat] * self.spec_input[1], axis = 1)
		spat = K.concatenate([spat] * self.spec_input[2], axis = 2)

		spec = K.concatenate([spec] * self.spat_input[3], axis = 3)

		mul = keras.layers.Multiply()([spat, spec])
		'''
		return mul


	def call_old(self, x):
		spat = x[0]
		spec = x[1]

		spat = tf.split(spat, self.spat_input[3], axis = 3)

		mul = []
		for tensor in spat:
			mul.append(tf.math.multiply(spec, tensor))

		mul = K.concatenate(mul, axis = 3)

		'''
		spat = K.concatenate([spat] * self.spec_input[1], axis = 1)
		spat = K.concatenate([spat] * self.spec_input[2], axis = 2)

		spec = K.concatenate([spec] * self.spat_input[3], axis = 3)

		mul = keras.layers.Multiply()([spat, spec])
		'''
		return mul


		

Name = 'base_model'
window_size = 9
window_depth = 103

def getModel(input_shape, num_classes):
	def AttentionModule(input_block):
		ib_depth = input_block.shape[3]
		ib_width = input_block.shape[1]
		spec_br = keras.layers.MaxPool3D(pool_size = (1, 1, ib_depth))(input_block)
		# spec_br shape (X, X, 1, 16)
		spec_br = keras.layers.Conv3D(kernel_size = (3, 3, 1), padding='same', filters = 16, activation='softmax')(spec_br)
		# spec_br shape (X, X, 1, 16)

		spat_br = keras.layers.AveragePooling3D(pool_size = (ib_width, ib_width, 1))(input_block)
		# spat_br shape (1, 1, Y, 16)
		spat_br = keras.layers.Conv3D(kernel_size = (1, 1, 2), padding='same', filters = 16, activation='softmax')(spat_br)
		# spat_br shape (1, 1, Y, 16)

		spec_spat_attention = MultiplySpectralSpatialAttention()([spat_br, spec_br])
		# spec_spat shape (X, X, Y, 16)
		res = keras.layers.Multiply()([spec_spat_attention, input_block])
		print(res)
		# res shape (X, X, Y, 16)
		return res
		
	def DenseBlock(input_block):
		attention_module1 = AttentionModule(input_block)
		
		res = keras.layers.Concatenate(axis = 4)([input_block, attention_module1])
		attention_module2 = keras.layers.Conv3D(kernel_size = (3, 3, 3), padding='same', filters = 16)(res)
		attention_module2 = keras.layers.BatchNormalization()(attention_module2)
		attention_module2 = keras.layers.Activation('relu')(attention_module2)

		attention_module2 = AttentionModule(attention_module2)

		res = keras.layers.Concatenate(axis = 4)([res, attention_module2])
		
		res = keras.layers.Conv3D(kernel_size = (3, 3, 3), padding='same', filters = 16)(res)
		res = keras.layers.BatchNormalization()(res)
		res = keras.layers.Activation('relu')(res)

		return res

	def zero_loss(y_true, y_pred):
		#print(y_pred.shape)
		return 0.5 * tf.keras.backend.sum(y_pred, axis=0)

	def empty_loss(y_true, y_pred):
		#print(y_pred.shape)
		return 0.0

	input_layer = keras.Input(shape = (9, 9, 103, 1))
	input_labels_layer = keras.Input(shape = (9,))
	# shape (9, 9, 103, 1)
	net = input_layer
	net = keras.layers.Conv3D(kernel_size = (3, 3, 3), padding='same', filters = 16)(net)
	net = keras.layers.BatchNormalization()(net)
	net = keras.layers.Activation('relu')(net)
	# shape (9, 9, 103, 16)
	net = DenseBlock(net)
	# shape (9, 9, 103, 16)
	net = keras.layers.Conv3D(kernel_size = (1, 1, 53), padding='valid', filters = 16)(net)
	net = keras.layers.BatchNormalization()(net)
	net = keras.layers.Activation('relu')(net)
	# shape (9, 9, 51, 16)
	net = keras.layers.AveragePooling3D(pool_size = (9, 9, 1), strides = (2, 2, 1), padding='same')(net)
	# shape (5, 5, 51, 16)
	net = DenseBlock(net)
	net = keras.layers.AveragePooling3D(pool_size = (5, 5, 1))(net)
	net = keras.layers.Flatten()(net)
	dense_vector = net
	net = keras.layers.Dropout(0.5)(net)
	result = keras.layers.Dense(9, activation='softmax', name='d')(net)
	cll = CenterLossLayer(9, name = 'cll')([dense_vector, input_labels_layer])

	model = keras.Model(inputs=[input_layer, input_labels_layer], outputs=[result, cll])
	optimizer = tfa.optimizers.SGDW(weight_decay = 1e-5, momentum=0.9, nesterov=True)
	#optimizer = tf.keras.optimizers.SGD(momentum=0.9, nesterov=True)
	#optimizer = tf.keras.optimizers.Adam()
	loss = tf.keras.losses.CategoricalCrossentropy()
	model.compile(
		optimizer = optimizer,
		loss = [loss, zero_loss],
		loss_weights = [1, 0.0],
		metrics = [tf.keras.metrics.CategoricalAccuracy()]
	)
	model_stripped = keras.Model(inputs=[input_layer], outputs=[result])
	model_stripped.compile(
		loss = empty_loss,
		metrics = [tf.keras.metrics.CategoricalAccuracy()]
	)

	return model, model_stripped

def prepareData(data, labels):
	data_one_hot = np.zeros(shape = (data.shape[0], 9))
	data_one_hot[np.arange(data.shape[0]), labels.reshape((labels.shape[0],))] = 1.0
	data_zeros = np.zeros(shape = (data.shape[0], 1))
	inputs = [data, data_one_hot]
	outputs = [data_one_hot, data_zeros]
	return inputs, outputs

total_epochs = 80
def scheduler(epoch):
	lr = 0.006
	print(lr)
	return lr