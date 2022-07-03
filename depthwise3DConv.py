import tensorflow as tf
import numpy as np

keras_to_tf_names = {
	'channels_last' 	: 'NDHWC',
	'channels_first' 	: 'NCDHW',
	'same' 				: 'SAME',
	'valid'				: 'VALID'
}

class DepthwiseConv3D(tf.keras.layers.Layer):
	def __init__(self,
		kernel_size,
		strides=(1, 1, 1),
		padding='valid',
		depth_multiplier=1,
		data_format='channels_last' ,
		use_bias=True
		):
		super(DepthwiseConv3D, self).__init__()

		self.kernel_size = [*kernel_size]
		self.strides = [*strides]
		self.padding = keras_to_tf_names[padding]
		self.depth_multiplier = depth_multiplier
		self.data_format = keras_to_tf_names[data_format]
		self.use_bias = use_bias

	def build(self, input_shape):
		self.shape = input_shape
		self.num_of_channels = input_shape[-1] if self.data_format == 'NDHWC' else input_shape[1]
		self.channel_axis = -1 if self.data_format == 'NDHWC' else 1
		
		self.build_net()


	def call(self, inputs):	
		channels = tf.split(inputs, inputs.shape[self.channel_axis], axis=self.channel_axis)
		
		convolved_channels =  [None] * self.num_of_channels

		for c in range(self.num_of_channels):
			convolved_channels[c] = tf.nn.convolution(
				input = channels[c],
				filters = self.filters[c],
				padding = self.padding,
				strides = list([1, *self.strides, 1]),
				data_format = self.data_format
			)

		concatenated = tf.concat(convolved_channels, axis=4)
		if self.use_bias:
			concatenated = tf.nn.bias_add(concatenated, self.bias, self.data_format)
		return concatenated

	def get_config(self):
		config = {
			'kernel_size': self.kernel_size,
			'strides': self.strides,
			'padding': self.padding,
			'depth_multiplier': self.depth_multiplier,
			'data_format': self.data_format,
			'use_bias': self.use_bias,
			'input_shape': self.shape
		}

		return config

	def from_config(config):
		#print(config)
		inst = DepthwiseConv3D(config['kernel_size'])
		inst.strides = config['strides']
		inst.padding = config['padding']
		inst.depth_multiplier = config['depth_multiplier']
		inst.data_format = config['data_format']
		inst.use_bias = config['use_bias']

		return inst

	def build_net(self):
		self.num_of_channels = self.shape[-1] if self.data_format == 'NDHWC' else self.shape[1]
		self.channel_axis = -1 if self.data_format == 'NDHWC' else 1

		self.filters = []
		
		if self.use_bias:
			'''
			zp = None

			if self.padding == 'SAME':
				zp = [(ks_dim - 1) // 2 for ks_dim in self.kernel_size]
			else:
				zp = [0, 0, 0]

			bias_shape = [(self.shape[1] - self.kernel_size[0] + 2 * zp[0]) // self.strides[0] + 1,
							(self.shape[2] - self.kernel_size[1] + 2 * zp[1]) // self.strides[1] + 1,
							(self.shape[3] - self.kernel_size[2] + 2 * zp[2]) // self.strides[2] + 1,
							self.depth_multiplier * self.num_of_channels]
			#print('Bias shape:', bias_shape)
			self.bias = self.add_weight(shape=bias_shape,
						 		initializer='random_normal',
								trainable=True,
								dtype = np.float32,
								name = 'bias')
			'''
			self.bias = self.add_weight(shape = [self.num_of_channels * self.depth_multiplier],
				 		initializer='random_normal',
						trainable=True,
						dtype = np.float32,
						name = 'bias'
			)

		#print(list(self.kernel_size + [1, self.depth_multiplier]))
		for c in range(self.num_of_channels):
			self.filters.append(
				self.add_weight(shape = list(self.kernel_size + [1, self.depth_multiplier]),
						 		initializer = 'random_normal',
								trainable = True,
								dtype = np.float32,
								name = 'filter_' + str(c))
			)