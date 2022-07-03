import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
import numpy as np

from resnet3d import Resnet3DBuilder 


Name = 'base_model'
window_size = 9
window_depth = 20

def getModel(input_shape, num_classes):
	
	
	
	model = Resnet3DBuilder.build(input_shape, num_classes, (3, 4), base_filters = 32)
	#optimizer = tfa.optimizers.SGD(weight_decay = 1e-5, momentum=0.9)
	optimizer = tf.keras.optimizers.SGD(momentum=0.9, nesterov=False)
	#optimizer = tf.keras.optimizers.Adam()
	loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
	model.compile(
		optimizer = optimizer,
		loss = loss,
		metrics = [tf.keras.metrics.CategoricalAccuracy()]
	)

	return model, model

def prepareData(data, labels, num_classes):
	data_one_hot = np.zeros(shape = (data.shape[0], num_classes))
	data_one_hot[np.arange(data.shape[0]), labels.reshape((labels.shape[0],))] = 1.0
	inputs = [data]
	outputs = [data_one_hot]
	return inputs, outputs
total_epochs = 40
def scheduler(epoch):
	lr = 0.001
	print(lr)
	return lr