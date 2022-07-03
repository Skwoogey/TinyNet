import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.utils import Sequence
import math
import random

class TrainSequencer(Sequence):
	def __init__(self, X, Y, bs, prepareDataFunc):
		self.data = list(zip(X, Y))
		self.num_classes = len(Y[0])
		random.shuffle(self.data)
		self.bs = bs
		self.PDF = prepareDataFunc

	def __len__(self):
		return math.ceil(len(self.data) / self.bs)

	def __getitem__(self, idx):
		batch_x = []
		batch_y = []

		batch = self.data[idx * self.bs : (idx + 1) * self.bs]
		for x, y in batch:
			batch_x.append(x)
			batch_y.append(y)

		batch_x = np.stack(batch_x, axis = 0)
		batch_y = np.array(batch_y, dtype = int)
		#print(batch_x.shape, batch_y.shape)

		batch_x, batch_y = self.PDF(batch_x, batch_y, self.num_classes)
		#print(batch_x[0].shape, batch_x[1].shape, batch_y[0].shape, batch_y[1].shape)
		return batch_x, batch_y

	def on_epoch_end(self):
		random.shuffle(self.data)


class PredictSequencer(Sequence):
	def __init__(self, X, bs):
		self.X = X
		self.bs = bs

	def __len__(self):
		return math.ceil(len(self.X) / self.bs)

	def __getitem__(self, idx):
		batch_x = self.X[idx * self.bs : (idx + 1) * self.bs]

		batch_x = np.stack(batch_x, axis = 0)
		
		return batch_x