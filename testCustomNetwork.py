#!python3.8

import argparse

# arg1 - image_file
# arg2 - ground_truth_file
# arg3 - size of the window
# arg4 - path to save model

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', help='model.py file with model setups')
parser.add_argument('--image', '-i', help='image file')
parser.add_argument('--ground_truth', '-gt', help='ground truth file')
parser.add_argument('--save_path', '-sp', help='path where to save h5 trained model')
#parser.add_argument('--ghost_ratio', '-gr', help='ghost ratio value', type=int)
#parser.add_argument('-bn', help='include to use batch normalization', action='store_true')
parser.add_argument('--batch_size', '-bs', help='batch size for training', type=int, default = 32)
parser.add_argument('--dataset_split', '-ds', help='ratios to split dataset into (train, valid, test)',
						nargs=3, type=float, default = [0.05, 0.05, 0.9])
parser.add_argument('--equal_splits', '-es', help='use equal number of samples per class instead of weighted number of samples', action='store_true')
parser.add_argument('--augment', '-a', help='augment training data', action='store_true')
parser.add_argument('--normalize', '-n', help='normalize input image', action='store_true')
#parser.add_argument('--center_loss', '-cl', help='use center loss for training', action='store_true')
#parser.add_argument('--center_loss_weight', '-clw', help='use center loss weight', type=float, default=0.9)
#parser.add_argument('--dpr', help='dropout rates', type=float, nargs=2)

args = parser.parse_args()

import tensorflow as tf
import numpy as np
from network import makeCustomFast3DCNN
from utils import createTrainDataPerClass, loadData, padImage,\
	createTrainValidTestSplits, printEvaluation, saveTrainInfo
import sys
import os
from importlib import import_module
from datetime import datetime
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from splitGenerator import splitGenerator
from HSICubeBuilder import HSIImageHandler, CoordSys
from Sequencers import TrainSequencer, PredictSequencer

data_file = args.image
gt_file = args.ground_truth
model_py = args.model.replace('/', '.').replace('\\', '.').replace('.py', '').replace('..', '.')
model_base_path = args.save_path

model_struct = import_module(model_py)
print(model_struct)

window_size = model_struct.window_size

image, ground_truth =loadData(data_file, gt_file)
image = np.ascontiguousarray(image).astype(np.float64)
if args.normalize:
	band_max = image.max(axis = (0, 1))
	image = image / band_max
print(image.shape, image.dtype)
print(image.flags)
window_depth = image.shape[2]

'''
padded_image = padImage(image, window_size)
print(padded_image.shape)

data_per_class = createTrainDataPerClass(padded_image, ground_truth, window_size)
num_of_classes = len(data_per_class)

splitGen = splitGenerator(data_per_class)

if not args.equal_splits:
	args.dataset_split = [x / sum(args.dataset_split) for x in args.dataset_split]
	train_data, train_labels = splitGen.getPercentageSplit(args.dataset_split[0], args.augment)
else:
	args.dataset_split[1] = args.dataset_split[1] / (args.dataset_split[1] + args.dataset_split[2])
	args.dataset_split[2] = args.dataset_split[2] / (args.dataset_split[1] + args.dataset_split[2])
	train_data, train_labels = splitGen.getCountSplit(args.dataset_split[0], args.augment)
valid_data, valid_labels = splitGen.getPercentageSplit(args.dataset_split[1], args.augment)
test_data, test_labels = splitGen.getRemainingSamples()
'''
HSI_IH = HSIImageHandler(image, ground_truth, window_size)
num_of_classes = HSI_IH.class_count

if not args.equal_splits:
	args.dataset_split = [x / sum(args.dataset_split) for x in args.dataset_split]
	train_data, train_labels = HSI_IH.getPercentageSplit(args.dataset_split[0], args.augment)
else:
	args.dataset_split[1] = args.dataset_split[1] / (args.dataset_split[1] + args.dataset_split[2])
	args.dataset_split[2] = args.dataset_split[2] / (args.dataset_split[1] + args.dataset_split[2])
	train_data, train_labels = HSI_IH.getCountSplit(int(args.dataset_split[0]), args.augment)
valid_data, valid_labels = HSI_IH.getPercentageSplit(args.dataset_split[1], False)
test_data, test_labels = HSI_IH.getRemainingSamples()
print("Train split", len(train_data), len(train_labels))
print("Test split", len(test_data), len(test_labels))

print(len(train_data[0]))
sample_shape = train_data[0].shape

model, model_for_save = model_struct.getModel(sample_shape, num_of_classes)
model.summary()

num_of_epochs = model_struct.total_epochs

scheduler = tf.keras.callbacks.LearningRateScheduler(model_struct.scheduler)

train_data = np.stack(train_data, axis = 0)
train_labels = np.array(train_labels)

valid_data = np.stack(valid_data, axis = 0)
valid_labels = np.array(valid_labels)
print(valid_data.shape)

train_split = model_struct.prepareData(train_data, train_labels)
valid_split = model_struct.prepareData(valid_data, valid_labels)
#print(len(train_split))
#print(train_split[0].shape)
#print(train_split[1].shape)

history = model.fit(
	x=train_split[0],
	y=train_split[1],
	batch_size=args.batch_size,
	epochs=num_of_epochs,
	validation_data = valid_split,
	callbacks=[scheduler]
)

Sqncr = PredictSequencer(test_data, 64)
y_pred = model_for_save.predict(Sqncr, verbose = 1)
y_pred = np.argmax(y_pred, axis = 1)

y_true = np.array(test_labels)

printEvaluation(y_pred, y_true)

ns_base_path = model_base_path + os.path.sep + model_py.split('.')[-1]
ns_base_path = ns_base_path + os.path.sep + datetime.now().strftime("%m-%d_%H-%M-%S")

if not os.path.exists(ns_base_path):
	os.makedirs(ns_base_path)

model_for_save.save(ns_base_path + os.path.sep + 'model_' + model_py.split('.')[-1]+ '.h5')
ns_base_path = ns_base_path + os.path.sep + "result.xlsx"

saveTrainInfo(history, model_for_save, ns_base_path)