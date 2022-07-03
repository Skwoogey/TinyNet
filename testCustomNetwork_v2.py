

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', help='model.py file with model setups')
parser.add_argument('--dataset_code', '-dc', help='dataset code')
parser.add_argument('--save_path', '-sp', help='path where to save h5 trained model')
parser.add_argument('--ghost_ratio', '-gr', help='ghost ratio value', type=int)
parser.add_argument('-bn', help='include to use batch normalization', action='store_true')
parser.add_argument('--batch_size', '-bs', help='batch size for training', type=int, default = 32)
parser.add_argument('--dataset_split', '-ds', help='ratios to split dataset into (train, valid, test)',
						nargs=3, type=float, default = [0.35, 0.35, 0.3])
parser.add_argument('--equalized_split', '-es', help='augment training data', action='store_true')
parser.add_argument('--augment_config', '-ac', help='augment_config.py file with augment setups')
parser.add_argument('--center_loss', '-cl', help='use center loss for training', action='store_true')
parser.add_argument('--center_loss_weight', '-clw', help='use center loss weight', type=float, default=0.9)
parser.add_argument('--no_save', '-ns', help='do not save model', action='store_true')
parser.add_argument('--pretrained_model', '-pm', help='pretrained model to set initial weights with', default=None)
parser.add_argument('--center_loss_type', '-clt', help='type of center loss CL/CCL', default='CL')
parser.add_argument('--verbose', '-v', help='training verbosity', default=1, type = int)
parser.add_argument('--axes_order', '-ao', help='order of sample axes to be created', default='HWDC')

parser.add_argument('--adjust_epochsnum', '-ae', help='adjust epochs num to amount of variants', type=int, default = 1)
parser.add_argument('--custom_prep_flow', '-cpf', help='custom prep flow')

parser.add_argument('--test_saved_transform', '-tst', help='path to saved transform')

args = parser.parse_args()

import tensorflow as tf
import numpy as np
from utils import createTrainDataPerClass, loadData, padImage, createTrainValidTestSplits, \
	getLayersFromModel, printEvaluation, saveTrainInfo, copyPretrainedModel
import sys
import os
import time
from importlib import import_module
from datetime import datetime
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from HSIDatasets import getDataset, getDatasetInfo
from HSICubeBuilder import HSIImageHandler, CoordSys
from HSICubePredictSequencer import HSICubePredictSequencer
from Sequencers import TrainSequencer, PredictSequencer
import matplotlib.pyplot as plt

# IMPORTING MODEL

model_py = args.model.replace('/', '.').replace('\\', '.').replace('.py', '').replace('..', '.')
model_base_path = args.save_path

model_struct = import_module(model_py)
print(model_struct)

window_size = model_struct.window_size
window_depth_required = model_struct.window_depth

image, ground_truth = getDataset(args.dataset_code, bands = window_depth_required)
datasetInfo = getDatasetInfo(args.dataset_code)
print(datasetInfo['shape'])

# CREATING IMAGE HANDLER

HSI_IH = HSIImageHandler(image, ground_truth, window_size, axes_order=args.axes_order)
#HSI_IH.populatePixels(window_size)
num_of_classes = HSI_IH.class_count

# IMPORTING AND APPLYING VARIANTS

print('Applying variants....')
augment_config_py = args.augment_config.replace('/', '.').replace('\\', '.').replace('.py', '').replace('..', '.')
augment_config = import_module(augment_config_py)
augment_config.applyVariants(HSI_IH)

if args.test_saved_transform != None:
	HSI_IH.clear()
	HSI_IH.addSavedTransforms(args.test_saved_transform)
	HSI_IH.populatePixels(False)
    
print('Variants applied')

# CREATING TRAINING SPLITS

if not args.equalized_split:
	dataset_split = [x / sum(args.dataset_split) for x in args.dataset_split]
	train = HSI_IH.getPercentageSplit(dataset_split[0], augment_variant = augment_config.augment_variant, augment_transform = augment_config.augment_transform)

else:
	dataset_split = [x / sum(args.dataset_split[1:]) for x in args.dataset_split[1:]]
	dataset_split = [0] + dataset_split
	train = HSI_IH.getCountSplit(int(args.dataset_split[0]), augment_variant = augment_config.augment_variant, augment_transform = augment_config.augment_transform)
valid = None
if dataset_split[1] != 0:
	valid = HSI_IH.getPercentageSplit(dataset_split[1])
test = HSI_IH.getRemainingSamples()

ones = np.identity(num_of_classes, dtype=np.float32)
train = (train[0], [ones[x] for x in train[1]])
test = (test[0], [ones[x] for x in test[1]])

print("Train split", len(train[0]), len(train[1]))
if valid != None:
    print("Valid split", len(valid[0]), len(valid[1]))
    valid = (valid[0], [ones[x] for x in valid[1]])
print("Test split", len(test[0]), len(test[1]))

# IMPORTING MODEL

sample_shape = train[0][0].shape

model, model_for_save = model_struct.getModel(sample_shape, num_of_classes)
model.summary()

# COPY PRETRAINED MODEL AND PREPARE DATA PIPELINES

if args.pretrained_model != None:
	copyPretrainedModel(model_for_save, args.pretrained_model)

num_of_epochs = model_struct.total_epochs * args.adjust_epochsnum

scheduler = tf.keras.callbacks.LearningRateScheduler(model_struct.scheduler)

if valid != None:
	valid = [np.stack(valid[0], axis = 0), np.array(valid[1])]
	valid = model_struct.prepareData(valid[0], valid[1], num_of_classes)
	valid = (valid[0], valid[1])

# USING CUSTOM PREPROCESSING FLOW

if args.custom_prep_flow == None:
	trainSqncr = TrainSequencer(train[0], train[1], args.batch_size, model_struct.prepareData)
else:
	prep_flow_py = args.custom_prep_flow.replace('/', '.').replace('\\', '.').replace('.py', '').replace('..', '.')
	prep_flow = import_module(prep_flow_py)
	print(train[0][0].shape, train[1][0].shape)
	trainSqncr = prep_flow.getFlow(train[0], train[1], args.batch_size, num_of_classes, model_struct.prepareData)

# TRAINING

train_start = time.time()

print("trainSqncr")
print(trainSqncr)
history = model.fit(
	trainSqncr,
	epochs=num_of_epochs,
	validation_data=valid,
	callbacks=[scheduler],
	verbose=min(args.verbose, 2)
)

train_end = time.time()

# EVALUATION

Sqncr = HSICubePredictSequencer(HSI_IH, augment_variant = augment_config.augment_variant, augment_transform = augment_config.augment_transform, batch_size = args.batch_size)
y_true = np.array([x.label for x in HSI_IH.labeled_pixels])
y_pred = model_for_save.predict(Sqncr, verbose=min(args.verbose, 1))

predict_end = time.time()

# UNANIMOUSLY CLASSIFIED TEST - DIDN'T USE

y_pred = np.argmax(y_pred, axis = 1)
y_pred = np.reshape(y_pred, (-1, Sqncr.samples_per_pixel))
y_pred_default = y_pred[:, 0]
#print(y_pred.shape)
y_true_all = [0] * num_of_classes
y_false_all = [0] * num_of_classes
num_UC = 0
num_UCT = 0
num_UCF = 0
for i in range(len(y_true)):
	if all(y_pred[i] == y_pred[i][0]):
		if y_true[i] == y_pred[i][0]:
			y_true_all[y_true[i]] += 1
			num_UCT += 1
		else:
			y_false_all[y_true[i]] += 1
			num_UCF += 1
		num_UC += 1
y_pred = [np.bincount(x).argmax() for x in y_pred]
print(list(zip(y_true_all, y_false_all)))
print('num_UCT \\ num_UCF:', num_UCT, num_UCF)
print('num_UC:', num_UC)
print('UC_accuracy:', num_UCT / num_UC)
print('### CONJOINED PREDICTION ###')
printEvaluation(y_pred, y_true)

print('### REGULAR PREDICTION ###')
printEvaluation(y_pred_default, y_true)

print("Training Time:", train_end - train_start)
print("Prediction Time:", predict_end - train_end)

# SAVING

if args.no_save:
	exit()

#ns_base_path = model_base_path + os.path.sep + model_py.split('.')[-1]
ns_base_path = model_base_path
ns_base_path = ns_base_path + os.path.sep + datetime.now().strftime("%m-%d_%H-%M-%S")

if not os.path.exists(ns_base_path):
	os.makedirs(ns_base_path)

model_for_save.save(ns_base_path + os.path.sep + 'model_' + model_py.split('.')[-1]+ '.h5')

saveTrainInfo(history, model_for_save, ns_base_path + os.path.sep + "result.xlsx")

pgt = HSI_IH.gt.copy()
pxls = HSI_IH.labeled_pixels

for t, p in zip(pxls, y_pred):
    pgt[t.coordinates[0], t.coordinates[1]] = p + 1

plt.axis('off')    
plt.imshow(pgt, interpolation="none", cmap='jet')
plt.savefig(ns_base_path + os.path.sep + "result.png", dpi='figure', bbox_inches='tight', pad_inches=0)
#plt.show()